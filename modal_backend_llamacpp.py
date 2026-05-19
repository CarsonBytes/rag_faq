"""
Modal RAG backend — single unified vector index.

All uploaded documents live in one VectorStoreIndex persisted to /index_store.
Queries search across all documents by default. An optional `doc_id` (mapped
to the metadata field `original_filename`) scopes retrieval to a single doc.
"""
import modal
import os
import re
import asyncio
import logging
import time
import threading

MODEL_CACHE_DIR = "/model-cache"
GGUF_PATH = f"{MODEL_CACHE_DIR}/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"
EMBED_CACHE_DIR = f"{MODEL_CACHE_DIR}/sentence_transformers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Single unified index lives here.
INDEX_DIR = "/index_store"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Image build helpers ----------

def download_all_models():
    import os
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    gguf_dir = f"{MODEL_CACHE_DIR}/gguf"
    os.makedirs(gguf_dir, exist_ok=True)
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir=gguf_dir,
        local_dir_use_symlinks=False,
    )
    print(f"GGUF downloaded → {gguf_dir}")

    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = EMBED_CACHE_DIR
    SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Embedding model downloaded → {EMBED_CACHE_DIR}")


def build_llama_cpp_cuda():
    import subprocess
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"
    env["FORCE_CMAKE"] = "1"
    subprocess.check_call(
        [
            "pip", "install", "llama-cpp-python",
            "--force-reinstall", "--no-cache-dir", "--no-binary", "llama-cpp-python",
        ],
        env=env,
    )


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake", "curl", "wget", "patchelf")
    .pip_install("torch", extra_options="--index-url https://download.pytorch.org/whl/cpu")
    .pip_install(
        "llama-index",
        "llama-index-embeddings-huggingface",
        "huggingface_hub",
        "sentence-transformers",
        "fastapi[standard]",
        "python-multipart",
        "nest-asyncio",
    )
    .run_function(download_all_models)
    .env({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SENTENCE_TRANSFORMERS_HOME": EMBED_CACHE_DIR,
    })
    .run_function(build_llama_cpp_cuda, gpu="T4")
    .pip_install("llama-index-llms-llama-cpp", extra_options="--no-deps")
)

index_volume = modal.Volume.from_name("rag-index-store", create_if_missing=True)
app = modal.App(name="llamacpp-rag-backend", image=image)


# ---------- Helpers ----------

def _sanitize_doc_id(filename: str) -> str:
    """Map a filename to a stable, safe ID used as the Document.id_ and metadata key."""
    safe = re.sub(r"[^\w\-\.]", "_", filename or "document.txt")
    return safe[:120] or "document.txt"


def _has_persisted_index() -> bool:
    return os.path.exists(os.path.join(INDEX_DIR, "docstore.json"))


# ---------- Lazy singletons ----------

_llm = None
_embed_model = None
_index = None        # Single unified VectorStoreIndex
_model_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

    if not os.path.exists(GGUF_PATH):
        raise RuntimeError(f"GGUF not found at {GGUF_PATH}. Re-run `modal deploy`.")

    t0 = time.time()
    _llm = LlamaCPP(
        model_path=GGUF_PATH,
        temperature=0.1,
        max_new_tokens=256,
        context_window=4096,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        model_kwargs={"n_gpu_layers": -1, "logits_all": False},
        verbose=False,
    )
    logger.info(f"LLM loaded in {time.time()-t0:.1f}s.")
    return _llm


def get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    _embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=EMBED_CACHE_DIR,
    )
    logger.info("Embedding model loaded.")
    return _embed_model


def get_or_load_index():
    """Return the single unified index. Loads from disk on first access."""
    global _index
    if _index is not None:
        return _index
    if not _has_persisted_index():
        return None
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    _index = load_index_from_storage(storage_context)
    logger.info("Unified index loaded from storage.")
    return _index


def _list_indexed_filenames(index) -> list:
    """Return sorted, deduped list of original_filename values present in the docstore."""
    if index is None or not getattr(index, "docstore", None):
        return []
    seen = set()
    for doc_id, doc in index.docstore.docs.items():
        meta = getattr(doc, "metadata", None) or {}
        name = meta.get("original_filename") or doc_id
        seen.add(name)
    return sorted(seen)


def _truncate_at_prompt_markers(text: str) -> str:
    for marker in ("[/INST]", "[INST]", "<<SYS>>", "<</SYS>>", "<</SYS", "</s>", "<|im_end|>", "<|im_start|>"):
        if marker in text:
            text = text[:text.index(marker)]
    return text.strip()


# ---------- Modal class ----------

@app.cls(
    gpu="T4",
    volumes={INDEX_DIR: index_volume},
    max_containers=1,
    min_containers=0,
    scaledown_window=120,
    timeout=600,
    enable_memory_snapshot=True,
)
class RAGBackend:

    @modal.enter(snap=True)
    def snapshot_cpu_state(self):
        import nest_asyncio
        nest_asyncio.apply()
        get_embed_model()
        logger.info("CPU snapshot ready: imports + embed model in RAM (~200 MB).")

    @modal.enter(snap=False)
    def load_gpu(self):
        from llama_index.core import Settings
        Settings.llm = get_llm()
        Settings.embed_model = get_embed_model()
        logger.info("GPU model loaded. Container ready.")

    # ---------- Health ----------

    @modal.fastapi_endpoint(method="GET", label="health")
    def health(self):
        docs = []
        try:
            index = get_or_load_index()
            docs = _list_indexed_filenames(index)
        except Exception:
            logger.exception("health: failed to enumerate")
        return {
            "status": "ok",
            "index_exists": len(docs) > 0,
            "indexed_docs": docs,
        }

    # ---------- List ----------

    @modal.fastapi_endpoint(method="GET", label="list")
    def list_docs(self):
        try:
            index = get_or_load_index()
            return {"status": "ok", "docs": _list_indexed_filenames(index)}
        except Exception as e:
            logger.exception("list_docs failed")
            return {"status": "error", "docs": [], "message": str(e)}

    # ---------- Delete ----------

    @modal.fastapi_endpoint(method="POST", label="delete")
    async def delete(self, payload: dict):
        """
        Accepts {"filename": "doc.txt"}.
        Removes the document from the unified index.
        """
        global _index

        filename = (payload.get("filename") or "").strip()
        if not filename:
            return {"status": "error", "message": "No filename provided."}
        doc_id = _sanitize_doc_id(filename)

        def _do_delete():
            global _index
            from llama_index.core import StorageContext, load_index_from_storage

            if not _has_persisted_index():
                return False, []

            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            index = load_index_from_storage(storage_context)

            # Find the doc_id whose original_filename matches the requested filename.
            target_id = None
            for did, doc in index.docstore.docs.items():
                meta = getattr(doc, "metadata", None) or {}
                if meta.get("original_filename") == filename or did == doc_id:
                    target_id = did
                    break

            if not target_id:
                return False, _list_indexed_filenames(index)

            try:
                index.delete_ref_doc(target_id, delete_from_docstore=True)
            except Exception:
                logger.exception("delete_ref_doc failed for '%s'", target_id)
                return False, _list_indexed_filenames(index)

            index.storage_context.persist(persist_dir=INDEX_DIR)
            _index = index
            remaining = _list_indexed_filenames(index)
            logger.info("Deleted '%s'. Remaining docs: %d", filename, len(remaining))
            return True, remaining

        try:
            with _model_lock:
                deleted, remaining = await asyncio.to_thread(_do_delete)
            await index_volume.commit.aio()
            return {
                "status": "success" if deleted else "not_found",
                "filename": filename,
                "indexed_docs": remaining,
                "message": (
                    f"Deleted '{filename}'." if deleted
                    else f"'{filename}' not found in the index."
                ),
            }
        except Exception as e:
            logger.exception("Delete failed")
            return {"status": "error", "message": str(e)}

    # ---------- Upload ----------

    @modal.fastapi_endpoint(method="POST", label="upload")
    async def upload(self, payload: dict):
        """
        Accepts {"text": "...", "filename": "doc.txt"}.
        Adds the document to the unified index. Re-uploading the same filename
        replaces the previous version of that document.
        """
        global _index

        text = (payload.get("text") or "").strip()
        filename = (payload.get("filename") or "document.txt").strip()
        doc_id = _sanitize_doc_id(filename)

        if not text:
            return {"status": "error", "message": "No text provided."}
        if len(text) > 20_000_000:
            return {"status": "error", "message": "Text exceeds 20 MB limit."}

        def _do_upload():
            global _index
            from llama_index.core import (
                Settings, Document, VectorStoreIndex,
                StorageContext, load_index_from_storage,
            )

            Settings.llm = get_llm()
            Settings.embed_model = get_embed_model()

            doc = Document(
                text=text,
                id_=doc_id,
                metadata={
                    "original_filename": filename,
                    "doc_id": doc_id,
                    "char_count": len(text),
                },
            )

            if _has_persisted_index():
                storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
                index = load_index_from_storage(storage_context)
                # Remove prior version of this doc (if any) before re-inserting.
                try:
                    if doc_id in index.docstore.docs:
                        index.delete_ref_doc(doc_id, delete_from_docstore=True)
                except Exception:
                    logger.exception("Failed to delete prior version of '%s' (continuing)", doc_id)
                index.insert(doc)
            else:
                index = VectorStoreIndex.from_documents([doc])

            os.makedirs(INDEX_DIR, exist_ok=True)
            index.storage_context.persist(persist_dir=INDEX_DIR)
            _index = index

            all_docs = _list_indexed_filenames(index)
            logger.info(
                "Indexed '%s' (%d chars). Total docs now: %d",
                doc_id, len(text), len(all_docs),
            )
            return len(text), all_docs

        try:
            with _model_lock:
                char_count, all_docs = await asyncio.to_thread(_do_upload)
            await index_volume.commit.aio()
            return {
                "status": "success",
                "filename": filename,
                "doc_id": doc_id,
                "char_count": char_count,
                "indexed_docs": all_docs,
                "message": f"Indexed {char_count:,} characters from '{filename}'.",
            }
        except Exception as e:
            logger.exception("Upload failed")
            return {"status": "error", "message": str(e)}

    # ---------- Query ----------

    @modal.fastapi_endpoint(method="POST", label="query")
    async def query(self, payload: dict):
        """
        Accepts {"question": "...", "doc_ids": ["<filename>", ...], "doc_id": "<filename>"}.
        - With one or more doc_ids: retrieval is scoped to those documents (OR filter).
        - Empty/missing: retrieval searches across ALL indexed documents.
        - `doc_id` (singular) is kept for backward compatibility.
        """
        question = (payload.get("question") or "").strip()

        # Build the filter list from doc_ids (preferred) and/or doc_id (legacy).
        raw_ids = payload.get("doc_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        legacy_single = (payload.get("doc_id") or "").strip()
        if legacy_single and legacy_single not in raw_ids:
            raw_ids = list(raw_ids) + [legacy_single]
        doc_filters = [str(d).strip() for d in raw_ids if d and str(d).strip()]

        if not question:
            return {"status": "error", "answer": "No question provided."}
        if len(question) > 2000:
            return {"status": "error", "answer": "Question exceeds 2,000 character limit."}

        def _prepare_embed():
            from llama_index.core import Settings
            Settings.embed_model = get_embed_model()

        await asyncio.to_thread(_prepare_embed)

        index = get_or_load_index()
        if index is None:
            return {
                "status": "no_index",
                "answer": "No document has been indexed yet. Please upload a document first.",
            }

        def _do_query():
            from llama_index.core import Settings, PromptTemplate
            from llama_index.core.vector_stores import (
                MetadataFilter, MetadataFilters, FilterCondition,
            )

            Settings.llm = get_llm()
            concise_qa = PromptTemplate(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Answer the question in 2-4 sentences. Be direct and concise. "
                "If the answer requires information from multiple documents, synthesise it. "
                "No preamble, no restating the question, no closing remarks.\n"
                "Question: {query_str}\n"
                "Answer: "
            )

            engine_kwargs = dict(
                similarity_top_k=5,           # higher so multi-doc questions can pull from several files
                response_mode="compact",
                text_qa_template=concise_qa,
            )
            if doc_filters:
                filters = [MetadataFilter(key="original_filename", value=d) for d in doc_filters]
                engine_kwargs["filters"] = MetadataFilters(
                    filters=filters,
                    condition=FilterCondition.OR if len(filters) > 1 else FilterCondition.AND,
                )

            query_engine = index.as_query_engine(**engine_kwargs)
            t0 = time.time()
            response = query_engine.query(question)
            elapsed = time.time() - t0
            answer = _truncate_at_prompt_markers(str(response))

            # Collect which docs contributed to the answer.
            sources = []
            try:
                seen = set()
                for node in getattr(response, "source_nodes", []) or []:
                    meta = getattr(node, "metadata", None) or {}
                    name = meta.get("original_filename")
                    if name and name not in seen:
                        seen.add(name)
                        sources.append(name)
            except Exception:
                logger.exception("Failed to collect source filenames")

            logger.info(
                "Query answered in %.1fs (filters=%s, sources=%s): %s...",
                elapsed, doc_filters, sources, answer[:120],
            )
            return answer, round(elapsed, 2), sources

        try:
            with _model_lock:
                answer, elapsed, sources = await asyncio.to_thread(_do_query)

            if not answer:
                return {
                    "status": "ok",
                    "answer": "I couldn't find a relevant answer. Please try rephrasing your question.",
                    "elapsed": elapsed,
                    "sources": sources,
                }
            return {
                "status": "ok",
                "answer": answer,
                "elapsed": elapsed,
                "sources": sources,
                "doc_filters": doc_filters,
            }
        except Exception as e:
            logger.exception("Query failed")
            return {"status": "error", "answer": f"Backend error: {str(e)}"}
