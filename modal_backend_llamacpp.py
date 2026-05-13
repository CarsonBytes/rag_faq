import modal
import os
import asyncio
import logging
import time
import threading

# All model files live inside the image — no volume needed for models.
MODEL_CACHE_DIR = "/model-cache"
GGUF_PATH = f"{MODEL_CACHE_DIR}/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"
EMBED_CACHE_DIR = f"{MODEL_CACHE_DIR}/sentence_transformers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

INDEX_VOL_PATH = "/index_store"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Image build helpers (run once, baked into the image layer) ----------

def download_all_models():
    """
    Pre-download every model file into the image so cold starts never
    touch the network.  Runs on a CPU worker during `modal deploy`.
    """
    import os
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    # --- GGUF (Qwen 2.5 3B) ---
    gguf_dir = f"{MODEL_CACHE_DIR}/gguf"
    os.makedirs(gguf_dir, exist_ok=True)
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir=gguf_dir,
        local_dir_use_symlinks=False,
    )
    print(f"GGUF downloaded → {gguf_dir}")

    # --- Embedding model (BGE-small) ---
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = EMBED_CACHE_DIR
    SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Embedding model downloaded → {EMBED_CACHE_DIR}")


def build_llama_cpp_cuda():
    """Compile llama-cpp-python with CUDA support. Must run on a GPU worker."""
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
    # Download models into the image BEFORE setting offline mode.
    .run_function(download_all_models)
    # Lock HF Hub to offline so every subsequent container start (including
    # snapshot creation and restoration) loads from disk with zero HTTP calls.
    .env({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SENTENCE_TRANSFORMERS_HOME": EMBED_CACHE_DIR,
    })
    # Build llama-cpp-python with CUDA on a GPU worker.
    .run_function(build_llama_cpp_cuda, gpu="T4")
    # Install the llama-index wrapper as a separate committed layer so it
    # doesn't overwrite the CUDA-compiled llama-cpp-python binary.
    .pip_install("llama-index-llms-llama-cpp", extra_options="--no-deps")
)

index_volume = modal.Volume.from_name("rag-index-store", create_if_missing=True)
app = modal.App(name="llamacpp-rag-backend", image=image)


# ---------- Lazy singletons ----------

_llm = None
_embed_model = None
_index_cache = None
_model_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

    if not os.path.exists(GGUF_PATH):
        raise RuntimeError(
            f"GGUF not found at {GGUF_PATH}. Re-run `modal deploy` to rebuild the image."
        )

    t0 = time.time()
    _llm = LlamaCPP(
        model_path=GGUF_PATH,
        temperature=0.1,
        max_new_tokens=128,
        context_window=2048,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        model_kwargs={"n_gpu_layers": -1, "logits_all": False},
        verbose=False,
    )
    logger.info(f"LLM loaded in {time.time()-t0:.1f}s (n_gpu_layers=-1).")
    return _llm


def get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # Point explicitly to the baked-in cache so no network lookup occurs.
    _embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=EMBED_CACHE_DIR,
    )
    logger.info("Embedding model loaded.")
    return _embed_model


def get_or_load_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    from llama_index.core import StorageContext, load_index_from_storage
    if not os.path.exists(f"{INDEX_VOL_PATH}/docstore.json"):
        return None
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_VOL_PATH)
    _index_cache = load_index_from_storage(storage_context)
    logger.info("Index loaded from storage.")
    return _index_cache


def _truncate_at_prompt_markers(text: str) -> str:
    """Strip any leaked prompt-template text the model appended to its answer."""
    for marker in ("[INST]", "<<SYS>>", "<</SYS>>", "<</SYS"):
        if marker in text:
            text = text[:text.index(marker)]
    return text.strip()


# ---------- Modal class ----------

@app.cls(
    gpu="T4",
    volumes={INDEX_VOL_PATH: index_volume},
    max_containers=1,
    min_containers=0,
    scaledown_window=120,
    timeout=600,
    # CPU-only memory snapshot: Modal restores the Python process state instantly
    # on cold starts (no re-importing packages, no re-initialising embed model).
    # GPU memory is NOT snapshotted — experimental GPU snapshots caused SIGSEGV
    # (exit 139) with llama-cpp's CUDA context and have been removed.
    enable_memory_snapshot=True,
)
class RAGBackend:

    @modal.enter(snap=True)
    def snapshot_cpu_state(self):
        """
        Runs ONCE before the memory snapshot is taken.
        Captures only lightweight CPU state (~200 MB total):
          - all Python package imports
          - embed model weights (22 MB CPU tensors)
        The GGUF (2.1 GB) is intentionally excluded — including it bloated the
        snapshot to 2.1 GB, making creation + restore slower than a plain disk
        read.  GGUF is loaded from the baked-in image path in load_gpu() below.
        """
        import nest_asyncio
        nest_asyncio.apply()
        get_embed_model()
        logger.info("CPU snapshot ready: imports + embed model in RAM (~200 MB).")

    @modal.enter(snap=False)
    def load_gpu(self):
        """
        Runs on every cold start including snapshot restores.
        Loads GGUF from image disk into GPU VRAM (~7 s).
        Small snapshot (~200 MB) restores in <1 s, so total cold start is
        ~8 s vs ~33 s without any snapshot.
        """
        from llama_index.core import Settings
        Settings.llm = get_llm()
        Settings.embed_model = get_embed_model()  # already cached, instant
        logger.info("GPU model loaded. Container ready.")

    # ---------- Health ----------

    @modal.fastapi_endpoint(method="GET", label="health")
    def health(self):
        index_exists = os.path.exists(f"{INDEX_VOL_PATH}/docstore.json")
        filename = None
        char_count = None
        if index_exists:
            try:
                index = get_or_load_index()
                if index and index.docstore.docs:
                    doc_id = list(index.docstore.docs.keys())[0]
                    doc = index.docstore.get_document(doc_id)
                    if doc:
                        filename = doc.metadata.get("original_filename", doc_id)
                        char_count = doc.metadata.get("char_count")
            except Exception:
                pass
        return {
            "status": "ok",
            "index_exists": index_exists,
            "indexed_filename": filename,
            "indexed_char_count": char_count,
        }

    # ---------- Upload ----------

    @modal.fastapi_endpoint(method="POST", label="upload")
    async def upload(self, payload: dict):
        """Accepts {"text": "...", "filename": "doc.txt"}"""
        global _index_cache

        text = (payload.get("text") or "").strip()
        filename = (payload.get("filename") or "document.txt").strip()

        if not text:
            return {"status": "error", "message": "No text provided."}
        if len(text) > 20_000_000:
            return {"status": "error", "message": "Text exceeds 20 MB limit."}

        def _do_upload():
            global _index_cache
            from llama_index.core import Settings, Document, VectorStoreIndex
            llm = get_llm()
            embed_model = get_embed_model()
            Settings.llm = llm
            Settings.embed_model = embed_model
            doc = Document(
                text=text,
                id_=filename,
                metadata={
                    "original_filename": filename,
                    "char_count": len(text),
                },
            )
            index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)
            index.storage_context.persist(persist_dir=INDEX_VOL_PATH)
            _index_cache = index
            logger.info(f"Indexed '{filename}' ({len(text):,} chars)")
            return len(text)

        try:
            with _model_lock:
                char_count = await asyncio.to_thread(_do_upload)
            await index_volume.commit.aio()
            return {
                "status": "success",
                "filename": filename,
                "char_count": char_count,
                "message": f"Indexed {char_count:,} characters from '{filename}'.",
            }
        except Exception as e:
            logger.exception("Upload failed")
            return {"status": "error", "message": str(e)}

    # ---------- Query ----------

    @modal.fastapi_endpoint(method="POST", label="query")
    async def query(self, payload: dict):
        """Accepts {"question": "..."}"""
        question = (payload.get("question") or "").strip()
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
                "answer": "No document has been uploaded yet. Please upload a .txt file first.",
            }

        def _do_query():
            from llama_index.core import Settings
            Settings.llm = get_llm()
            query_engine = index.as_query_engine(
                similarity_top_k=2,
                response_mode="compact",
            )
            t0 = time.time()
            response = query_engine.query(question)
            elapsed = time.time() - t0
            answer = _truncate_at_prompt_markers(str(response))
            logger.info(f"Query answered in {elapsed:.1f}s: {answer[:120]}...")
            return answer, round(elapsed, 2)

        try:
            with _model_lock:
                answer, elapsed = await asyncio.to_thread(_do_query)

            if not answer:
                return {
                    "status": "ok",
                    "answer": "I couldn't find a relevant answer. Please try rephrasing your question.",
                    "elapsed": elapsed,
                }
            return {"status": "ok", "answer": answer, "elapsed": elapsed}
        except Exception as e:
            logger.exception("Query failed")
            return {"status": "error", "answer": f"Backend error: {str(e)}"}
