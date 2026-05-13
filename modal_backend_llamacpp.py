import modal
import os
import nest_asyncio
import asyncio
import logging
import time
import threading

MODEL_VOL_PATH = "/models"
INDEX_VOL_PATH = "/index_store"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_llama_cpp_cuda():
    import subprocess
    # The nvidia/cuda base image may set CC=clang, but we have gcc/g++.
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    # Isolate CMAKE_ARGS so it doesn't leak into unrelated builds (e.g. patchelf).
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"
    env["FORCE_CMAKE"] = "1"
    # Force-reinstall so the CUDA-enabled build overwrites any generic wheel.
    # Use --no-binary llama-cpp-python only (not :all:) so deps use prebuilt wheels.
    subprocess.check_call(
        [
            "pip", "install", "llama-cpp-python",
            "--force-reinstall", "--no-cache-dir", "--no-binary", "llama-cpp-python"
        ],
        env=env,
    )
    # Install the llama-index wrapper now that llama-cpp-python is built.
    subprocess.check_call([
        "pip", "install", "llama-index-llms-llama-cpp", "--no-cache-dir"
    ])

def download_models():
    """Pre-download embedding model weights into image cache.
    The 2 GB GGUF is loaded at runtime into GPU memory and preserved
    via Modal's memory snapshot — no volume needed."""
    from sentence_transformers import SentenceTransformer
    # 22M model, loads in <1s from disk during snapshot creation.
    _ = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Embedding model pre-downloaded")

# Use NVIDIA CUDA devel image so nvcc is available during image build.
# Modal docs explicitly recommend this for libraries that need CUDA toolkit.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])  # removes chatty prints on entry
    .apt_install("git", "build-essential", "cmake", "curl", "wget", "patchelf")
    # Install CPU-only torch FIRST — embeddings (bge-small-en-v1.5) are tiny and fast
    # on CPU.  Avoids NCCL symbol conflicts with nvidia/cuda:12.4.0-devel system libs.
    .pip_install(
        "torch",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "llama-index",
        "llama-index-embeddings-huggingface",
        "huggingface_hub",
        "sentence-transformers",  # needed for download_models
        "fastapi[standard]",
        "python-multipart",
        "nest-asyncio",
    )
    .run_function(download_models)
    .run_function(build_llama_cpp_cuda, gpu="T4")
)

index_volume = modal.Volume.from_name(
    "rag-index-store", create_if_missing=True)

app = modal.App(name="llamacpp-rag-backend", image=image)

# ---------- 惰性加载：启动时不加载模型 ----------
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

    model_path = f"{MODEL_VOL_PATH}/qwen2.5-3b-instruct-q4_k_m.gguf"
    if not os.path.exists(model_path):
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            local_dir=MODEL_VOL_PATH,
            local_dir_use_symlinks=False,
        )

    _llm = LlamaCPP(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=128,          # reduced for faster responses
        context_window=4096,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        model_kwargs={
            "n_gpu_layers": -1,    # offload all layers to GPU
            "logits_all": False,
        },
        verbose=False,
    )
    # Verify GPU offload actually happened
    try:
        n_gpu = _llm._model.n_gpu_layers
        logger.info(f"LLM loaded successfully. GPU layers offloaded: {n_gpu}")
    except Exception:
        logger.warning("Could not verify GPU offload — may be running on CPU!")
    return _llm


def get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    logger.info("Embedding model loaded successfully.")
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


@app.cls(
    gpu="T4",
    volumes={INDEX_VOL_PATH: index_volume},
    max_containers=1,
    min_containers=0,
    scaledown_window=120,
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class RAGBackend:
    """Modal class with GPU memory snapshot for instant cold starts."""

    @modal.enter(snap=True)
    def load(self):
        """Preload models into GPU/RAM once; Modal snapshots the memory state."""
        import nest_asyncio
        nest_asyncio.apply()
        from llama_index.core import Settings
        Settings.llm = get_llm()
        Settings.embed_model = get_embed_model()
        logger.info(
            f"Snapshot-ready: llm={type(Settings.llm).__name__}, "
            f"embed={type(Settings.embed_model).__name__}"
        )

    @modal.fastapi_endpoint(method="GET", label="health")
    def health(self):
        index_file_exists = os.path.exists(f"{INDEX_VOL_PATH}/docstore.json")
        filename = None
        if index_file_exists:
            try:
                index = get_or_load_index()
                if index and index.docstore.docs:
                    # Try to get original filename from metadata first
                    doc_id = list(index.docstore.docs.keys())[0]
                    doc = index.docstore.get_document(doc_id)
                    filename = doc.metadata.get("original_filename", doc_id) if doc else doc_id
            except Exception:
                pass
        return {"status": "ok", "message": "Backend is alive", "index_exists": index_file_exists, "indexed_filename": filename}

    @modal.fastapi_endpoint(method="GET", label="debug")
    def debug(self):
        """Inspect current Settings to verify embed model is not OpenAI."""
        from llama_index.core import Settings as S
        return {
            "llm_type": type(S.llm).__name__ if S.llm else None,
            "embed_type": type(S.embed_model).__name__ if S.embed_model else None,
            "embed_module": str(S.embed_model.__class__.__module__) if S.embed_model else None,
        }

    # ---------- 上传文档 ----------
    @modal.fastapi_endpoint(method="POST", label="upload")
    async def upload(self, payload: dict):
        """POST JSON body: {"text": "...", "filename": "doc.txt"}"""
        global _index_cache
        text = payload.get("text", "")
        filename = payload.get("filename", "document.txt")
        try:
            if not text or not text.strip():
                return {"status": "error", "message": "Empty text"}

            def _do_upload():
                global _index_cache
                llm = get_llm()
                embed_model = get_embed_model()
                from llama_index.core import Settings, Document, VectorStoreIndex
                Settings.llm = llm
                Settings.embed_model = embed_model
                logger.info(
                    f"Upload Settings: llm={type(llm).__name__}, "
                    f"embed={type(embed_model).__name__}"
                )
                # Store original filename in metadata
                doc = Document(text=text, id_=filename, metadata={"original_filename": filename})
                index = VectorStoreIndex.from_documents(
                    [doc], embed_model=embed_model,
                )
                index.storage_context.persist(persist_dir=INDEX_VOL_PATH)
                _index_cache = index
                return len(text)

            with _model_lock:
                text_len = await asyncio.to_thread(_do_upload)
            await index_volume.commit.aio()
            return {"status": "success", "message": f"Indexed {text_len} characters"}
        except Exception as e:
            logger.exception("Upload failed")
            return {"status": "error", "message": str(e)}

    # ---------- 查询 ----------
    @modal.fastapi_endpoint(method="POST", label="query")
    async def query(self, payload: dict):
        """POST JSON body: {"question": "..."}"""
        question = payload.get("question", "")
        if not question:
            return {"answer": "Missing 'question' in JSON body."}
        try:
            # Ensure Settings.embed_model is set BEFORE any index load
            def _prepare():
                embed_model = get_embed_model()
                from llama_index.core import Settings
                Settings.embed_model = embed_model
                return embed_model

            embed_model = await asyncio.to_thread(_prepare)
            logger.info(
                f"Query Settings.embed_model set to {type(embed_model).__name__}"
            )

            index = get_or_load_index()
            if index is None:
                return {
                    "answer": "No document uploaded. Please upload a .txt file first."
                }

            def _do_query():
                llm = get_llm()
                from llama_index.core import Settings
                Settings.llm = llm
                logger.info(f"Query Settings.llm set to {type(llm).__name__}")
                query_engine = index.as_query_engine(
                    similarity_top_k=2,
                    response_mode="compact",
                )
                logger.info(f"Running query: {question}")
                t0 = time.time()
                response = query_engine.query(question)
                t1 = time.time()
                answer = str(response).strip()
                logger.info(f"Query answered in {t1-t0:.1f}s: {answer[:200]}...")
                return answer

            with _model_lock:
                answer = await asyncio.to_thread(_do_query)

            if not answer or answer == "None":
                return {
                    "answer": "I couldn't find a relevant answer. Please try a different question."
                }
            return {"answer": answer}
        except Exception as e:
            logger.exception("Query failed")
            return {"answer": f"Backend error: {str(e)}"}

# GPU snapshots eliminate the need for keep_warm / warmup_model functions.
# The @modal.enter(snap=True) hook in RAGBackend.load() handles model preloading
# and Modal restores GPU memory state on every cold start.
