import modal
import os
import nest_asyncio
import asyncio
import logging
import time

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

# Use NVIDIA CUDA devel image so nvcc is available during image build.
# Modal docs explicitly recommend this for libraries that need CUDA toolkit.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])  # removes chatty prints on entry
    .apt_install("git", "build-essential", "cmake", "curl", "wget", "patchelf")
    .pip_install(
        "llama-index",
        "llama-index-embeddings-huggingface",
        "huggingface_hub",
        "fastapi[standard]",
        "python-multipart",
        "nest-asyncio",
    )
    .run_function(build_llama_cpp_cuda, gpu="T4")
)

model_volume = modal.Volume.from_name("llamacpp-models", create_if_missing=True)
index_volume = modal.Volume.from_name("rag-index-store", create_if_missing=True)

app = modal.App(name="llamacpp-rag-backend", image=image)

# ---------- 惰性加载：启动时不加载模型 ----------
_llm = None
_embed_model = None
_index_cache = None
_model_lock = asyncio.Lock()

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
        model_volume.commit()

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

@app.function(
    gpu="T4",
    volumes={MODEL_VOL_PATH: model_volume, INDEX_VOL_PATH: index_volume},
    max_containers=1,
    min_containers=1,
    timeout=600,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.asgi_app()
def fastapi_app():
    nest_asyncio.apply()
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from pydantic import BaseModel
    from llama_index.core import Document, VectorStoreIndex, Settings
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Container starting up...")
        # Preload lightweight embed model; set Settings so index ops never fallback to OpenAI.
        embed_model = await asyncio.to_thread(get_embed_model)
        Settings.embed_model = embed_model
        logger.info(f"Embed model ready: {type(embed_model).__name__}")
        llm = await asyncio.to_thread(get_llm)
        Settings.llm = llm
        logger.info(f"LLM ready: {type(llm).__name__}")
        yield
        logger.info("Container shutting down...")

    web_app = FastAPI(title="Qwen 2.5 RAG API", lifespan=lifespan)

    # ---------- 健康检查（总是可用） ----------
    @web_app.get("/health")
    async def health():
        return {"status": "ok", "message": "Backend is alive"}

    @web_app.get("/debug")
    async def debug():
        """Inspect current Settings to verify embed model is not OpenAI."""
        from llama_index.core import Settings as S
        return {
            "llm_type": type(S.llm).__name__ if S.llm else None,
            "embed_type": type(S.embed_model).__name__ if S.embed_model else None,
            "embed_module": str(S.embed_model.__class__.__module__) if S.embed_model else None,
        }

    class QueryRequest(BaseModel):
        question: str

    # ---------- 上传文档（按需加载模型） ----------
    @web_app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        global _index_cache
        try:
            content = await file.read()
            text = content.decode("utf-8")
            if not text.strip():
                raise HTTPException(status_code=400, detail="File is empty")

            async with _model_lock:
                # llama-cpp-python is not thread-safe; run blocking init in thread
                llm = await asyncio.to_thread(get_llm)
                embed_model = await asyncio.to_thread(get_embed_model)
                Settings.llm = llm
                Settings.embed_model = embed_model
                logger.info(f"Upload Settings: llm={type(llm).__name__}, embed={type(embed_model).__name__}")

                doc = Document(text=text, id_=file.filename)
                # Explicitly pass embed_model so index is bound to HuggingFace, not OpenAI
                index = await asyncio.to_thread(
                    VectorStoreIndex.from_documents,
                    [doc],
                    embed_model=embed_model,
                )
                await asyncio.to_thread(index.storage_context.persist, persist_dir=INDEX_VOL_PATH)
                await index_volume.commit.aio()
                _index_cache = index

            return {"status": "success", "message": f"Indexed {len(text)} characters"}
        except Exception as e:
            logger.exception("Upload failed")
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- 查询（按需加载模型和索引） ----------
    @web_app.post("/query")
    async def query(req: QueryRequest):
        try:
            # Ensure embed model is in Settings BEFORE any index load
            # (load_index_from_storage may resolve defaults if Settings is empty)
            embed_model = await asyncio.to_thread(get_embed_model)
            Settings.embed_model = embed_model
            logger.info(f"Query Settings.embed_model set to {type(embed_model).__name__}")

            index = get_or_load_index()
            if index is None:
                return {"answer": "No document uploaded. Please upload a .txt file first."}

            async with _model_lock:
                llm = await asyncio.to_thread(get_llm)
                Settings.llm = llm
                logger.info(f"Query Settings.llm set to {type(llm).__name__}")

                # compact = single LLM call; avoid refine/tree_summarize which make multiple calls
                # Settings.llm is already set above; as_query_engine picks it up automatically
                query_engine = index.as_query_engine(
                    similarity_top_k=2,
                    response_mode="compact",
                )
                logger.info(f"Running query: {req.question}")
                t0 = time.time()
                response = await asyncio.to_thread(query_engine.query, req.question)
                t1 = time.time()
                answer = str(response).strip()
                logger.info(f"Query answered in {t1-t0:.1f}s: {answer[:200]}...")

            if not answer or answer == "None":
                return {"answer": "I couldn't find a relevant answer. Please try a different question."}
            return {"answer": answer}
        except Exception as e:
            logger.exception("Query failed")
            return {"answer": f"Backend error: {str(e)}"}

    return web_app

# ---------- 可选预热（注释掉 schedule） ----------
@app.function(
    gpu="T4",
    volumes={MODEL_VOL_PATH: model_volume},
    # schedule=modal.Period(seconds=300),   # 需要保活可以取消注释
)
def keep_warm():
    from llama_index.core import Settings
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
    _ = Settings.llm.complete("ping")
    print("Keep warm: container kept alive")

@app.function(
    gpu="T4",
    volumes={MODEL_VOL_PATH: model_volume},
    timeout=300,
)
def warmup_model():
    from llama_index.core import Settings
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
    _ = Settings.llm.complete("Hello")
    print("Warmup completed")