import modal
import os
import nest_asyncio

MODEL_VOL_PATH = "/models"
INDEX_VOL_PATH = "/index_store"

def build_llama_cpp_cuda():
    import subprocess
    os.environ["CMAKE_ARGS"] = "-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;89"
    subprocess.check_call(["pip", "install", "llama-cpp-python", "--upgrade", "--no-cache-dir"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "curl", "wget")
    .pip_install(
        "llama-index",
        "llama-index-llms-llama-cpp",
        "llama-index-embeddings-huggingface",
        "huggingface_hub",
        "fastapi[standard]",
        "python-multipart",
        "nest-asyncio",
    )
    .run_function(build_llama_cpp_cuda, gpu="any")
)

model_volume = modal.Volume.from_name("llamacpp-models", create_if_missing=True)
index_volume = modal.Volume.from_name("rag-index-store", create_if_missing=True)

app = modal.App(name="llamacpp-rag-backend", image=image)

def get_llm():
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

    return LlamaCPP(
        model_path=model_path,
        temperature=0.0,                # 确定性输出，更快
        max_new_tokens=128,             # 缩短回复长度
        context_window=4096,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        model_kwargs={
            "n_gpu_layers": -1,
            "logits_all": False,
        },
        verbose=False,
    )

def get_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # 更轻量的 embedding 模型
    return HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

def init_settings():
    from llama_index.core import Settings
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()

# 全局索引缓存
_index_cache = None

def get_or_load_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    from llama_index.core import StorageContext, load_index_from_storage
    if not os.path.exists(f"{INDEX_VOL_PATH}/docstore.json"):
        return None
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_VOL_PATH)
    _index_cache = load_index_from_storage(storage_context)
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
    from llama_index.core import Document, VectorStoreIndex

    init_settings()
    # 预热：提前加载索引（如果存在）
    _ = get_or_load_index()

    web_app = FastAPI(title="Qwen 2.5 RAG API")

    class QueryRequest(BaseModel):
        question: str

    @web_app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        global _index_cache
        try:
            content = await file.read()
            text = content.decode("utf-8")
            if not text.strip():
                raise HTTPException(status_code=400, detail="File is empty")
            doc = Document(text=text, id_=file.filename)
            index = VectorStoreIndex.from_documents([doc])
            index.storage_context.persist(persist_dir=INDEX_VOL_PATH)
            await index_volume.commit.aio()    # 异步提交
            _index_cache = index
            return {"status": "success", "message": f"Indexed {len(text)} characters"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/query")
    async def query(req: QueryRequest):
        try:
            index = get_or_load_index()
            if index is None:
                return {"answer": "No document uploaded yet. Please use /upload first."}
            query_engine = index.as_query_engine(similarity_top_k=2)
            response = query_engine.query(req.question)
            answer = str(response).strip()
            if not answer or answer == "None":
                return {"answer": "I couldn't find an answer. Try rephrasing."}
            return {"answer": answer}
        except Exception as e:
            print(f"Query error: {e}")
            return {"answer": f"Backend error: {str(e)}"}

    return web_app