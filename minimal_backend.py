import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi", "uvicorn")

app = modal.App("simple-rag-backend", image=image)

@app.function(min_containers=1, timeout=600)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    web_app = FastAPI()

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.post("/upload")
    async def upload():
        return {"status": "upload placeholder"}

    @web_app.post("/query")
    async def query():
        return {"answer": "Hello from simple backend"}

    return web_app