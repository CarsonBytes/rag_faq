import gradio as gr
import requests
import os

BACKEND_URL = "https://carsonbytes--llamacpp-rag-backend-fastapi-app.modal.run"

def chat(message, history, uploaded_state):
    # 先给一个占位消息，避免用户无聊等待
    if not uploaded_state:
        yield "⚠️ Please upload a document first using the 'Build Index' button."
        return
    if not message:
        yield ""
        return

    # 发送"正在思考"提示（yield 的第一个值会立刻显示）
    yield "🤔 Thinking... (this may take 5–15 seconds on first query)"

    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": message},
            timeout=120   # 🔥 增加到 120 秒
        )
        data = resp.json()
        ans = data.get("answer", "⚠️ Received an empty response.")
        # 覆盖之前的占位消息
        yield ans
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def on_upload(file):
    if not file:
        return False, "❌ No file selected."

    try:
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f, "text/plain")}
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
            response.raise_for_status()
        return True, "✅ Index built successfully! You can now ask questions."
    except Exception as e:
        return False, f"❌ Upload failed: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Fast RAG with Qwen 2.5 (Optimized)")
    uploaded = gr.State(False)

    with gr.Row():
        file_input = gr.File(label="Upload .txt file", file_types=[".txt"])
        upload_btn = gr.Button("Build Index")
        status = gr.Markdown("**Status:** Waiting for document...")

    upload_btn.click(
        on_upload,
        inputs=[file_input],
        outputs=[uploaded, status]
    )

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[uploaded],
        title="Ask questions about your document"
    )

if __name__ == "__main__":
    demo.launch()