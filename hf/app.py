"""
RAG Chat Interface for Hugging Face Space.
Connects to Modal backend for document indexing and question answering.
"""
import gradio as gr
import requests
import os
import time
import threading
import re
from typing import Optional, Generator, Tuple

# ---------- Configuration Constants ----------
BACKEND_URLS = {
    "query": "https://carsonbytes--query.modal.run/",
    "debug": "https://carsonbytes--debug.modal.run/",
    "upload": "https://carsonbytes--upload.modal.run/",
    "health": "https://carsonbytes--health.modal.run/",
}

WARMUP_TIMEOUT = 15  # seconds
QUERY_TIMEOUT = 60   # seconds
UPLOAD_TIMEOUT = 120  # seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
MAX_QUESTION_LENGTH = 1000  # characters


# ---------- Utility Functions ----------
def get_backend_url(endpoint: str) -> str:
    """Get the URL for a specific backend endpoint."""
    return BACKEND_URLS.get(endpoint, BACKEND_URLS["query"])


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def clean_answer(answer: str) -> str:
    """Remove prompt-template bleed-through from the model's response."""
    # Truncate at [INST] — everything after is a leaked prompt
    if '[INST]' in answer:
        answer = answer[:answer.index('[INST]')]
    # Truncate at any remaining SYS markers
    for marker in ['<<SYS>>', '<</SYS>>', '<</SYS']:
        if marker in answer:
            answer = answer[:answer.index(marker)]
    return answer.strip()


# ---------- Backend Communication Functions ----------
def check_health() -> Tuple[bool, Optional[str]]:
    """Check if backend is running and return (index_exists, indexed_filename)."""
    try:
        resp = requests.get(get_backend_url("health"), timeout=WARMUP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            index_exists = data.get("index_exists", False)
            filename = data.get("indexed_filename", None)
            return index_exists, filename
        return False, None
    except Exception:
        return False, None


def upload_document(file_obj) -> Tuple[bool, Optional[str], str]:
    """
    Upload a document to the backend for indexing.
    Returns tuple: (success: bool, filename: str, message: str)
    """
    if file_obj is None:
        return False, None, "❌ No file selected."

    file_size = os.path.getsize(file_obj.name) if os.path.exists(file_obj.name) else 0
    if file_size > MAX_FILE_SIZE:
        return False, None, f"❌ File too large. Maximum size is {format_file_size(MAX_FILE_SIZE)}."

    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            text_content = f.read()

        if not text_content or not text_content.strip():
            return False, None, "❌ File is empty."

        filename = os.path.basename(file_obj.name)

        payload = {"text": text_content, "filename": filename}
        response = requests.post(
            get_backend_url("upload"),
            json=payload,
            timeout=UPLOAD_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success":
            return True, filename, f"✅ Indexed {filename}"
        else:
            return False, None, f"❌ {data.get('message', 'Upload failed')}"

    except requests.exceptions.Timeout:
        return False, None, "❌ Upload timed out. Please try a smaller file."
    except UnicodeDecodeError:
        return False, None, "❌ Could not decode file. Please use a text file."
    except Exception as e:
        return False, None, f"❌ Upload failed: {str(e)}"


def query_backend(message: str) -> Generator[str, None, None]:
    """Send a question to the backend and yield the response string."""
    if not message or not message.strip():
        return

    if len(message) > MAX_QUESTION_LENGTH:
        message = message[:MAX_QUESTION_LENGTH]

    yield "🤔 Thinking..."

    try:
        response = requests.post(
            get_backend_url("query"),
            json={"question": message},
            timeout=QUERY_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "⚠️ Received an empty response.")
        yield clean_answer(answer)

    except requests.exceptions.Timeout:
        yield "❌ Query timed out. Please try again."
    except requests.exceptions.ConnectionError:
        yield "❌ Cannot connect to backend. Please try again later."
    except Exception as e:
        yield f"❌ Error: {str(e)}"


def update_index_display(filename: Optional[str], is_indexed: bool) -> str:
    if is_indexed and filename:
        return f"**Indexed file:** 📄 {filename}"
    elif is_indexed:
        return "**Indexed file:** ✅ Document indexed (filename unknown)"
    return "**Indexed file:** No document indexed"


def load_index_state() -> Tuple[bool, Optional[str], str]:
    """Called on page load — uses /health to get current index state."""
    is_indexed, filename = check_health()
    display = update_index_display(filename, is_indexed)
    return is_indexed, filename, display


def warmup_with_status(start_time: float = None) -> Generator[Tuple[str, bool, str, float], None, None]:
    """
    Polls the health endpoint in a background thread and ticks the timer every second.
    Yields: (status_message, success, button_label, start_time)
    """
    if start_time is None:
        start_time = time.time()

    result = {"done": False, "success": False, "status_code": None, "error": None}

    def do_request():
        try:
            resp = requests.get(get_backend_url("health"), timeout=None)
            result["status_code"] = resp.status_code
            result["success"] = resp.status_code == 200
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = threading.Thread(target=do_request, daemon=True)
    thread.start()

    while not result["done"]:
        elapsed = time.time() - start_time
        msg = f"⏳ Loading GPU snapshots... ({elapsed:.1f}s elapsed)"
        yield msg, False, f"⏳ Warming up... ({elapsed:.1f}s)", start_time
        time.sleep(1)

    elapsed = time.time() - start_time
    if result["success"]:
        msg = f"✅ Backend ready ({elapsed:.1f}s)"
        yield msg, True, msg, start_time
    elif result["error"]:
        msg = f"❌ Cannot connect: {result['error']} ({elapsed:.1f}s)"
        yield msg, False, f"❌ Failed ({elapsed:.1f}s)", start_time
    else:
        msg = f"❌ Backend returned status {result['status_code']} ({elapsed:.1f}s)"
        yield msg, False, f"❌ Failed ({elapsed:.1f}s)", start_time


# ---------- Main App ----------
def create_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        page = gr.State("warmup")
        document_indexed = gr.State(False)
        indexed_filename = gr.State(None)

        # --- Warmup page ---
        with gr.Column(elem_id="warmup-page") as warmup_page:
            gr.Markdown(
                """
                <div style="text-align: center; margin-top: 15vh;">
                    <h1 style="font-size: 3em; margin-bottom: 0.5em;">📚 RAG Chat Interface</h1>
                    <p style="font-size: 1.2em; color: #666;">Powered by Qwen 2.5 & Modal</p>
                    <br>
                    <p style="color: #555;">Click the button below to warm up the backend and get started.</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column():
                    warmup_btn = gr.Button(
                        "🔥 Warm Up Backend",
                        variant="primary",
                        size="lg",
                    )

            warmup_status = gr.Markdown("")
            ready = gr.State(False)
            start_time_state = gr.State(0.0)

            warmup_btn.click(
                fn=lambda: time.time(),
                outputs=[start_time_state]
            ).then(
                fn=warmup_with_status,
                inputs=[start_time_state],
                outputs=[warmup_status, ready, warmup_btn, start_time_state]
            )

        # --- Chat page (initially hidden) ---
        with gr.Column(elem_id="chat-page", visible=False) as chat_page:
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 2em;">
                    <h1>📚 RAG Chat Interface</h1>
                    <p style="color: #666;">Upload a document, then ask questions about its content.</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column():
                    index_status = gr.Markdown("**Indexed file:** No document indexed")

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload .txt file",
                        file_types=[".txt"],
                        file_count="single",
                        show_label=True
                    )
                with gr.Column(scale=1):
                    upload_btn = gr.Button("📤 Build Index", variant="primary", size="lg")

            with gr.Row():
                upload_status = gr.Markdown("**Status:** Waiting for document...")

            upload_btn.click(
                fn=upload_document,
                inputs=[file_input],
                outputs=[document_indexed, indexed_filename, upload_status]
            ).then(
                fn=update_index_display,
                inputs=[indexed_filename, document_indexed],
                outputs=[index_status]
            )

            gr.Markdown(
                """
                <div style="margin-top: 2em; margin-bottom: 1em;">
                    <h3>💬 Ask Questions</h3>
                </div>
                """
            )

            def chat_wrapper(message: str, history: list, is_indexed: bool) -> Generator[str, None, None]:
                if not is_indexed:
                    yield "⚠️ Please upload a document first using the 'Build Index' button."
                    return
                yield from query_backend(message)

            gr.ChatInterface(
                fn=chat_wrapper,
                additional_inputs=[document_indexed],
                title=None,
                description=None,
            )

        # Switch pages when warmup completes
        ready.change(
            fn=lambda r: (gr.update(visible=not r), gr.update(visible=r), "chat" if r else "warmup"),
            inputs=[ready],
            outputs=[warmup_page, chat_page, page]
        )

        # On page load, check /health and populate index state for chat page
        demo.load(
            fn=load_index_state,
            outputs=[document_indexed, indexed_filename, index_status]
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
