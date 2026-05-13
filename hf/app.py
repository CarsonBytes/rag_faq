"""
RAG Chat Interface for Hugging Face Space.
Connects to Modal backend for document indexing and question answering.
"""
import gradio as gr
import requests
import os
import time
import re
from typing import Tuple, Optional, Generator

# ---------- Configuration Constants ----------
BACKEND_URLS = {
    "query": "https://carsonbytes--query.modal.run",
    "debug": "https://carsonbytes--debug.modal.run",
    "upload": "https://carsonbytes--upload.modal.run",
    "health": "https://carsonbytes--health.modal.run",
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
    """Clean up the answer - remove any trailing prompt templates or artifacts."""
    import re
    # Remove [INST], <<SYS>>, and other prompt template markers
    # First remove the full prompt template block
    cleaned = re.sub(r'\s*\[INST\]\s*<<SYS>>\s*.*?<<SYS>>\s*<<SYS>>\s*$', '', answer, flags=re.DOTALL)
    # Then remove any remaining [INST], <<SYS>>, <</SYS>> markers
    cleaned = re.sub(r'\s*\[INST\]', '', cleaned)
    cleaned = re.sub(r'\s*<<SYS>>', '', cleaned)
    cleaned = re.sub(r'\s*<</SYS>>', '', cleaned)
    # Remove any remaining prompt instructions
    cleaned = re.sub(r'\s*You are a helpful assistant.*?$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*Based on the context information.*?$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*Query:.*?$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


# ---------- Backend Communication Functions ----------
def check_health() -> Tuple[bool, Optional[str]]:
    """Check if backend is running and responsive. Returns (success, filename)."""
    try:
        resp = requests.get(
            get_backend_url("health"),
            timeout=WARMUP_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            index_exists = data.get("index_exists", False)
            filename = data.get("indexed_filename", None)
            return index_exists, filename
        return False, None
    except Exception:
        return False, None


def check_index_status() -> Tuple[bool, Optional[str]]:
    """
    Check if an index exists by sending a test query.
    Returns tuple: (index_exists: bool, filename: str or None)
    """
    try:
        response = requests.post(
            get_backend_url("query"),
            json={"question": "__check_index_status__"},
            timeout=QUERY_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "")
        
        # Check if "No document uploaded" is in the response
        if "no document uploaded" in answer.lower():
            return False, None
        
        # Index exists - try to get filename from response metadata if available
        filename = data.get("indexed_filename", None)
        return True, filename
        
    except Exception:
        # On error, assume no index
        return False, None


def upload_document(file_obj) -> Tuple[bool, Optional[str], str]:
    """
    Upload a document to the backend for indexing.
    Accepts file object, extracts text content.
    Returns tuple: (success: bool, filename: str, message: str)
    """
    if file_obj is None:
        return False, None, "❌ No file selected."

    # Check file size
    file_size = os.path.getsize(file_obj.name) if os.path.exists(file_obj.name) else 0
    if file_size > MAX_FILE_SIZE:
        return False, None, f"❌ File too large. Maximum size is {format_file_size(MAX_FILE_SIZE)}."

    try:
        # Read file content as text
        with open(file_obj.name, "r", encoding="utf-8") as f:
            text_content = f.read()

        if not text_content or not text_content.strip():
            return False, None, "❌ File is empty."

        filename = os.path.basename(file_obj.name)

        # Send JSON payload to backend
        payload = {
            "text": text_content,
            "filename": filename
        }
        response = requests.post(
            get_backend_url("upload"),
            json=payload,
            timeout=UPLOAD_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success":
            indexed_name = data.get("message", filename)
            return True, indexed_name, f"✅ Indexed {indexed_name}"
        else:
            return False, None, f"❌ {data.get('message', 'Upload failed')}"

    except requests.exceptions.Timeout:
        return False, None, "❌ Upload timed out. Please try a smaller file."
    except UnicodeDecodeError:
        return False, None, "❌ Could not decode file. Please use a text file."
    except Exception as e:
        return False, None, f"❌ Upload failed: {str(e)}"


def query_backend(message: str, history: list) -> Generator[Tuple[str, list], None, None]:
    """
    Send a question to the backend and yield responses.
    Yields status updates and final answer.
    """
    if not message or not message.strip():
        yield "", history
        return

    # Validate question length
    if len(message) > MAX_QUESTION_LENGTH:
        message = message[:MAX_QUESTION_LENGTH]

    # Add user message to history
    history.append((message, None))

    # Initial thinking message
    yield "🤔 Thinking...", history

    try:
        response = requests.post(
            get_backend_url("query"),
            json={"question": message},
            timeout=QUERY_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "⚠️ Received an empty response.")
        # Clean up the answer
        answer = clean_answer(answer)

        # Update history with the answer
        history[-1] = (message, answer)
        yield "", history

    except requests.exceptions.Timeout:
        history[-1] = (message, "❌ Query timed out. Please try again.")
        yield "", history
    except requests.exceptions.ConnectionError:
        history[-1] = (message, "❌ Cannot connect to backend. Please try again later.")
        yield "", history
    except Exception as e:
        history[-1] = (message, f"❌ Error: {str(e)}")
        yield "", history


def update_index_display(filename: Optional[str], is_indexed: bool) -> str:
    """Update the index status display based on current state."""
    if is_indexed and filename:
        return f"**Indexed file:** 📄 {filename}"
    elif is_indexed:
        return "**Indexed file:** ✅ Document indexed (filename unknown)"
    return "**Indexed file:** No document indexed"


def initialize_index_state() -> Tuple[bool, Optional[str], str]:
    """Check index status on page load and return state values."""
    is_indexed, filename = check_index_status()
    
    # Update display text based on index status
    if is_indexed and filename:
        display_text = f"**Indexed file:** 📄 {filename}"
    elif is_indexed:
        display_text = "**Indexed file:** ✅ Document indexed (filename unknown)"
    else:
        display_text = "**Indexed file:** No document indexed"
    
    return is_indexed, filename, display_text


def warmup_with_status(progress=gr.Progress(), start_time: float = None) -> Generator[Tuple[str, bool, str, float], None, None]:
    """
    Warmup function to initialize the backend on Space startup.
    Returns tuple: (status_message: str, success: bool, loading_text: str, current_time: float)
    Only calls the health endpoint - no query endpoint.
    Displays elapsed time during warmup.
    Waits indefinitely for backend to become ready.
    """
    if start_time is None:
        start_time = time.time()
    
    # Show loading state with progress message
    elapsed = time.time() - start_time
    yield f"⏳ Loading GPU snapshots to speed up cold startup. It would take < 15 seconds... ({elapsed:.1f}s elapsed)", False, f"⏳ Warming up... ({elapsed:.1f}s)", start_time
    
    try:
        resp = requests.get(
            get_backend_url("health"),
            timeout=None  # Wait indefinitely
        )
        
        elapsed = time.time() - start_time
        if resp.status_code == 200:
            yield f"✅ Backend ready ({elapsed:.1f}s)", True, f"✅ Backend ready ({elapsed:.1f}s)", start_time
        else:
            yield f"❌ Backend returned status {resp.status_code} ({elapsed:.1f}s)", False, f"❌ Backend returned status ({elapsed:.1f}s)", start_time
            
    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start_time
        yield f"❌ Cannot connect to backend. Please check your connection. ({elapsed:.1f}s)", False, f"❌ Cannot connect ({elapsed:.1f}s)", start_time
    except Exception as e:
        elapsed = time.time() - start_time
        yield f"❌ Warm-up failed: {str(e)} ({elapsed:.1f}s)", False, f"❌ Warm-up failed: {str(e)} ({elapsed:.1f}s)", start_time


# ---------- UI Components ----------
def create_warmup_page() -> Tuple[gr.Blocks, gr.State]:
    """Create the warm-up landing page."""
    with gr.Blocks(fill_height=True) as demo:
        # Header
        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 15vh;">
                <h1 style="font-size: 3em; margin-bottom: 0.5em;">📚 RAG Chat Interface</h1>
                <p style="font-size: 1.2em; color: #666;">Powered by Qwen 2.5 & Modal</p>
                <br>
                <p style="color: #555;">Click the button below to warm up the backend and get started.</p>
            </div>
            """,
            elem_classes=["center"]
        )
        
        with gr.Row():
            with gr.Column():
                warmup_btn = gr.Button(
                    "🔥 Warm Up Backend",
                    variant="primary",
                    size="lg",
                    elem_classes=["warmup-btn"]
                )
        
        warmup_status = gr.Markdown("", elem_classes=["status-msg"])
        
        # Hidden state for navigation
        ready = gr.State(False)
        start_time = gr.State(time.time())
        
        warmup_btn.click(
            fn=lambda: time.time(),
            outputs=[start_time]
        ).then(
            fn=warmup_with_status,
            inputs=[start_time],
            outputs=[warmup_status, ready, warmup_btn, start_time]
        )
        
    return demo, ready


def create_chat_page() -> gr.Blocks:
    """Create the main chat interface page."""
    with gr.Blocks() as demo:
        # Header
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 2em;">
                <h1>📚 RAG Chat Interface</h1>
                <p style="color: #666;">Upload a document, then ask questions about its content.</p>
            </div>
            """
        )

        # ---------- State ----------
        document_indexed = gr.State(False)
        indexed_filename = gr.State(None)

        # ---------- Index Status Display ----------
        with gr.Row():
            with gr.Column():
                index_status = gr.Markdown("**Indexed file:** No document indexed")

        # ---------- File Upload Section ----------
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

        # ---------- Chat Interface Section ----------
        gr.Markdown(
            """
            <div style="margin-top: 2em; margin-bottom: 1em;">
                <h3>💬 Ask Questions</h3>
            </div>
            """
        )

        def chat_wrapper(message: str, history: list) -> Generator[Tuple[str, list], None, None]:
            """Wrapper that checks if document is indexed before querying."""
            if not document_indexed.value:
                yield "⚠️ Please upload a document first using the 'Build Index' button.", history
                return
            yield from query_backend(message, history)

        gr.ChatInterface(
            fn=chat_wrapper,
            title=None,
            description=None,
        )

    return demo


# Main application
def create_app() -> gr.Blocks:
    """Create the main application with page navigation."""
    with gr.Blocks() as demo:
        # Page state
        page = gr.State("warmup")
        
        # Index state
        document_indexed = gr.State(False)
        indexed_filename = gr.State(None)
        
        # Container for page content
        page_container = gr.Column()
        
        with page_container:
            # Warmup page
            with gr.Column(elem_id="warmup-page") as warmup_page:
                gr.Markdown(
                    """
                    <div style="text-align: center; margin-top: 15vh;">
                        <h1 style="font-size: 3em; margin-bottom: 0.5em;">📚 RAG Chat Interface</h1>
                        <p style="font-size: 1.2em; color: #666;">Powered by Qwen 2.5 & Modal</p>
                        <br>
                        <p style="color: #555;">Click the button below to warm up the backend and get started.</p>
                    </div>
                    """,
                    elem_classes=["center"]
                )
                
                with gr.Row():
                    with gr.Column():
                        warmup_btn = gr.Button(
                            "🔥 Warm Up Backend",
                            variant="primary",
                            size="lg",
                            elem_classes=["warmup-btn"]
                        )
                
                warmup_status = gr.Markdown("", elem_classes=["status-msg"])
                
                # Hidden state for navigation
                ready = gr.State(False)
                start_time = gr.State(time.time())
                
                warmup_btn.click(
                    fn=lambda: time.time(),
                    outputs=[start_time]
                ).then(
                    fn=warmup_with_status,
                    inputs=[start_time],
                    outputs=[warmup_status, ready, warmup_btn, start_time]
                )
            
            # Chat page (initially hidden)
            with gr.Column(elem_id="chat-page", visible=False) as chat_page:
                # Header
                gr.Markdown(
                    """
                    <div style="text-align: center; margin-bottom: 2em;">
                        <h1>📚 RAG Chat Interface</h1>
                        <p style="color: #666;">Upload a document, then ask questions about its content.</p>
                    </div>
                    """
                )

                # ---------- Index Status Display ----------
                with gr.Row():
                    with gr.Column():
                        index_status = gr.Markdown("**Indexed file:** No document indexed")

                # ---------- File Upload Section ----------
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

                # ---------- Chat Interface Section ----------
                gr.Markdown(
                    """
                    <div style="margin-top: 2em; margin-bottom: 1em;">
                        <h3>💬 Ask Questions</h3>
                    </div>
                    """
                )

                def chat_wrapper(message: str, history: list) -> Generator[Tuple[str, list], None, None]:
                    """Wrapper that checks if document is indexed before querying."""
                    if not document_indexed.value:
                        yield "⚠️ Please upload a document first using the 'Build Index' button.", history
                        return
                    yield from query_backend(message, history)

                gr.ChatInterface(
                    fn=chat_wrapper,
                    title=None,
                    description=None,
                )

        # Navigation logic - only trigger when ready changes from False to True
        ready.change(
            fn=lambda r: (gr.update(visible=False), gr.update(visible=True), "chat" if r else "warmup"),
            inputs=[ready],
            outputs=[warmup_page, chat_page, page]
        )
    
    return demo


# Launch the application
if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
