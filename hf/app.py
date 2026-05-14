"""
RAG Chat Interface for Hugging Face Space.
Connects to Modal backend for document indexing and question answering.
"""
import gradio as gr
import requests
import os
import time
import threading
from typing import Optional, Generator, Tuple

# ---------- Configuration Constants ----------
BACKEND_URLS = {
    "query": "https://carsonbytes--query.modal.run/",
    "debug": "https://carsonbytes--debug.modal.run/",
    "upload": "https://carsonbytes--upload.modal.run/",
    "health": "https://carsonbytes--health.modal.run/",
}

WARMUP_TIMEOUT = 15
QUERY_TIMEOUT = 60
UPLOAD_TIMEOUT = 120
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_QUESTION_LENGTH = 1000

# ---------- Sample Annual Reports ----------
SAMPLE_LABELS = {
    "Annual Report 2023": 2023,
    "Annual Report 2024": 2024,
    "Annual Report 2025": 2025,
    "Annual Report 2026": 2026,
}

SAMPLE_DOCUMENTS = {
    2023: """\
TechVision Corp — Annual Report 2023

LETTER TO SHAREHOLDERS
Dear Shareholders,
2023 was a year of resilient growth despite macro headwinds. We delivered record revenue
while investing heavily in our AI platform and expanding into three new markets.

FINANCIAL HIGHLIGHTS
Total Revenue:        $4.82 billion  (+12% YoY)
Gross Profit:         $2.17 billion  (45.0% margin)
Operating Income:     $681 million   (14.1% margin)
Net Income:           $512 million
EPS (diluted):        $3.18
Free Cash Flow:       $743 million
Headcount (year-end): 18,400

SEGMENT REVENUE
  Cloud Services:     $2.61B  (+21%)
  Enterprise Software:$1.44B  (+4%)
  Professional Svcs:  $0.77B  (+2%)

KEY RISKS
1. Increasing competition from hyperscale cloud providers threatening Cloud Services margins.
2. Customer concentration — top 10 clients represent 34% of revenue.
3. Cybersecurity threats and potential data-breach liability.
4. Regulatory scrutiny around AI model outputs in the EU (AI Act compliance costs).
5. Foreign-exchange headwinds: 38% of revenue is denominated in non-USD currencies.

EXECUTIVE LEADERSHIP
CEO:   Sarah Chen        (since 2019)
CFO:   Marcus Webb       (since 2021)
COO:   Priya Nair        (since 2022)
CTO:   Daniel Kowalski   (since 2020)
""",
    2024: """\
TechVision Corp — Annual Report 2024

LETTER TO SHAREHOLDERS
Dear Shareholders,
2024 marked our strongest growth year since IPO. AI-powered features drove record
adoption in Cloud Services and we completed the acquisition of DataBridge Inc.

FINANCIAL HIGHLIGHTS
Total Revenue:        $5.74 billion  (+19% YoY)
Gross Profit:         $2.70 billion  (47.1% margin)
Operating Income:     $874 million   (15.2% margin)
Net Income:           $658 million
EPS (diluted):        $4.07
Free Cash Flow:       $921 million
Headcount (year-end): 21,750 (includes ~1,200 from DataBridge acquisition)

SEGMENT REVENUE
  Cloud Services:     $3.38B  (+30%)
  Enterprise Software:$1.52B  (+6%)
  Professional Svcs:  $0.84B  (+9%)

KEY RISKS
1. Integration risk from DataBridge acquisition — potential system and culture misalignment.
2. GPU supply constraints limiting AI infrastructure expansion capacity.
3. Rising interest rates increasing cost of capital for planned data-center builds.
4. Talent retention — attrition in AI/ML roles reached 14%, above industry average.
5. Geopolitical risk: 12% of revenue exposed to APAC regions under trade restrictions.

EXECUTIVE LEADERSHIP
CEO:   Sarah Chen        (since 2019)
CFO:   Marcus Webb       (since 2021)
COO:   James Okafor      (joined 2024, replaced Priya Nair)
CTO:   Daniel Kowalski   (since 2020)
Chief AI Officer: Lena Park (new role, appointed 2024)
""",
    2025: """\
TechVision Corp — Annual Report 2025

LETTER TO SHAREHOLDERS
Dear Shareholders,
2025 was a transformational year. We crossed $7 billion in revenue, launched TechVision AI Studio,
and completed the full integration of DataBridge. Our AI platform now serves over 6,000 enterprise clients.

FINANCIAL HIGHLIGHTS
Total Revenue:        $7.11 billion  (+24% YoY)
Gross Profit:         $3.52 billion  (49.5% margin)
Operating Income:     $1.14 billion  (16.0% margin)
Net Income:           $867 million
EPS (diluted):        $5.36
Free Cash Flow:       $1.18 billion
Headcount (year-end): 25,300

SEGMENT REVENUE
  Cloud Services:     $4.40B  (+30%)
  Enterprise Software:$1.74B  (+15%)
  Professional Svcs:  $0.97B  (+15%)

KEY RISKS
1. Commoditization of AI features — competitors offering similar AI tooling at lower price points.
2. Regulatory compliance costs (EU AI Act fully effective; NIST AI RMF adoption in US contracts).
3. Data-center energy costs up 28% YoY, pressuring infrastructure margins.
4. Concentration in Cloud Services (62% of revenue) increases segment-specific risk exposure.
5. Macroeconomic slowdown risk — enterprise IT budgets under review in 40% of Fortune 500 accounts.

EXECUTIVE LEADERSHIP
CEO:   Sarah Chen        (since 2019)
CFO:   Rachel Torres     (appointed Q2 2025, replaced Marcus Webb)
COO:   James Okafor      (since 2024)
CTO:   Daniel Kowalski   (since 2020)
Chief AI Officer: Lena Park (since 2024)
""",
    2026: """\
TechVision Corp — Annual Report 2026

LETTER TO SHAREHOLDERS
Dear Shareholders,
2026 delivered another year of double-digit growth. TechVision AI Studio reached 10,000 enterprise
deployments and we launched our sovereign-cloud offering in five new countries.

FINANCIAL HIGHLIGHTS
Total Revenue:        $8.63 billion  (+21% YoY)
Gross Profit:         $4.40 billion  (51.0% margin)
Operating Income:     $1.47 billion  (17.0% margin)
Net Income:           $1.12 billion
EPS (diluted):        $6.91
Free Cash Flow:       $1.55 billion
Headcount (year-end): 28,900

SEGMENT REVENUE
  Cloud Services:     $5.46B  (+24%)
  Enterprise Software:$2.07B  (+19%)
  Professional Svcs:  $1.10B  (+13%)

KEY RISKS
1. Sovereign-cloud expansion exposes TechVision to new jurisdictional data-residency obligations.
2. Increasing AI regulation globally — 23 countries now have enacted AI-specific legislation.
3. Model hallucination liability: three enterprise clients filed claims in 2026 over AI output errors.
4. Supply-chain risk: sole-source dependency on two semiconductor vendors for proprietary AI chips.
5. Executive succession planning — CEO Sarah Chen announced intent to transition by end of 2027.

EXECUTIVE LEADERSHIP
CEO:   Sarah Chen        (since 2019; transition planned for 2027)
CFO:   Rachel Torres     (since 2025)
COO:   James Okafor      (since 2024)
CTO:   Nina Vasquez      (appointed 2026, replaced Daniel Kowalski)
Chief AI Officer: Lena Park (since 2024)
""",
}

SAMPLE_QUESTIONS = [
    "What was the total revenue in respective years?",
    "List the top 3 risks mentioned.",
    "Who are the key executives?",
    "Sort revenue by year (from highest to lowest)",
]


# ---------- Utility Functions ----------
def get_backend_url(endpoint: str) -> str:
    return BACKEND_URLS.get(endpoint, BACKEND_URLS["query"])


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def clean_answer(answer: str) -> str:
    if '[INST]' in answer:
        answer = answer[:answer.index('[INST]')]
    for marker in ['<<SYS>>', '<</SYS>>', '<</SYS']:
        if marker in answer:
            answer = answer[:answer.index(marker)]
    return answer.strip()


# ---------- Backend Communication ----------
def check_health() -> Tuple[bool, Optional[str], Optional[int]]:
    """Returns (index_exists, indexed_filename, indexed_char_count)."""
    try:
        resp = requests.get(get_backend_url("health"), timeout=WARMUP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return (
                data.get("index_exists", False),
                data.get("indexed_filename", None),
                data.get("indexed_char_count", None),
            )
        return False, None, None
    except Exception:
        return False, None, None


def _upload_text(text_content: str, filename: str) -> Tuple[bool, Optional[str], Optional[int], str]:
    payload = {"text": text_content, "filename": filename}
    try:
        response = requests.post(
            get_backend_url("upload"),
            json=payload,
            timeout=UPLOAD_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            char_count = data.get("char_count")
            return True, filename, char_count, f"✅ Indexed **{filename}**"
        return False, None, None, f"❌ {data.get('message', 'Upload failed')}"
    except requests.exceptions.Timeout:
        return False, None, None, "❌ Upload timed out. Please try a smaller file."
    except Exception as e:
        return False, None, None, f"❌ Upload failed: {str(e)}"


def upload_document(file_obj) -> Tuple[bool, Optional[str], Optional[int], str]:
    if file_obj is None:
        return False, None, None, "❌ No file selected."
    file_size = os.path.getsize(file_obj.name) if os.path.exists(file_obj.name) else 0
    if file_size > MAX_FILE_SIZE:
        return False, None, None, f"❌ File too large. Max {format_file_size(MAX_FILE_SIZE)}."
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            text_content = f.read()
    except UnicodeDecodeError:
        return False, None, None, "❌ Could not decode file. Please use a UTF-8 text file."
    if not text_content or not text_content.strip():
        return False, None, None, "❌ File is empty."
    filename = os.path.basename(file_obj.name)
    return _upload_text(text_content, filename)


def upload_pasted_text(text: str, filename: str = "pasted.txt") -> Tuple[bool, Optional[str], Optional[int], str]:
    """Index user-pasted text directly."""
    if not text or not text.strip():
        return False, None, None, "⚠️ Please paste some text first."
    filename = (filename or "pasted.txt").strip() or "pasted.txt"
    if not filename.endswith(".txt"):
        filename += ".txt"
    return _upload_text(text, filename)


def load_selected_samples(selected_labels: list) -> Tuple[bool, Optional[str], Optional[int], str]:
    """Combine and upload one or more sample annual reports."""
    if not selected_labels:
        return False, None, None, "⚠️ Please select at least one report."
    years = sorted([SAMPLE_LABELS[lbl] for lbl in selected_labels])
    parts = []
    for year in years:
        separator = "=" * 60
        parts.append(f"{separator}\nANNUAL REPORT {year}\n{separator}\n{SAMPLE_DOCUMENTS[year]}")
    combined = "\n\n".join(parts)
    if len(years) == 1:
        filename = f"annual_report_{years[0]}.txt"
    else:
        filename = f"annual_reports_{'_'.join(str(y) for y in years)}.txt"
    return _upload_text(combined, filename)


def build_preview_text(selected_labels: list) -> Tuple[str, str]:
    """Return (text, text) — same value for the visible textbox and hidden state."""
    if not selected_labels:
        return "", ""
    years = sorted([SAMPLE_LABELS[lbl] for lbl in selected_labels])
    parts = []
    for year in years:
        separator = "=" * 60
        parts.append(f"{separator}\nANNUAL REPORT {year}\n{separator}\n{SAMPLE_DOCUMENTS[year]}")
    text = "\n\n".join(parts)
    return text, text


def query_backend(message: str) -> Generator[str, None, None]:
    if not message or not message.strip():
        return
    if len(message) > MAX_QUESTION_LENGTH:
        message = message[:MAX_QUESTION_LENGTH]
    yield "🤔 Thinking..."
    try:
        response = requests.post(
            get_backend_url("query"),
            json={"question": message},
            timeout=QUERY_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        answer = clean_answer(data.get("answer", "⚠️ Received an empty response."))
        elapsed = data.get("elapsed")
        if elapsed is not None:
            answer += f"\n\n<sub style='color:#888'>⏱️ {elapsed:.1f} seconds</sub>"
        yield answer
    except requests.exceptions.Timeout:
        yield "❌ Query timed out. Please try again."
    except requests.exceptions.ConnectionError:
        yield "❌ Cannot connect to backend. Please try again later."
    except Exception as e:
        yield f"❌ Error: {str(e)}"


def update_index_display(filename: Optional[str], is_indexed: bool, char_count: Optional[int] = None) -> str:
    if is_indexed and filename:
        detail = f" ({char_count:,} characters)" if char_count else ""
        return f"📄 **Indexed** {filename}{detail}"
    elif is_indexed:
        return "📄 **Indexed** (document ready)"
    return "📄 *No document indexed yet*"


def load_index_state() -> Tuple[bool, Optional[str], Optional[int], str]:
    is_indexed, filename, char_count = check_health()
    return is_indexed, filename, char_count, update_index_display(filename, is_indexed, char_count)


def warmup_with_status(start_time: float = None) -> Generator[Tuple[str, bool, str, float], None, None]:
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

    threading.Thread(target=do_request, daemon=True).start()

    STEPS = [
        (0, "⏳ Initializing GPU model (Qwen 2.5 3B)..."),
        (3, "⏳ Initializing GPU model (Qwen 2.5 3B)...\n✅ Embedding model ready (BGE-small)"),
        (6, "⏳ Initializing GPU model (Qwen 2.5 3B)...\n✅ Embedding model ready (BGE-small)\n⏳ Loading index from volume..."),
        (9, "✅ GPU model loaded (Qwen 2.5 3B)\n✅ Embedding model ready (BGE-small)\n⏳ Loading index from volume..."),
    ]

    while not result["done"]:
        elapsed = time.time() - start_time
        step_msg = STEPS[0][1]
        for threshold, msg in STEPS:
            if elapsed >= threshold:
                step_msg = msg
        yield f"{step_msg}\n\n*({elapsed:.1f}s elapsed)*", False, f"⏳ Warming up... ({elapsed:.1f}s)", start_time
        time.sleep(1)

    elapsed = time.time() - start_time
    if result["success"]:
        final = (
            "✅ GPU model loaded (Qwen 2.5 3B)\n"
            "✅ Embedding model ready (BGE-small)\n"
            "✅ Index volume ready\n\n"
            f"**✅ Backend ready in {elapsed:.1f}s**"
        )
        yield final, True, f"✅ Ready ({elapsed:.1f}s)", start_time
    elif result["error"]:
        yield f"❌ Cannot connect: {result['error']} ({elapsed:.1f}s)", False, f"❌ Failed ({elapsed:.1f}s)", start_time
    else:
        yield f"❌ Backend returned status {result['status_code']} ({elapsed:.1f}s)", False, f"❌ Failed ({elapsed:.1f}s)", start_time


# ---------- Main App ----------
def create_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        page = gr.State("warmup")
        document_indexed = gr.State(False)
        indexed_filename = gr.State(None)
        indexed_char_count = gr.State(None)

        # ── Warmup page ──────────────────────────────────────────────────────
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
                    warmup_btn = gr.Button("🔥 Warm Up Backend", variant="primary", size="lg")

            warmup_status = gr.Markdown("")
            ready = gr.State(False)
            start_time_state = gr.State(0.0)

            warmup_btn.click(
                fn=lambda: time.time(),
                outputs=[start_time_state],
            ).then(
                fn=warmup_with_status,
                inputs=[start_time_state],
                outputs=[warmup_status, ready, warmup_btn, start_time_state],
            )

        # ── Chat page ────────────────────────────────────────────────────────
        with gr.Column(elem_id="chat-page", visible=False) as chat_page:
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 1em;">
                    <h1>📚 RAG Chat Interface</h1>
                </div>
                """
            )

            index_status = gr.Markdown("📄 *No document indexed yet*")

            # Define respond up-front so any button below can reference it
            def respond(message, history, is_indexed):
                history = history or []
                if not message or not message.strip():
                    yield history, message
                    return
                user_msg = {"role": "user", "content": message}
                if not is_indexed:
                    yield history + [user_msg, {"role": "assistant", "content": "⚠️ Please index a document first (see section 1)."}], ""
                    return
                yield history + [user_msg, {"role": "assistant", "content": "🤔 Thinking..."}], ""
                final_answer = "🤔 Thinking..."
                for chunk in query_backend(message):
                    final_answer = chunk
                    yield history + [user_msg, {"role": "assistant", "content": final_answer}], ""

            # Returns (indexed?, filename, char_count, display_md, preview_text)
            def _after_index(success, filename, char_count, source_text):
                display = update_index_display(filename, success, char_count)
                preview = source_text if success else ""
                return success, filename, char_count, display, preview

            # ── Section 1: Select files to index ──────────────────────────
            with gr.Accordion("📥 1. Select files to index", open=True):
                with gr.Tabs():
                    # — Tab A: sample files —
                    with gr.Tab("📋 Sample reports"):
                        sample_selector = gr.CheckboxGroup(
                            choices=list(SAMPLE_LABELS.keys()),
                            value=[],
                            label="Pick one or more sample annual reports",
                            interactive=True,
                        )
                        load_samples_btn = gr.Button("🚀 Index Selected Reports", variant="primary")

                    # — Tab B: paste text —
                    with gr.Tab("✏️ Paste text"):
                        paste_input = gr.Textbox(
                            label="Paste text here",
                            lines=10,
                            max_lines=20,
                            placeholder="Paste any text content you want to ask questions about…",
                        )
                        paste_filename = gr.Textbox(
                            label="Filename (optional)",
                            value="pasted.txt",
                            lines=1,
                        )
                        load_paste_btn = gr.Button("🚀 Index Pasted Text", variant="primary")

                    # — Tab C: upload file —
                    with gr.Tab("📁 Upload file"):
                        file_input = gr.File(
                            label="Upload .txt file",
                            file_types=[".txt"],
                            file_count="single",
                        )
                        upload_btn = gr.Button("🚀 Index Uploaded File", variant="primary")

            # ── Section 2: Preview indexed text ───────────────────────────
            with gr.Accordion("📖 2. Preview indexed text", open=False):
                preview_box = gr.Textbox(
                    value="",
                    lines=18,
                    max_lines=18,
                    interactive=False,
                    show_label=False,
                    placeholder="Index a document in section 1 to see its content here.",
                )

            # ── Section 3: Chat ────────────────────────────────────────────
            with gr.Accordion("💬 3. Ask questions", open=True):
                gr.Markdown("<small style='color:#888'>Quick questions:</small>")
                with gr.Row():
                    q_btns = [gr.Button(q, size="sm") for q in SAMPLE_QUESTIONS]

                chatbot = gr.Chatbot(height=420, show_label=False, value=[])

                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Type your question and press Enter…",
                        lines=1,
                        scale=9,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button("Send ▶", scale=1, variant="primary")

            # ── Wiring for Section 1 actions ──────────────────────────────
            def _samples_action(selected_labels):
                success, filename, char_count, _msg = load_selected_samples(selected_labels)
                source_text = build_preview_text(selected_labels)[0] if success else ""
                return _after_index(success, filename, char_count, source_text)

            load_samples_btn.click(
                fn=_samples_action,
                inputs=[sample_selector],
                outputs=[document_indexed, indexed_filename, indexed_char_count, index_status, preview_box],
            )

            def _paste_action(text, filename):
                success, fn_used, char_count, _msg = upload_pasted_text(text, filename)
                return _after_index(success, fn_used, char_count, text if success else "")

            load_paste_btn.click(
                fn=_paste_action,
                inputs=[paste_input, paste_filename],
                outputs=[document_indexed, indexed_filename, indexed_char_count, index_status, preview_box],
            )

            def _upload_action(file_obj):
                success, fn_used, char_count, _msg = upload_document(file_obj)
                source_text = ""
                if success and file_obj is not None and os.path.exists(file_obj.name):
                    try:
                        with open(file_obj.name, "r", encoding="utf-8") as f:
                            source_text = f.read()
                    except Exception:
                        source_text = ""
                return _after_index(success, fn_used, char_count, source_text)

            upload_btn.click(
                fn=_upload_action,
                inputs=[file_input],
                outputs=[document_indexed, indexed_filename, indexed_char_count, index_status, preview_box],
            )

            # Quick-question buttons: fill the textbox, then auto-submit
            for btn, q in zip(q_btns, SAMPLE_QUESTIONS):
                btn.click(
                    fn=lambda x=q: x,
                    outputs=[question_input],
                ).then(
                    fn=respond,
                    inputs=[question_input, chatbot, document_indexed],
                    outputs=[chatbot, question_input],
                )

            send_btn.click(
                fn=respond,
                inputs=[question_input, chatbot, document_indexed],
                outputs=[chatbot, question_input],
            )
            question_input.submit(
                fn=respond,
                inputs=[question_input, chatbot, document_indexed],
                outputs=[chatbot, question_input],
            )

        # Page switch when warmup completes
        ready.change(
            fn=lambda r: (gr.update(visible=not r), gr.update(visible=r), "chat" if r else "warmup"),
            inputs=[ready],
            outputs=[warmup_page, chat_page, page],
        )

        # Restore index state on page load
        demo.load(
            fn=load_index_state,
            outputs=[document_indexed, indexed_filename, indexed_char_count, index_status],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
