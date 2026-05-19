"""
RAG Chat Interface for Hugging Face Space — Gradio UI layer.
All pure logic and backend HTTP lives in utils.py for testability.
"""
import gradio as gr
import os
import time
import threading
import requests
from typing import Generator, Tuple

from utils import (
    BACKEND_URLS,
    MAX_FILE_SIZE,
    load_sample_data,
    get_backend_url,
    format_file_size,
    extract_text_from_file,
    resolve_file_path,
    check_health,
    fetch_indexed_docs,
    upload_text_to_backend,
    upload_pasted_text,
    query_backend,
    update_index_display,
    load_index_state,
    delete_document,
)


# ---------- Sample Data ----------
_SAMPLE_DATA = load_sample_data()
SAMPLE_LABELS: dict    = _SAMPLE_DATA["labels"]
SAMPLE_DOCUMENTS: dict = {int(k): v for k, v in _SAMPLE_DATA["documents"].items()}
SAMPLE_QUESTIONS: list = _SAMPLE_DATA["sample_questions"]


# ---------- Sample-specific helpers ----------

def load_selected_samples(selected_labels: list) -> Tuple[bool, str, int, str]:
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
    return upload_text_to_backend(combined, filename)


def build_preview_text(selected_labels: list) -> str:
    if not selected_labels:
        return ""
    years = sorted([SAMPLE_LABELS[lbl] for lbl in selected_labels])
    parts = []
    for year in years:
        separator = "=" * 60
        parts.append(f"{separator}\nANNUAL REPORT {year}\n{separator}\n{SAMPLE_DOCUMENTS[year]}")
    return "\n\n".join(parts)


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
        page             = gr.State("warmup")
        document_indexed = gr.State(False)
        indexed_docs     = gr.State([])
        active_docs      = gr.State([])   # selected docs for query scope; [] = all

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

            def respond(message, history, is_indexed, current_docs):
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
                # current_docs is a list (possibly empty = search all)
                for chunk in query_backend(message, current_docs):
                    final_answer = chunk
                    yield history + [user_msg, {"role": "assistant", "content": final_answer}], ""

            def _after_index(success, doc_id, char_count, source_text, error_msg=""):
                """
                Shared post-index handler.
                Returns: (indexed, docs, active_docs, index_status, preview_box, dropdown_update)
                The freshly-uploaded doc is added to the active selection.
                """
                if success:
                    docs = fetch_indexed_docs()
                    new_active = [doc_id] if doc_id in docs else docs[:]
                    display = update_index_display(docs)
                    preview = source_text or ""
                    return (
                        True,
                        docs,
                        new_active,
                        display,
                        preview,
                        gr.update(choices=docs, value=new_active),
                    )
                else:
                    display = error_msg or "📄 *No document indexed yet*"
                    return False, gr.update(), gr.update(), display, "", gr.update()

            # ── Section 1: Select files to index ──
            with gr.Accordion("📥 1. Select files to index", open=True):
                with gr.Tabs():
                    with gr.Tab("📋 Sample reports"):
                        sample_selector = gr.CheckboxGroup(
                            choices=list(SAMPLE_LABELS.keys()),
                            value=[],
                            label="Pick one or more sample annual reports",
                            interactive=True,
                        )
                        load_samples_btn = gr.Button("🚀 Index Selected Reports", variant="primary")
                        samples_status = gr.Markdown("")

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
                        paste_status = gr.Markdown("")

                    with gr.Tab("📁 Upload file"):
                        file_input = gr.File(
                            label="Upload .txt, .pdf, or .docx (one or many)",
                            file_types=[".txt", ".pdf", ".docx"],
                            file_count="multiple",
                        )
                        gr.Markdown(
                            "<small style='color:#888'>Drop one or more files at once. "
                            "PDF and DOCX are parsed locally; only the extracted text is sent. "
                            "Indexing starts automatically.</small>"
                        )
                        upload_status = gr.Markdown("")

            # ── Document Selector (multi-select for query scope) ──
            with gr.Row():
                active_doc_dropdown = gr.Dropdown(
                    label="📂 Documents to query (also targets for Remove)",
                    choices=[],
                    value=[],
                    multiselect=True,
                    interactive=True,
                    scale=5,
                    info="Leave empty to search across ALL indexed documents.",
                )
                refresh_docs_btn = gr.Button("🔄 Refresh", scale=1, min_width=80)
                delete_doc_btn   = gr.Button("🗑️ Remove selected", scale=1, min_width=120, variant="stop")
            delete_status = gr.Markdown("")

            # ── Section 2: Preview indexed text ──
            with gr.Accordion("📖 2. Preview indexed text", open=True):
                index_status = gr.Markdown("📄 *No documents indexed yet*")
                preview_box = gr.Textbox(
                    value="",
                    lines=18,
                    max_lines=18,
                    interactive=False,
                    label="Content",
                    placeholder="Index a document in section 1 to see its content here.",
                )

            # ── Section 3: Chat ──
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

            INDEX_OUTPUTS = [
                document_indexed,
                indexed_docs,
                active_docs,
                index_status,
                preview_box,
                active_doc_dropdown,
            ]

            SAMPLES_LABEL = "🚀 Index Selected Reports"
            PASTE_LABEL   = "🚀 Index Pasted Text"

            def _busy(label):
                return gr.update(value=label, interactive=False)

            def _ready_btn(label):
                return gr.update(value=label, interactive=True)

            def _samples_action(selected_labels):
                yield _busy("📖 Preparing reports..."), "📖 Preparing reports…", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                if not selected_labels:
                    result = _after_index(False, None, None, "", "⚠️ Please select at least one report.")
                    yield (_ready_btn(SAMPLES_LABEL), "⚠️ Please select at least one report.") + result
                    return
                yield _busy("📤 Uploading & indexing..."), "📤 Uploading & indexing…", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                success, doc_id, char_count, msg = load_selected_samples(selected_labels)
                source_text = build_preview_text(selected_labels) if success else ""
                result = _after_index(success, doc_id, char_count, source_text, msg)
                final_status = "✅ Done." if success else msg
                yield (_ready_btn(SAMPLES_LABEL), final_status) + result

            load_samples_btn.click(
                fn=_samples_action,
                inputs=[sample_selector],
                outputs=[load_samples_btn, samples_status] + INDEX_OUTPUTS,
            )

            def _paste_action(text, filename):
                yield _busy("📖 Preparing text..."), "📖 Preparing text…", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                if not text or not text.strip():
                    result = _after_index(False, None, None, "", "⚠️ Please paste some text first.")
                    yield (_ready_btn(PASTE_LABEL), "⚠️ Please paste some text first.") + result
                    return
                yield _busy("📤 Uploading & indexing..."), "📤 Uploading & indexing…", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                success, doc_id, char_count, msg = upload_pasted_text(text, filename)
                result = _after_index(success, doc_id, char_count, text if success else "", msg)
                final_status = "✅ Done." if success else msg
                yield (_ready_btn(PASTE_LABEL), final_status) + result

            load_paste_btn.click(
                fn=_paste_action,
                inputs=[paste_input, paste_filename],
                outputs=[load_paste_btn, paste_status] + INDEX_OUTPUTS,
            )

            def _process_one_file(file_obj):
                """Parse + upload one file. Returns (ok, doc_id, char_count, text, msg)."""
                path = resolve_file_path(file_obj)
                base = os.path.basename(path) if path else "?"
                if not path:
                    return False, None, None, "", f"❌ {base}: no file selected."
                if not os.path.exists(path):
                    return False, None, None, "", f"❌ {base}: not found."
                if os.path.getsize(path) > MAX_FILE_SIZE:
                    return False, None, None, "", f"❌ {base}: too large (max {format_file_size(MAX_FILE_SIZE)})."
                text_content, err = extract_text_from_file(path)
                if err or not text_content or not text_content.strip():
                    return False, None, None, "", err or f"❌ {base}: empty or unreadable."
                stem, _ext = os.path.splitext(base)
                filename = f"{stem}.txt"
                success, doc_id, char_count, msg = upload_text_to_backend(text_content, filename)
                if success:
                    return True, doc_id, char_count, text_content, f"✅ {doc_id} ({char_count:,} chars)"
                return False, None, None, "", f"❌ {filename}: {msg}"

            def _upload_action(file_objs):
                """Handle one or many files. Yields progress then a final summary."""
                # Normalise to a list — Gradio may pass a single file or a list.
                if file_objs is None:
                    files = []
                elif isinstance(file_objs, list):
                    files = file_objs
                else:
                    files = [file_objs]

                if not files:
                    yield ("❌ No files selected.",) + _after_index(
                        False, None, None, "", "❌ No files selected.",
                    )
                    return

                total = len(files)
                line_log: list = []
                newly_indexed: list = []   # doc_ids of successful uploads
                last_text = ""

                for i, fobj in enumerate(files, start=1):
                    base = os.path.basename(resolve_file_path(fobj) or "?")
                    yield (
                        f"📖 [{i}/{total}] Parsing **{base}**…\n\n" + "\n".join(line_log),
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(),
                    )
                    ok, doc_id, _cc, text_content, msg = _process_one_file(fobj)
                    line_log.append(msg)
                    if ok:
                        newly_indexed.append(doc_id)
                        last_text = text_content
                    yield (
                        f"📤 [{i}/{total}] Done.\n\n" + "\n".join(line_log),
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(),
                    )

                # Final state — select ALL successfully uploaded docs (not just the last).
                ok_count = len(newly_indexed)
                fail_count = total - ok_count

                if ok_count:
                    all_docs = fetch_indexed_docs()
                    selected = [d for d in newly_indexed if d in all_docs]
                    if not selected:
                        selected = all_docs[:]  # fallback
                    final_tuple = (
                        True,
                        all_docs,
                        selected,
                        update_index_display(all_docs),
                        last_text,
                        gr.update(choices=all_docs, value=selected),
                    )
                else:
                    final_tuple = _after_index(False, None, None, "", "❌ All uploads failed.")

                header = (
                    f"✅ Indexed {ok_count}/{total} file{'s' if total != 1 else ''}."
                    if fail_count == 0
                    else f"⚠️ Indexed {ok_count}/{total} ({fail_count} failed)."
                )
                final_status = header + "\n\n" + "\n".join(line_log)
                yield (final_status,) + final_tuple

            file_input.upload(
                fn=_upload_action,
                inputs=[file_input],
                outputs=[upload_status] + INDEX_OUTPUTS,
            ).then(
                fn=lambda: gr.update(value=None),
                outputs=[file_input],
            )

            active_doc_dropdown.change(
                fn=lambda d: d or [],
                inputs=[active_doc_dropdown],
                outputs=[active_docs],
            )

            def _refresh_panel(current_active):
                """Re-fetch the doc list and keep selection where possible."""
                docs = fetch_indexed_docs()
                keep = [d for d in (current_active or []) if d in docs]
                return gr.update(choices=docs, value=keep), keep

            refresh_docs_btn.click(
                fn=_refresh_panel,
                inputs=[active_docs],
                outputs=[active_doc_dropdown, active_docs],
            )

            def _delete_action(current_active):
                """Delete every doc currently selected. Empty selection = no-op."""
                current_active = current_active or []
                if not current_active:
                    yield (
                        gr.update(),
                        "⚠️ Select at least one document to remove.",
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(),
                    )
                    return

                yield (
                    gr.update(),
                    f"🗑️ Removing {len(current_active)} document{'s' if len(current_active) != 1 else ''}...",
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                )

                removed = []
                failed = []
                remaining: list = []
                for name in current_active:
                    ok, remaining, _msg = delete_document(name)
                    (removed if ok else failed).append(name)

                if removed and not failed:
                    msg = f"🗑️ Removed {len(removed)} document{'s' if len(removed) != 1 else ''}."
                elif removed and failed:
                    msg = f"⚠️ Removed {len(removed)}, failed {len(failed)}."
                else:
                    msg = f"❌ Failed to remove {len(failed)} document{'s' if len(failed) != 1 else ''}."

                new_active = remaining[:] if remaining else []
                yield (
                    gr.update(choices=remaining, value=new_active),
                    msg,
                    len(remaining) > 0,
                    remaining,
                    new_active,
                    update_index_display(remaining),
                    "" if not remaining else gr.update(),
                )

            delete_doc_btn.click(
                fn=_delete_action,
                inputs=[active_docs],
                outputs=[
                    active_doc_dropdown,
                    delete_status,
                    document_indexed,
                    indexed_docs,
                    active_docs,
                    index_status,
                    preview_box,
                ],
            )

            for btn, q in zip(q_btns, SAMPLE_QUESTIONS):
                btn.click(
                    fn=lambda x=q: x,
                    outputs=[question_input],
                ).then(
                    fn=respond,
                    inputs=[question_input, chatbot, document_indexed, active_docs],
                    outputs=[chatbot, question_input],
                )

            send_btn.click(
                fn=respond,
                inputs=[question_input, chatbot, document_indexed, active_docs],
                outputs=[chatbot, question_input],
            )
            question_input.submit(
                fn=respond,
                inputs=[question_input, chatbot, document_indexed, active_docs],
                outputs=[chatbot, question_input],
            )

        ready.change(
            fn=lambda r: (gr.update(visible=not r), gr.update(visible=r), "chat" if r else "warmup"),
            inputs=[ready],
            outputs=[warmup_page, chat_page, page],
        )

        def _initial_load():
            from utils import check_health
            is_indexed, docs = check_health()
            # Default: nothing selected → query searches across all documents.
            return (
                is_indexed,
                docs,
                [],
                update_index_display(docs),
                gr.update(choices=docs, value=[]),
            )

        demo.load(
            fn=_initial_load,
            outputs=[
                document_indexed,
                indexed_docs,
                active_docs,
                index_status,
                active_doc_dropdown,
            ],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
