# RAG Knowledge Base

A self-hosted retrieval-augmented Q&A system over uploaded documents.
**Self-compiled CUDA llama.cpp + LlamaIndex on serverless GPU, behind a Gradio UI on Hugging Face Spaces.**

> Designed and built as a portfolio piece for IT team lead / engineering management roles.
> The infrastructure choices below — particularly the GPU cold-start optimisation and the unified-index architecture — are the centrepieces, not the code volume.

---

## The problem

Enterprise knowledge sits in PDFs, Word documents and plain-text runbooks that employees can't search. Off-the-shelf chatbots either need a SaaS provider to ingest your documents (security risk) or require months of MLOps work to host privately.

This project demonstrates a third path: **a self-hosted RAG service where your documents never leave your infrastructure, with GPU cold-starts measured in seconds rather than minutes**, deployable to commodity serverless GPU (Modal) without managing Kubernetes or a long-running cluster.

---

## Architecture

```
┌─────────────────────────────────┐         ┌──────────────────────────────────────┐
│  Hugging Face Space (Gradio)    │         │       Modal serverless GPU           │
│                                 │         │                                      │
│  • upload (.txt/.pdf/.docx)     │         │  ┌────────────────────────────────┐  │
│  • parse locally in-browser     │         │  │  CUDA llama.cpp (T4 GPU)       │  │
│  • POST /upload  (text only)    │ ──HTTP──▶  │  Qwen 2.5 3B GGUF (Q4_K_M)     │  │
│  • POST /query   (question +    │         │  │  BGE-small embeddings (CPU)    │  │
│       optional doc filter list) │         │  │  LlamaIndex unified VectorStore│  │
│  • GET  /list, /health          │         │  └────────────────────────────────┘  │
│  • POST /delete                 │         │                                      │
│                                 │         │  Modal Volume persists the index     │
│  Multi-select scope dropdown    │         │  CPU memory snapshot for fast        │
│  Source citations in answers    │         │  cold-start                          │
└─────────────────────────────────┘         └──────────────────────────────────────┘
```

**Key property:** only the parsed *text* of uploaded documents leaves the browser. PDF/DOCX parsing happens client-side via `pypdf` and `python-docx`; the binary file itself never crosses the network. For a regulated environment that's the difference between "shippable" and "not".

---

## Architecture decisions

### 1. Unified vector index over per-document indexes
Earlier iterations kept each document in its own subdirectory. That made single-doc queries simple but **prevented cross-document synthesis** — the retriever could never see ticket A and ticket B in the same query, so questions like *"Compare 2024 and 2025 revenue"* were impossible.

The current design uses one `VectorStoreIndex` with all documents inside it. Metadata filters scope queries to specific documents when requested; without a filter, retrieval ranges across everything. Re-uploading a document with the same filename replaces the prior version via `delete_ref_doc` + `insert`.

### 2. CPU memory snapshot, GPU context excluded
Modal's `enable_memory_snapshot` can restore container state in <1 second. The first naive approach included the GGUF model in the snapshot: that pushed the snapshot to 2.1 GB and made restoration slower than a cold disk read. Worse, attempts to include the *GPU* context produced `SIGSEGV` (CUDA contexts don't survive process serialisation).

The solution: snapshot only Python imports + the 22 MB CPU-resident embedding model. The 2.1 GB GGUF loads from baked-in disk into VRAM on every cold start. Total cold-start time dropped from ~33 s to ~8 s with no crashes.

### 3. Self-compiled llama-cpp-python with CUDA
The default `pip install llama-cpp-python` wheel does not enable CUDA. The Modal image rebuilds llama-cpp from source with `GGML_CUDA=on` on a T4 worker (so `CMAKE_CUDA_ARCHITECTURES=75` is set correctly). Then the LlamaIndex wrapper is installed with `--no-deps` so it does not overwrite the CUDA-compiled binary.

### 4. Offline model loading after build
Once models are baked into the image, environment is set to `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No container start performs any HTTP call to Hugging Face. Determinism, plus protection against HF outages.

### 5. Frontend logic split from Gradio
All pure logic (CSV parsing, HTTP client, file-format dispatch) lives in `hf/utils.py` with no Gradio dependency. The test suite imports from there and runs without Gradio installed at all — important because Gradio's dependency chain frequently breaks across Python versions (we hit `audioop` removal in 3.13, then `HfFolder` removal in `huggingface_hub`).

---

## Known limitations

- **Single concurrent request.** `max_containers=1` and a thread lock around the LLM call. The bottleneck is GPU memory, not request rate. For higher throughput, raise `max_containers` and let Modal scale.
- **2,048-character question limit.** Set defensively to prevent prompt-injection-style payloads from blowing the context window. The model itself supports 32k tokens; lift this in `modal_backend_llamacpp.py` if you need it.
- **No authentication on the Modal endpoints.** This is a portfolio demo. In production you would put the endpoints behind Modal's `web_endpoint` auth, an API gateway, or a thin auth proxy.
- **No streaming responses yet.** Responses are returned whole; suitable for the 2–4-sentence answers the prompt requests, but a longer-form output would benefit from token streaming.
- **No PII redaction.** Documents are passed verbatim to the LLM. For regulated data the upload path should run a redaction step first.

---

## Non-goals

- **Multi-tenant SaaS.** Single-org self-hosted is the target use case. No per-user isolation, no quotas, no billing.
- **Replacing managed RAG vendors.** Pinecone + a hosted LLM gets you to production faster. This project's value is *control* and *cost*, not convenience.
- **Beating frontier LLM quality on retrieval.** Qwen 2.5 3B is deliberately small for cost / latency reasons. The architecture supports swapping the GGUF for a larger model (7B, 14B) with no code changes — just storage and a larger GPU.

---

## Quick start

### Deploy the backend

```bash
pip install modal
modal token new                                    # one-time auth

# Edit the app name in modal_backend_llamacpp.py if you want your own URL.
modal deploy modal_backend_llamacpp.py             # ~10 min on first deploy (builds CUDA image)
```

Note the printed endpoints — they look like `https://<account>--<label>.modal.run/`.

### Run the frontend locally

```bash
cd hf
pip install -r requirements.txt
# Edit BACKEND_URLS in utils.py to point at your deployment.
python app.py
```

Or push the `hf/` directory to a Hugging Face Space (Gradio SDK) for a public URL.

### Tests

```bash
pip install pytest requests pypdf python-docx
pytest tests/ -v
```

23 tests on the pure-logic and HTTP layers. **No Gradio install required.**

---

## What's in this repo

| Path | Purpose |
|---|---|
| `modal_backend_llamacpp.py` | Modal app — image build, GPU class, FastAPI endpoints |
| `hf/app.py` | Gradio UI — multi-doc panel, scope filter, chat |
| `hf/utils.py` | Pure functions: HTTP client, file parsing, formatting |
| `hf/requirements.txt` | Frontend dependencies (no Gradio version pin — HF Spaces forces its own) |
| `hf/data/sample_reports.json` | Four fictional annual reports for demo data |
| `tests/test_app.py` | Pytest suite — runs without Gradio or live backend |

---

## Tech choices

| Choice | Why |
|---|---|
| **Modal** | Cheapest path to serverless GPU; pay-per-invocation; volume mounts persist the index across restarts |
| **llama.cpp + Qwen 2.5 3B GGUF** | Fits in T4 VRAM with room to spare; Q4_K_M quantisation keeps quality high for short factual answers; CPU/GPU split runs everywhere |
| **LlamaIndex** | Cleanest RAG framework as of 2026; metadata filters and `delete_ref_doc` semantics are exactly what multi-doc needs |
| **BGE-small** | 22 MB, runs on CPU, produces respectable retrieval for English text; trivial to swap for a multilingual model |
| **Gradio on HF Spaces** | Free hosting, automatic SSL, native file-upload UI; lowest-friction way to publish a public demo |

---

## What I'd build next

1. **Streaming responses.** Token-by-token output for longer answers. LlamaIndex supports this via `streaming=True` on the query engine.
2. **Authentication.** Modal `web_endpoint(auth=...)` or a thin auth proxy in front of the endpoints.
3. **Per-document permission scoping.** Metadata-level ACLs — a query carries the user's allowed-document set, retrieval is filtered accordingly. The unified-index architecture already supports this.
4. **Hybrid search.** BM25 + vector with a reranker; particularly useful for queries containing specific identifiers (ticket numbers, error codes) that vector search alone retrieves poorly.
5. **Evaluation harness.** Following the pattern from the Sprint Analyzer project, build a small eval set of question/expected-source pairs and measure recall@k and answer faithfulness over time.

Each of these is a 1–3 day addition, not a rewrite. The architecture was chosen to make them additive.
