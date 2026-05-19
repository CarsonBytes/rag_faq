"""
Tests for RAG Chat Interface — pure utility functions and backend HTTP layer.
Imports from utils.py only (no Gradio dependency).

Run from project root:
    pytest tests/ -v
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import requests as _requests

# Make hf/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hf"))

from utils import (
    clean_answer,
    format_file_size,
    extract_text_from_file,
    check_health,
    upload_pasted_text,
    update_index_display,
    fetch_indexed_docs,
)


# ---------- clean_answer ----------

class TestCleanAnswer:
    def test_strips_inst_marker(self):
        raw = "The answer is 42.[/INST] ignore this"
        assert clean_answer(raw) == "The answer is 42."

    def test_strips_im_end_marker(self):
        raw = "Revenue was $5B.<|im_end|>trailing"
        assert clean_answer(raw) == "Revenue was $5B."

    def test_strips_sys_marker(self):
        raw = "Paris.<<SYS>>system prompt leaked"
        assert clean_answer(raw) == "Paris."

    def test_no_marker_unchanged(self):
        raw = "A clean answer with no markers."
        assert clean_answer(raw) == raw

    def test_strips_leading_trailing_whitespace(self):
        raw = "  answer  "
        assert clean_answer(raw) == "answer"


# ---------- format_file_size ----------

class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(512) == "512 bytes"

    def test_kilobytes(self):
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"


# ---------- update_index_display ----------

class TestUpdateIndexDisplay:
    def test_empty_list(self):
        result = update_index_display([])
        assert "No documents" in result

    def test_single_doc(self):
        result = update_index_display(["report_2024.txt"])
        assert "1 document" in result
        assert "report_2024.txt" in result

    def test_multiple_docs(self):
        result = update_index_display(["a.txt", "b.txt", "c.txt"])
        assert "3 documents" in result

    def test_more_than_three_shows_overflow(self):
        docs = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
        result = update_index_display(docs)
        assert "and 2 more" in result


# ---------- extract_text_from_file ----------

class TestExtractTextFromFile:
    def test_reads_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello from test file.")
            path = f.name
        try:
            text, err = extract_text_from_file(path)
            assert err is None
            assert text == "Hello from test file."
        finally:
            os.unlink(path)

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            text, err = extract_text_from_file(path)
            assert text is None
            assert "Unsupported file type" in err
        finally:
            os.unlink(path)


# ---------- check_health (mocked HTTP) ----------

class TestCheckHealth:
    def test_returns_false_on_connection_error(self):
        with patch("utils.requests.get", side_effect=ConnectionError("refused")):
            indexed, docs = check_health()
        assert indexed is False
        assert docs == []

    def test_returns_docs_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "index_exists": True,
            "indexed_docs": ["report_2024.txt", "report_2025.txt"],
        }
        with patch("utils.requests.get", return_value=mock_resp):
            indexed, docs = check_health()
        assert indexed is True
        assert docs == ["report_2024.txt", "report_2025.txt"]

    def test_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("utils.requests.get", return_value=mock_resp):
            indexed, docs = check_health()
        assert indexed is False
        assert docs == []


# ---------- fetch_indexed_docs (mocked HTTP) ----------

class TestFetchIndexedDocs:
    def test_returns_docs_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "docs": ["a.txt", "b.txt"]}
        with patch("utils.requests.get", return_value=mock_resp):
            docs = fetch_indexed_docs()
        assert docs == ["a.txt", "b.txt"]

    def test_returns_empty_on_error(self):
        with patch("utils.requests.get", side_effect=ConnectionError("refused")):
            docs = fetch_indexed_docs()
        assert docs == []


# ---------- upload_pasted_text (input validation + mocked HTTP) ----------

class TestUploadPastedText:
    def test_empty_text_returns_error(self):
        success, doc_id, char_count, msg = upload_pasted_text("")
        assert success is False
        assert "paste some text" in msg

    def test_whitespace_only_returns_error(self):
        success, doc_id, char_count, msg = upload_pasted_text("   \n  ")
        assert success is False

    def test_successful_upload(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "filename": "pasted.txt",
            "doc_id": "pasted_txt",
            "char_count": 42,
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("utils.requests.post", return_value=mock_resp):
            success, doc_id, char_count, msg = upload_pasted_text("hello world", "pasted.txt")
        assert success is True
        assert char_count == 42

    def test_timeout_returns_error(self):
        with patch("utils.requests.post", side_effect=_requests.exceptions.Timeout):
            success, doc_id, char_count, msg = upload_pasted_text("x" * 100, "big.txt")
        assert success is False
        assert "timed out" in msg.lower()
