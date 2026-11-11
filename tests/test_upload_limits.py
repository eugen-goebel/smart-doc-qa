"""Tests for the upload size limits and the .docx zip bomb guard.

Real zip bombs are never written: the unpacked-size limit is lowered with
monkeypatch so a few MB of padding stand in for gigabytes.
"""

import io
import os
import struct
import zipfile
import zlib

import pypdf
import pypdf.errors
import pytest
import tomllib
from docx import Document
from streamlit.testing.v1 import AppTest

import agents.document_loader as loader_module
from agents.document_loader import DocumentLoader

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_DIR, "app.py")

MB = 1024 * 1024


def make_docx(path, padding=0, forge_header=False):
    """Write a valid .docx; `padding` extra bytes of whitespace go into document.xml.

    With forge_header=True the ZIP headers claim the original small size (and a
    matching CRC) while the compressed stream still holds all the padding bytes.
    """
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("Hello from the test document.")
    doc.save(buf)
    src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    original = src.read("word/document.xml")
    padded = original + b" " * padding  # trailing whitespace is valid XML

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            z.writestr(
                info.filename,
                padded if info.filename == "word/document.xml" else src.read(info.filename),
            )
    raw = bytearray(out.getvalue())

    if forge_header:
        crc = zlib.crc32(padded[: len(original)])
        pos = 0
        while (ci := raw.find(b"PK\x01\x02", pos)) >= 0:
            pos = ci + 4
            name_len = struct.unpack_from("<H", raw, ci + 28)[0]
            if raw[ci + 46 : ci + 46 + name_len] != b"word/document.xml":
                continue
            local = struct.unpack_from("<I", raw, ci + 42)[0]
            for base, crc_off, size_off in ((ci, 16, 24), (local, 14, 22)):
                struct.pack_into("<I", raw, base + crc_off, crc)
                struct.pack_into("<I", raw, base + size_off, len(original))

    with open(path, "wb") as f:
        f.write(raw)
    return str(path)


def make_pdf_bomb(path, raw_size):
    """Write a valid one-page PDF whose Flate content stream unpacks to `raw_size` bytes.

    The stream is compressed in chunks, so the raw data is never held in memory at once.
    """
    comp = zlib.compressobj(9)
    body = comp.compress(b"BT /F1 12 Tf 50 700 Td (Hello) Tj ET\n")
    chunk = b" " * MB
    for _ in range(raw_size // MB):
        body += comp.compress(chunk)
    body += comp.flush()

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(body) + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)

    with open(path, "wb") as f:
        f.write(out)
    return str(path)


@pytest.fixture
def loader():
    return DocumentLoader()


# --- Zip bomb guard ---


class TestDocxZipBomb:
    def test_normal_docx_still_loads(self, loader, tmp_path):
        path = make_docx(tmp_path / "ok.docx")
        doc = loader.load(path)
        assert "Hello from the test document." in doc.text

    def test_oversized_docx_is_rejected(self, loader, tmp_path, monkeypatch):
        monkeypatch.setattr(loader_module, "MAX_DOCX_UNPACKED_BYTES", 1 * MB, raising=False)
        path = make_docx(tmp_path / "big.docx", padding=3 * MB)

        with pytest.raises(ValueError, match="zip bomb"):
            loader.load(path)

    def test_forged_header_does_not_bypass_the_limit(self, loader, tmp_path, monkeypatch):
        monkeypatch.setattr(loader_module, "MAX_DOCX_UNPACKED_BYTES", 1 * MB, raising=False)
        path = make_docx(tmp_path / "forged.docx", padding=3 * MB, forge_header=True)

        # The declared sizes look harmless, so a header-only check would pass this file
        with zipfile.ZipFile(path) as zf:
            assert sum(i.file_size for i in zf.infolist()) < 1 * MB

        with pytest.raises(ValueError, match="zip bomb"):
            loader.load(path)


class TestBrokenDocx:
    def test_non_zip_content_gives_clear_error(self, loader, tmp_path):
        path = tmp_path / "fake.docx"
        path.write_bytes(b"this is plain text, not a zip archive")

        with pytest.raises(ValueError, match="Not a valid .docx"):
            loader.load(str(path))

    def test_truncated_docx_gives_clear_error(self, loader, tmp_path):
        good = make_docx(tmp_path / "good.docx")
        with open(good, "rb") as f:
            data = f.read()
        cut = tmp_path / "cut.docx"
        cut.write_bytes(data[: len(data) // 2])

        with pytest.raises(ValueError, match="Not a valid .docx"):
            loader.load(str(cut))


# --- PDF decompression bomb ---


class TestPdfDecompressionLimit:
    def test_normal_generated_pdf_still_loads(self, loader, tmp_path):
        # Guards the helper: the bomb below must fail because of its size, not a broken PDF
        path = make_pdf_bomb(tmp_path / "ok.pdf", raw_size=MB)
        assert "Hello" in loader.load(path).text

    def test_loader_relies_on_pypdf_limit_for_oversized_stream(self, loader, tmp_path):
        # The loader has no PDF size check of its own: pypdf's zlib limit is the only guard
        limit = pypdf.Configuration().zlib_maximum_output_length
        path = make_pdf_bomb(tmp_path / "bomb.pdf", raw_size=limit + 5 * MB)
        assert os.path.getsize(path) < 1 * MB  # ~80 MB of whitespace deflates to ~80 KB

        with pytest.raises(pypdf.errors.LimitReachedError):
            loader.load(path)


# --- Upload size limit ---


class TestUploadLimit:
    def test_config_and_constant_agree(self):
        with open(os.path.join(REPO_DIR, ".streamlit", "config.toml"), "rb") as f:
            config = tomllib.load(f)
        assert config["server"]["maxUploadSize"] == loader_module.MAX_UPLOAD_MB == 10

    @pytest.fixture()
    def app(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        return at

    def test_oversized_file_gets_a_clear_error(self, tmp_path, monkeypatch):
        # Lower limit keeps the run fast even if the guard were missing
        monkeypatch.setattr(loader_module, "MAX_UPLOAD_MB", 1, raising=False)
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()

        big = b"a" * (1 * MB + 1)
        at.sidebar.file_uploader[0].upload("huge.txt", big, "text/plain").run()

        assert not at.exception
        messages = [e.value for e in at.error]
        assert any("huge.txt" in m and "too large" in m for m in messages), messages
        assert "huge.txt" not in at.session_state["uploaded_files"]

    def test_file_within_limit_is_indexed(self, app):
        app.sidebar.file_uploader[0].upload(
            "small.txt", b"Revenue grew in 2025.", "text/plain"
        ).run()

        assert not app.exception
        assert "small.txt" in app.session_state["uploaded_files"]
