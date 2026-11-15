"""Document discovery, loading (PDF as in the original + new DOCX support) and chunking."""
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import config
from lang_utils import is_arabic, normalize_text

log = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class SourceFile:
    path: str
    source_id: str   # path relative to the docs folder (posix): unique even in sub-folders

    @property
    def file_name(self) -> str:
        return os.path.basename(self.path)


def is_indexable_name(name: str) -> bool:
    """Skips hidden files, Word lock files (~$x.docx), our upload temp files and unsupported types."""
    return (not name.startswith((".", "~$"))
            and os.path.splitext(name)[1].lower() in config.SUPPORTED_EXTENSIONS)


def discover_files(root: str) -> list[SourceFile]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if is_indexable_name(name):
                full = os.path.join(dirpath, name)
                found.append(SourceFile(full, os.path.relpath(full, root).replace(os.sep, "/")))
    return found


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_docx_sections(path: str) -> list[tuple[str | None, str]]:
    """(heading, text) per section in reading order; tables are flattened to 'a | b | c' rows."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    sections: list[tuple[str | None, str]] = []
    title, buf = None, []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((title, text))
        buf.clear()

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style.startswith("Heading") or style == "Title":
                flush()
                title = text
            buf.append(text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells, prev = [], None
                for cell in row.cells:
                    if cell._tc is prev:        # merged cells are returned repeatedly
                        continue
                    prev = cell._tc
                    value = cell.text.strip().replace("\n", " ")
                    if value:
                        cells.append(value)
                if cells:
                    buf.append(" | ".join(cells))
    flush()
    return sections


def load_documents(path: str):
    """Raw documents: one per PDF page (exactly as the original PyMuPDFLoader) or per DOCX section."""
    from langchain_core.documents import Document

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader  # handles Arabic PDFs better than pypdf
        return PyMuPDFLoader(path).load()
    if ext == ".docx":
        return [Document(page_content=text, metadata={"section": title})
                for title, text in extract_docx_sections(path)]
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_file(sf: SourceFile, file_hash: str, *, chunk_size: int, chunk_overlap: int, embed_model: str = ""):
    """Load, normalize (original Arabic cleanup) and split one file. Metadata keeps everything the
    original added (chunk_id, file_name, file_type, page_number, total_pages, language, ...)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = load_documents(sf.path)
    for d in docs:
        d.page_content = normalize_text(d.page_content)
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise EmptyDocumentError("no extractable text (scanned PDF? it needs OCR first)")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "۔", "؟", "؛", "،", ". ", "? ", "! ", " ", ""],
    )
    splits = text_splitter.split_documents(docs)

    mtime = os.path.getmtime(sf.path)
    modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for i, chunk in enumerate(splits):
        md = chunk.metadata
        md["source_id"] = sf.source_id
        md["chunk_index"] = i                              # 0-based, reading order inside the file
        md["chunk_id"] = f"{sf.source_id}::{i}"            # stable: does not shift when other files change
        md["total_chunks"] = len(splits)                   # lets sync detect half-indexed files
        md["file_name"] = sf.file_name
        md["file_type"] = os.path.splitext(sf.file_name)[1].lstrip(".").lower() or None
        md["file_hash"] = file_hash
        md["embed_model"] = embed_model
        md["page_number"] = md["page"] + 1 if "page" in md else None   # PDF: human-friendly (1-based)
        md.setdefault("total_pages", None)
        md.setdefault("section", None)                     # DOCX: nearest heading
        md["language"] = "ar" if is_arabic(chunk.page_content) else "en"
        md["char_count"] = len(chunk.page_content)
        md["modified_at"] = modified_at
        md["modified_ts"] = int(mtime)
    return splits
