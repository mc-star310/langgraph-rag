"""
语料接入：把 corpus/ 下的多种格式读进来，清洗分块后喂给 HybridRAG。
支持：.md / .txt / .pdf(PyMuPDF) / .docx(python-docx)
新增格式只需在 READERS 里加一个分支即可（即插即用）。
"""

import os

# 延迟导入，避免在没有这些包的环境里直接 import 报错
_READERS_READY = False


def _ensure_readers():
    global _READERS_READY
    if _READERS_READY:
        return
    # 这里只是确认包可用；真正 import 在使用时做，保持轻量
    _READERS_READY = True


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)


def _read_docx(path: str) -> str:
    from docx import Document
    d = Document(path)
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


READERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def ingest_file(rag, path: str) -> int:
    """读取单个文件并加入索引，返回加入的字符数。"""
    ext = os.path.splitext(path)[1].lower()
    reader = READERS.get(ext)
    if not reader:
        print(f"[skip] 不支持的格式: {path}")
        return 0
    text = reader(path)
    rag.add_text(text, source=os.path.basename(path))
    return len(text)


def ingest_corpus(rag, corpus_dir: str) -> int:
    """遍历 corpus_dir，把支持的文件全部接入。返回总字符数。"""
    if not os.path.isdir(corpus_dir):
        print(f"[warn] 语料目录不存在: {corpus_dir}")
        return 0
    total = 0
    for name in sorted(os.listdir(corpus_dir)):
        if name.startswith("."):
            continue
        p = os.path.join(corpus_dir, name)
        if os.path.isfile(p):
            total += ingest_file(rag, p)
    print(f"[ingest] 完成，共接入 {total} 字符，当前 {rag.stats()['num_chunks']} 个文本块")
    return total
