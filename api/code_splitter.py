from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from adalflow.core.component import DataComponent
from adalflow.components.data_process import TextSplitter
from adalflow.core.types import Document

logger = logging.getLogger(__name__)

_DEFINITION_TYPE_KEYWORDS = (
    "function",
    "method",
    "class",
    "interface",
    "struct",
    "enum",
    "trait",
    "impl",
    "module",
    "namespace",
    "type",
)


_EXT_TO_LANGUAGE: Dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "java": "java",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "hpp": "cpp",
    "cc": "cpp",
    "cs": "c_sharp",
    "go": "go",
    "rs": "rust",
    "php": "php",
    "rb": "ruby",
    "swift": "swift",
    "kt": "kotlin",
    "kts": "kotlin",
    "scala": "scala",
    "lua": "lua",
    "sh": "bash",
    "bash": "bash",
    "html": "html",
    "css": "css",
    "json": "json",
    "yml": "yaml",
    "yaml": "yaml",
    "toml": "toml",
    "md": "markdown",
}


@dataclass(frozen=True)
class CodeSplitterConfig:
    chunk_size_lines: int = 200
    chunk_overlap_lines: int = 20
    min_chunk_lines: int = 5
    enabled: bool = True


def _safe_import_tree_sitter() -> Tuple[Any, Any, Any]:
    module_candidates = [
        "tree_sitter_languages",  # module name used by tree-sitter-languages on most installs
    ]

    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
            get_parser = getattr(mod, "get_parser", None)
            if callable(get_parser):
                return get_parser, None, None
        except Exception:
            continue

    return None, None, None


def _iter_definition_like_nodes(root_node: Any) -> Iterable[Any]:
    for child in getattr(root_node, "children", []) or []:
        if not getattr(child, "is_named", False):
            continue
        node_type = getattr(child, "type", "") or ""
        lowered = node_type.lower()
        if any(k in lowered for k in _DEFINITION_TYPE_KEYWORDS):
            yield child


def _split_lines_with_overlap(
    lines: List[str], *, chunk_size_lines: int, chunk_overlap_lines: int
) -> List[Tuple[List[str], int]]:
    if chunk_size_lines <= 0:
        return [(lines, 0)]

    overlap = max(0, min(chunk_overlap_lines, chunk_size_lines - 1))
    chunks: List[Tuple[List[str], int]] = []
    start = 0
    n = len(lines)

    while start < n:
        end = min(n, start + chunk_size_lines)
        chunks.append((lines[start:end], start))
        if end >= n:
            break
        start = end - overlap

    return chunks


def _slice_text_by_bytes(text: str, start_byte: int, end_byte: int) -> str:
    b = text.encode("utf-8", errors="replace")
    return b[start_byte:end_byte].decode("utf-8", errors="replace")


def _byte_offset_to_line(text: str, byte_offset: int) -> int:
    prefix = _slice_text_by_bytes(text, 0, max(0, byte_offset))
    return prefix.count("\n") + 1


class TreeSitterCodeSplitter:
    def __init__(
        self,
        *,
        chunk_size_lines: int = 200,
        chunk_overlap_lines: int = 20,
        min_chunk_lines: int = 5,
        enabled: bool = True,
    ) -> None:
        self.config = CodeSplitterConfig(
            chunk_size_lines=chunk_size_lines,
            chunk_overlap_lines=chunk_overlap_lines,
            min_chunk_lines=min_chunk_lines,
            enabled=enabled,
        )
        self._get_parser, _, _ = _safe_import_tree_sitter()

    def is_available(self) -> bool:
        return self._get_parser is not None

    def split_document(self, doc: Document) -> List[Document]:
        if not self.config.enabled:
            return [doc]

        meta = getattr(doc, "meta_data", {}) or {}
        if not meta.get("is_code"):
            return [doc]

        file_type = (meta.get("type") or "").lower().lstrip(".")
        return self._split_code_text(doc.text or "", meta, file_type)

    def _get_language_name_candidates(self, file_type: str) -> List[str]:
        mapped = _EXT_TO_LANGUAGE.get(file_type)
        candidates: List[str] = []
        if mapped:
            candidates.append(mapped)
        if file_type and file_type not in candidates:
            candidates.append(file_type)
        return candidates

    def _try_get_parser(self, file_type: str) -> Any:
        if self._get_parser is None:
            return None

        for name in self._get_language_name_candidates(file_type):
            try:
                return self._get_parser(name)
            except Exception:
                continue
        return None

    def _split_code_text(self, text: str, meta: Dict[str, Any], file_type: str) -> List[Document]:
        parser = self._try_get_parser(file_type)
        if parser is None:
            return self._fallback_line_split(text, meta)

        try:
            tree = parser.parse(text.encode("utf-8", errors="replace"))
        except Exception:
            return self._fallback_line_split(text, meta)

        root = getattr(tree, "root_node", None)
        if root is None:
            return self._fallback_line_split(text, meta)

        nodes = list(_iter_definition_like_nodes(root))
        if not nodes:
            return self._fallback_line_split(text, meta)

        pieces: List[Tuple[str, int]] = []
        for node in nodes:
            try:
                start_b = int(getattr(node, "start_byte"))
                end_b = int(getattr(node, "end_byte"))
            except Exception:
                continue
            snippet = _slice_text_by_bytes(text, start_b, end_b)
            start_line = _byte_offset_to_line(text, start_b)
            pieces.append((snippet, start_line))

        if not pieces:
            return self._fallback_line_split(text, meta)

        docs: List[Document] = []
        for snippet, start_line in pieces:
            snippet_lines = snippet.splitlines(True)
            if len(snippet_lines) < self.config.min_chunk_lines:
                continue

            if len(snippet_lines) <= self.config.chunk_size_lines:
                docs.append(self._make_chunk_doc(snippet, meta, start_line))
                continue

            for sub, sub_start_idx in _split_lines_with_overlap(
                snippet_lines,
                chunk_size_lines=self.config.chunk_size_lines,
                chunk_overlap_lines=self.config.chunk_overlap_lines,
            ):
                sub_text = "".join(sub)
                docs.append(self._make_chunk_doc(sub_text, meta, start_line + sub_start_idx))

        if not docs:
            return self._fallback_line_split(text, meta)

        for i, d in enumerate(docs):
            d.meta_data["chunk_index"] = i
            d.meta_data["chunk_total"] = len(docs)

        return docs

    def _fallback_line_split(self, text: str, meta: Dict[str, Any]) -> List[Document]:
        lines = text.splitlines(True)
        docs: List[Document] = []
        for sub, start_idx in _split_lines_with_overlap(
            lines,
            chunk_size_lines=self.config.chunk_size_lines,
            chunk_overlap_lines=self.config.chunk_overlap_lines,
        ):
            sub_text = "".join(sub)
            if len(sub) < self.config.min_chunk_lines:
                continue
            start_line = 1 + start_idx
            docs.append(self._make_chunk_doc(sub_text, meta, start_line))

        if not docs:
            return [Document(text=text, meta_data=dict(meta))]

        for i, d in enumerate(docs):
            d.meta_data["chunk_index"] = i
            d.meta_data["chunk_total"] = len(docs)

        return docs

    def _make_chunk_doc(self, chunk_text: str, meta: Dict[str, Any], start_line: int) -> Document:
        new_meta = dict(meta)
        new_meta["chunk_start_line"] = start_line
        file_path = new_meta.get("file_path")
        if file_path:
            new_meta["title"] = str(file_path)
        return Document(text=chunk_text, meta_data=new_meta)


class CodeAwareSplitter(DataComponent):
    def __init__(
        self,
        *,
        text_splitter: TextSplitter,
        code_splitter: TreeSitterCodeSplitter,
    ) -> None:
        super().__init__()
        self._text_splitter = text_splitter
        self._code_splitter = code_splitter

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output: List[Document] = []
        for doc in documents:
            meta = getattr(doc, "meta_data", {}) or {}
            file_path = meta.get("file_path") or meta.get("title") or "<unknown>"
            is_code = bool(meta.get("is_code"))
            logger.info("Splitting document: %s (is_code=%s)", file_path, is_code)
            if is_code:
                chunks = self._code_splitter.split_document(doc)
                logger.info("Split result: %s -> %d chunks (code)", file_path, len(chunks))
                output.extend(chunks)
            else:
                logger.info("TextSplitter start: %s", file_path)
                chunks = list(self._text_splitter([doc]))
                logger.info("TextSplitter result: %s -> %d chunks", file_path, len(chunks))
                output.extend(chunks)
        return output
