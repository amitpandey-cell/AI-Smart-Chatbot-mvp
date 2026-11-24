"""Sliding-window text chunker honoring paragraph and sentence boundaries.
"""

import re
from typing import List, Optional
from src.config import get_settings


class TextChunker:
    """Partitions text into overlapping segments respecting natural prose boundaries."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def _split_into_units(self, text: str) -> List[str]:
        """Split text into logical semantic units (paragraphs or sentences)."""
        paragraphs = text.split("\n\n")
        units: List[str] = []

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue

            if len(p_strip) <= self.chunk_size:
                units.append(p_strip)
            else:
                # Split large paragraphs by sentences
                sentences = re.split(r"(?<=[.!?])\s+", p_strip)
                for s in sentences:
                    s_strip = s.strip()
                    if s_strip:
                        units.append(s_strip)

        return units

    def chunk_text(self, text: str) -> List[str]:
        """Generate overlapping chunk windows from input text."""
        if not text or not text.strip():
            return []

        units = self._split_into_units(text.strip())
        if not units:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)

            # If a single unit itself exceeds chunk_size, cut it into character slices
            if unit_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0

                start = 0
                step = self.chunk_size - self.chunk_overlap
                while start < unit_len:
                    end = min(start + self.chunk_size, unit_len)
                    slice_str = unit[start:end].strip()
                    if slice_str:
                        chunks.append(slice_str)
                    start += max(1, step)
                continue

            # Check if adding unit exceeds chunk_size
            projected_len = current_len + unit_len + (1 if current_chunk else 0)
            if projected_len <= self.chunk_size:
                current_chunk.append(unit)
                current_len = projected_len
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())

                # Prepare overlap from tail of previous chunk
                overlap_units: List[str] = []
                overlap_len = 0
                for prev_unit in reversed(current_chunk):
                    if overlap_len + len(prev_unit) <= self.chunk_overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_len += len(prev_unit) + 1
                    else:
                        break

                current_chunk = overlap_units + [unit]
                current_len = sum(len(u) for u in current_chunk) + len(current_chunk) - 1

        if current_chunk:
            final_chunk_str = " ".join(current_chunk).strip()
            if not chunks or final_chunk_str != chunks[-1]:
                chunks.append(final_chunk_str)

        return chunks
