"""Document text extraction and validation for .pdf and .txt files.

Performs extension validation, size boundary checks, control character stripping,
and empty-file rejection.
"""

import io
from pathlib import Path
import re
from typing import Tuple, Union
import pypdf

from src.config import get_settings
from src.safety.logger import get_logger

logger = get_logger(__name__)


class DocumentExtractionError(Exception):
    """Raised when document extraction fails validation or parsing."""
    pass


class DocumentExtractor:
    """Extracts and sanitizes text from uploaded PDF and TXT documents."""

    def __init__(self):
        self.settings = get_settings()

    def validate_file(self, filename: str, file_bytes: bytes) -> None:
        """Validate file extension and size boundaries."""
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise DocumentExtractionError("Uploaded file is empty (0 bytes).")

        if len(file_bytes) > self.settings.max_upload_size_bytes:
            raise DocumentExtractionError(
                f"File size exceeds maximum allowed limit of {self.settings.MAX_UPLOAD_SIZE_MB}MB."
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in self.settings.ALLOWED_DOCUMENT_EXTENSIONS:
            raise DocumentExtractionError(
                f"Unsupported file extension '{suffix}'. Allowed types: {', '.join(sorted(self.settings.ALLOWED_DOCUMENT_EXTENSIONS))}"
            )

    def clean_text(self, text: str) -> str:
        """Strip control characters, excessive whitespace, and normalize linebreaks."""
        # Replace non-printable control characters except standard tabs and newlines
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        # Normalize carriage returns
        sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive newlines (more than 3 into 2)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def extract_text(self, filename: str, file_bytes: bytes) -> Tuple[str, str]:
        """Validate and extract sanitized plain text. Returns (cleaned_text, file_type)."""
        self.validate_file(filename, file_bytes)
        suffix = Path(filename).suffix.lower().lstrip(".")

        if suffix == "txt":
            try:
                raw_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback to latin-1
                raw_text = file_bytes.decode("latin-1", errors="replace")
        elif suffix == "pdf":
            try:
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages = []
                for idx, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(page_text)
                raw_text = "\n\n".join(pages)
            except Exception as exc:
                raise DocumentExtractionError(f"Failed to parse PDF document: {exc}")
        else:
            raise DocumentExtractionError(f"Unsupported document format: {suffix}")

        cleaned = self.clean_text(raw_text)
        if not cleaned:
            raise DocumentExtractionError("No readable text could be extracted from the document.")

        return cleaned, suffix
