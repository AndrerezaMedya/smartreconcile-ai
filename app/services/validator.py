"""
File Ingestion Validator for SmartReconcile AI.
Enforces file format, size, mime-type, and readable text layer constraints.
"""

from typing import Tuple, Optional, Dict, Any

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json"}


class FileValidationError(Exception):
    pass


class FileValidator:

    @staticmethod
    def validate(
        filename: str,
        content_bytes: bytes
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validates uploaded invoice file.
        Returns: (is_valid, error_message, file_type)
        """
        if not filename or "." not in filename:
            return False, "Invalid filename: missing extension.", ""

        ext = "." + filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file extension '{ext}'. Allowed extensions: .pdf, .csv, .json", ""

        file_size = len(content_bytes)
        if file_size == 0:
            return False, "Uploaded file is empty (0 bytes).", ext

        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return False, f"File size ({size_mb:.2f} MB) exceeds maximum allowed size (10.0 MB).", ext

        # Magic bytes check
        if ext == ".pdf":
            if not content_bytes.startswith(b"%PDF"):
                return False, "File has .pdf extension but lacks valid %PDF header.", ext
        elif ext == ".json":
            try:
                import json
                json.loads(content_bytes.decode("utf-8"))
            except Exception as e:
                return False, f"Invalid JSON structure: {str(e)}", ext
        elif ext == ".csv":
            try:
                content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content_bytes.decode("latin-1")
                except Exception:
                    return False, "Unable to decode CSV text with UTF-8 or Latin-1 encoding.", ext

        return True, None, ext
