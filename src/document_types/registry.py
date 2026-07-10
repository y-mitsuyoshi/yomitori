"""Registry for document-type plugins with auto-detection."""

from src.document_types.base import DocumentType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentTypeRegistry:
    """Register document types and auto-detect from images."""

    def __init__(self) -> None:
        self._types: list[DocumentType] = []

    def register(self, doc_type: DocumentType) -> None:
        """Register a document type.

        Args:
            doc_type: DocumentType instance.
        """
        self._types.append(doc_type)
        logger.debug("Registered document type: %s", doc_type.document_type_id)

    def detect(self, image: "np.ndarray") -> DocumentType:
        """Auto-detect the document type from an image.

        Args:
            image: Input image (BGR ndarray).

        Returns:
            Best-matching DocumentType.

        Raises:
            RuntimeError: If no document types are registered.
        """
        import numpy as np  # noqa: F811 — local import keeps base.py decoupled

        if not self._types:
            raise RuntimeError("No document types registered")
        candidates = [(dt, dt.detect(image)) for dt in self._types]
        candidates.sort(key=lambda x: x[1], reverse=True)
        best, score = candidates[0]
        logger.info(
            "Auto-detected document type: %s (score=%.3f)",
            best.document_type_id,
            score,
        )
        return best

    def get_by_id(self, type_id: str) -> DocumentType:
        """Retrieve a document type by ID.

        Args:
            type_id: Document type ID string.

        Returns:
            The matching DocumentType.

        Raises:
            ValueError: If the ID is unknown.
        """
        for dt in self._types:
            if dt.document_type_id == type_id:
                return dt
        raise ValueError(f"Unknown document type: {type_id}")

    def list_types(self) -> list[str]:
        """List all registered document type IDs.

        Returns:
            List of type ID strings.
        """
        return [dt.document_type_id for dt in self._types]

    def __len__(self) -> int:
        return len(self._types)