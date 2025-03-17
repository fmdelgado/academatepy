"""
Abstract base class for embedders.
"""

import os
import logging
from abc import ABC, abstractmethod


class AbstractEmbedder(ABC):
    """Abstract base class for all embedders."""

    def __init__(self, embeddings, vector_store_path, embedding_type,
                 embeddings_path=None, verbose=False):
        """Initialize the embedder."""
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.embedding_type = embedding_type
        self.verbose = verbose

        # Set up logger
        self.logger = self._setup_logger()

        # Set up embedding paths
        if embeddings_path is None:
            self.embeddings_base_path = f"{vector_store_path}/embeddings"
        else:
            self.embeddings_base_path = embeddings_path

        self.embeddings_path = f"{self.embeddings_base_path}/{embedding_type}_embeddings"
        os.makedirs(self.embeddings_path, exist_ok=True)

    def _setup_logger(self):
        """Set up logger for this embedder."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        level = logging.DEBUG if self.verbose else logging.INFO
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger

    def embeddings_exist(self):
        """
        Check if embeddings already exist.

        Returns:
            bool: True if embeddings exist, False otherwise
        """
        if os.path.exists(os.path.join(self.embeddings_path, 'chroma.sqlite3')) or \
                os.path.exists(os.path.join(self.embeddings_path, 'index')):
            self.logger.info(f"Existing {self.embedding_type} embeddings found at {self.embeddings_path}")
            return True
        return False

    @abstractmethod
    def embed(self, data):
        """
        Embed the provided data.

        Args:
            data: Data to embed

        Returns:
            any: Embedding results
        """
        pass

    @abstractmethod
    def validate_embeddings(self, data):
        """
        Validate that embeddings are complete and correct.

        Args:
            data: Original data to validate against

        Returns:
            bool: True if valid, False otherwise
        """
        pass