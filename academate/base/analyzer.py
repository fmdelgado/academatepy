"""
Abstract base class for analyzers.
"""

import os
import logging
from abc import ABC, abstractmethod


class AbstractAnalyzer(ABC):
    """Abstract base class for all analyzers."""

    def __init__(self, vector_store_path, analysis_type, verbose=False):
        """
        Initialize the analyzer.

        Args:
            vector_store_path (str): Path to vector store
            analysis_type (str): Type of analysis ('metrics', 'visualization')
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        self.vector_store_path = vector_store_path
        self.analysis_type = analysis_type
        self.verbose = verbose

        # Set up logger
        self.logger = self._setup_logger()

        # Set up analysis directory
        self.analysis_dir = f"{vector_store_path}/{analysis_type}"
        os.makedirs(self.analysis_dir, exist_ok=True)

    def _setup_logger(self):
        """Set up logger for this analyzer."""
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

    @abstractmethod
    def analyze(self, data):
        """
        Analyze the provided data.

        Args:
            data: Data to analyze

        Returns:
            any: Analysis results
        """
        pass

    @abstractmethod
    def save_results(self, results, filename):
        """
        Save analysis results.

        Args:
            results: Analysis results
            filename (str): Filename for saved results
        """
        pass