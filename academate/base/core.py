"""
Core base class for all Academate variants.
"""

import os
import logging
import hashlib
import re
import pandas as pd
from abc import ABC, abstractmethod


class AcademateBase(ABC):
    """Base class for all Academate variants."""

    def __init__(self, topic, llm, embeddings, criteria_dict, vector_store_path,
                 literature_df=None, content_column="record", embeddings_path=None,
                 pdf_location=None, verbose=False, chunksize=25):
        """
        Initialize the base Academate class.

        Args:
            topic (str): Topic of the systematic review
            llm (object): Large language model instance
            embeddings (object): Embeddings model instance
            criteria_dict (dict): Dictionary of inclusion/exclusion criteria
            vector_store_path (str): Path to store vector embeddings
            literature_df (pd.DataFrame, optional): DataFrame with literature
            content_column (str, optional): Column name for content. Defaults to "record".
            embeddings_path (str, optional): Path for embeddings. Defaults to None.
            pdf_location (str, optional): Path for PDF files. Defaults to None.
            verbose (bool, optional): Verbose output. Defaults to False.
            chunksize (int, optional): Chunk size for processing. Defaults to 25.
        """
        self.topic = topic
        self.llm = llm
        self.embeddings = embeddings
        self.criteria_dict = criteria_dict
        self.vector_store_path = vector_store_path
        self.content_column = content_column
        self.verbose = verbose
        self.chunksize = chunksize

        # Set embeddings path
        self.embeddings_path = embeddings_path or f"{self.vector_store_path}/embeddings"
        # Create screening-specific paths
        self.embeddings_path1 = f"{self.embeddings_path}/screening1_embeddings"
        self.embeddings_path2 = f"{self.embeddings_path}/screening2_embeddings"

        # Set PDF location
        self.pdf_location = pdf_location or f"{self.vector_store_path}/pdfs"

        # Set up logger
        self.logger = self._setup_logger()

        # Create directories
        self._create_directories()

        # Process literature dataframe if provided
        if literature_df is not None:
            self.literature_df = self._preprocess_literature_df(literature_df)
        else:
            self.literature_df = None

        # Initialize results storage
        self.results_screening1 = None
        self.results_screening2 = None

    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger(f"{self.__class__.__name__}")

        # Set logging level based on verbose flag
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Only add handlers if they are not already added
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False  # Prevent duplicate logs

        # Suppress DEBUG messages from pdfminer and others if not verbose
        if not self.verbose:
            logging.getLogger('pdfminer').setLevel(logging.WARNING)
            logging.getLogger('httpcore').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('chromadb').setLevel(logging.INFO)
            logging.getLogger('chromadb.telemetry').setLevel(logging.ERROR)

        return logger

    def _create_directories(self):
        """Create necessary directories for storage."""
        # Create main directories
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
        os.makedirs(self.pdf_location, exist_ok=True)

        # Create screening-specific directories
        self.screening1_dir = f"{self.vector_store_path}/screening1"
        self.embeddings_path1 = f"{self.embeddings_path}/screening1_embeddings"
        os.makedirs(self.screening1_dir, exist_ok=True)
        os.makedirs(self.embeddings_path1, exist_ok=True)

        self.screening2_dir = f"{self.vector_store_path}/screening2"
        self.embeddings_path2 = f"{self.embeddings_path}/screening2_embeddings"
        os.makedirs(self.screening2_dir, exist_ok=True)
        os.makedirs(self.embeddings_path2, exist_ok=True)

        # Set directory permissions
        try:
            for directory in [self.vector_store_path, self.embeddings_path, self.pdf_location,
                              self.screening1_dir, self.embeddings_path1,
                              self.screening2_dir, self.embeddings_path2]:
                os.chmod(directory, 0o777)  # Grant all permissions
        except Exception as e:
            self.logger.warning(f"Could not set permissions: {e}")

    def _preprocess_literature_df(self, df):
        """
        Preprocess the literature DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Make a copy and reset index
        df = df.copy()
        df.reset_index(drop=True, inplace=True)

        # Add uniqueid column
        df['uniqueid'] = df.apply(self.generate_uniqueid, axis=1)

        # Check and remove duplicates
        self._check_and_remove_duplicates(df)

        return df

    def generate_uniqueid(self, row):
        """
        Generate a unique ID for a row based on its content.

        Args:
            row (pd.Series): DataFrame row

        Returns:
            str: Unique ID
        """
        # Normalize while preserving some punctuation
        normalized_description = re.sub(r'[^a-zA-Z0-9 \-():]', '', row[self.content_column])

        # Normalize whitespace
        normalized_description = ' '.join(normalized_description.split())

        # Create a hash
        key_string = f"{normalized_description}"
        id_record = hashlib.sha256(key_string.encode()).hexdigest()[:20]

        return id_record

    def _check_and_remove_duplicates(self, df):
        """
        Check for duplicate unique IDs and remove them.

        Args:
            df (pd.DataFrame): DataFrame to check
        """
        num_duplicates = len(df) - len(set(df['uniqueid']))

        if num_duplicates > 0:
            self.logger.warning(f"Detected {num_duplicates} duplicate unique IDs in the DataFrame.")

            # Find and log duplicate rows
            duplicate_rows = df[df.duplicated('uniqueid', keep=False)]
            duplicate_rows = duplicate_rows.sort_values(by='uniqueid')

            if self.verbose:
                self.logger.warning("Duplicate rows based on uniqueid:")
                for index, row in duplicate_rows.iterrows():
                    self.logger.warning(f"  uniqueid: {row['uniqueid']}, Record: {row[self.content_column]}")

            # Remove duplicates
            df.drop_duplicates(subset='uniqueid', keep='first', inplace=True)
            self.logger.info(f"Removed duplicates. {len(df)} rows remaining.")
        else:
            self.logger.info("No duplicate unique IDs found.")

    @abstractmethod
    def run_screening1(self):
        """Run the title/abstract screening process."""
        pass

    @abstractmethod
    def run_screening2(self):
        """Run the full-text screening process."""
        pass