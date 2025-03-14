"""
Literature embedder implementation.
"""

import os
import time
from tqdm import tqdm
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
from academate.base.embedder import AbstractEmbedder
import threading


class RateLimiter:
    """Rate limiter for API calls using token bucket algorithm."""

    def __init__(self, rate=150, per=60):
        """
        Initialize the rate limiter.

        Args:
            rate (int): Number of tokens (requests) allowed per time period
            per (int): Time period in seconds
        """
        self.rate = rate  # Tokens per period
        self.per = per  # Period in seconds
        self.tokens = rate  # Start with a full bucket
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """
        Acquire a token, waiting if necessary.

        Returns:
            float: Time to wait before making the request
        """
        with self.lock:
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * (self.rate / self.per)
            self.tokens = min(self.rate, self.tokens + tokens_to_add)
            self.last_refill = now

            # If no tokens available, calculate wait time
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                self.tokens = 0
                return wait_time

            # Consume a token
            self.tokens -= 1
            return 0


class LiteratureEmbedder(AbstractEmbedder):
    """Embedder for literature (titles and abstracts)."""

    def __init__(self, embeddings, vector_store_path, embedding_type, embeddings_path=None, verbose=False):
        """
        Initialize the literature embedder.

        Args:
            embeddings (object): Embeddings model
            vector_store_path (str): Path to vector store
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        super().__init__(
            embeddings=embeddings,
            vector_store_path=vector_store_path,
            embedding_type=embedding_type,
            verbose=verbose,
            embeddings_path=embeddings_path
        )

        self.db = None
        # Initialize rate limiter (120 requests per minute, leaving buffer)
        self.rate_limiter = RateLimiter(rate=120, per=60)
        self.embeddings_path = embeddings_path or f"{vector_store_path}/embeddings"

    def embed(self, df, content_column="Record"):
        """
        Embed the literature DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with literature
            content_column (str, optional): Column name for content. Defaults to "Record".

        Returns:
            object: Chroma DB instance
        """
        self.logger.info(f"Embedding {len(df)} records")

        # Create DataFrameLoader
        loader = DataFrameLoader(df, page_content_column=content_column)

        # Load all documents
        all_docs = loader.load()

        # Ensure 'uniqueid' is in metadata and is a string
        for doc in all_docs:
            if 'uniqueid' in doc.metadata:
                doc.metadata['uniqueid'] = str(doc.metadata['uniqueid'])
            else:
                self.logger.warning(f"'uniqueid' not found in metadata for doc: {doc}")

        # Check if Chroma collection exists
        if self.embeddings_exist():
            self.logger.info(f"Loading existing Chroma index from {self.embeddings_path}")
            self.db = Chroma(
                collection_name="literature",
                persist_directory=self.embeddings_path,
                embedding_function=self.embeddings
            )
        else:
            self.logger.info(f"Creating new Chroma index at {self.embeddings_path}")
            self.db = Chroma(
                collection_name="literature",
                persist_directory=self.embeddings_path,
                embedding_function=self.embeddings
            )

            # Reduce batch size to avoid rate limits
            batch_size = 20  # Reduced from 100 to 20
            total_batches = (len(all_docs) + batch_size - 1) // batch_size

            with tqdm(total=total_batches, desc="Embedding documents") as progress_bar:
                for i in range(0, len(all_docs), batch_size):
                    # Check rate limit before processing batch
                    wait_time = self.rate_limiter.acquire()
                    if wait_time > 0:
                        self.logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds before next batch")
                        time.sleep(wait_time)

                    try:
                        batch_docs = all_docs[i:i + batch_size]
                        texts = [doc.page_content for doc in batch_docs]
                        metadatas = [doc.metadata for doc in batch_docs]

                        # Ensure all metadata values are strings
                        metadatas = [{k: str(v) for k, v in metadata.items()} for metadata in metadatas]

                        self.db.add_texts(texts, metadatas=metadatas)
                    except Exception as e:
                        if "RATE_LIMIT_EXCEEDED" in str(e):
                            self.logger.warning(
                                f"Rate limit exceeded. Waiting 60 seconds before retrying batch {i // batch_size + 1}/{total_batches}")
                            time.sleep(60)  # Wait a full minute on rate limit errors
                            # Try again with a smaller batch
                            smaller_batch = batch_docs[:len(batch_docs) // 2]
                            smaller_texts = [doc.page_content for doc in smaller_batch]
                            smaller_metadatas = [doc.metadata for doc in smaller_batch]
                            smaller_metadatas = [{k: str(v) for k, v in metadata.items()} for metadata in
                                                 smaller_metadatas]

                            self.db.add_texts(smaller_texts, metadatas=smaller_metadatas)

                            # Process the second half after another delay
                            time.sleep(5)
                            smaller_batch = batch_docs[len(batch_docs) // 2:]
                            smaller_texts = [doc.page_content for doc in smaller_batch]
                            smaller_metadatas = [doc.metadata for doc in smaller_batch]
                            smaller_metadatas = [{k: str(v) for k, v in metadata.items()} for metadata in
                                                 smaller_metadatas]

                            self.db.add_texts(smaller_texts, metadatas=smaller_metadatas)
                        else:
                            self.logger.error(f"Error embedding batch {i // batch_size + 1}/{total_batches}: {str(e)}")
                            # Try to continue with next batch
                            continue

                    progress_bar.update(1)
                    # Add a small delay between batches for safety
                    time.sleep(0.5)

        # Validate the embeddings
        self.validate_embeddings(df)

        return self.db

    def validate_embeddings(self, df):
        """
        Validate that all records are properly embedded.

        Args:
            df (pd.DataFrame): Original DataFrame

        Returns:
            bool: True if valid, False otherwise
        """
        # Get all uniqueids from the Chroma DB
        indexed_uniqueids = self.get_indexed_uniqueids()

        # Get all uniqueids from the DataFrame
        df_uniqueids = set(df['uniqueid'].astype(str))

        # Find missing and extra uniqueids
        missing_uniqueids = df_uniqueids - indexed_uniqueids
        extra_uniqueids = indexed_uniqueids - df_uniqueids

        if missing_uniqueids:
            self.logger.warning(f"Missing {len(missing_uniqueids)} documents in the index")
            if self.verbose:
                self.logger.debug(f"Missing uniqueids: {missing_uniqueids}")
            return False
        else:
            self.logger.info("All documents are correctly embedded in the index")

        if extra_uniqueids:
            self.logger.warning(
                f"There are {len(extra_uniqueids)} extra documents in the index not present in the dataframe")
            if self.verbose:
                self.logger.debug(f"Extra uniqueids: {extra_uniqueids}")

        return len(missing_uniqueids) == 0

    def get_indexed_uniqueids(self):
        """
        Get all uniqueids currently in the index.

        Returns:
            set: Set of uniqueids
        """
        uniqueids = set()

        if self.db is None:
            return uniqueids

        # Get all documents from the collection, including metadatas
        results = self.db._collection.get(include=['metadatas'])

        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if metadata and 'uniqueid' in metadata:
                    uniqueids.add(metadata['uniqueid'])

        return uniqueids