"""
Abstract base class for screeners.
"""

import os
import pickle
import threading
import tempfile
import shutil
import asyncio
import logging
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
import nest_asyncio
nest_asyncio.apply()

class AbstractScreener(ABC):
    """Abstract base class for all screeners."""

    def __init__(self, llm, criteria_dict, vector_store_path, screening_type, verbose=False):

        """
        Initialize the screener.

        Args:
            llm (object): Language model instance
            criteria_dict (dict): Criteria for screening
            vector_store_path (str): Path to vector store
            screening_type (str): Type of screening ('screening1' or 'screening2')
            verbose (bool, optional): Verbose output. Defaults to False.
            selection_method (str, optional): Method for article selection.
                                              Options: 'all_criteria', 'clustering'.
                                              Defaults to 'all_criteria'.
        """
        self.llm = llm
        self.criteria_dict = criteria_dict
        self.vector_store_path = vector_store_path
        self.screening_type = screening_type
        self.verbose = verbose

        # Set up logger
        self.logger = self._setup_logger()

        # Set up directories
        self.screening_dir = f"{vector_store_path}/{screening_type}"
        os.makedirs(self.screening_dir, exist_ok=True)

        # Initialize storage
        self.record2answer = {}
        self.missing_records = set()
        self.results = None

        # Load existing data if available
        self._load_existing_data()

    def _setup_logger(self):
        """Set up logger for this screener."""
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

    def _load_existing_data(self):
        """Load existing screening data if available."""
        record2answer_path = f'{self.screening_dir}/{self.screening_type}_predicted_criteria.pkl'
        missing_records_path = f'{self.screening_dir}/{self.screening_type}_missing_records.pkl'

        if os.path.exists(record2answer_path):
            with open(record2answer_path, 'rb') as file:
                self.record2answer = pickle.load(file)

        if os.path.exists(missing_records_path):
            with open(missing_records_path, 'rb') as file:
                self.missing_records = pickle.load(file)

        self.logger.info(
            f"Loaded {len(self.record2answer)} existing records and "
            f"{len(self.missing_records)} missing records for {self.screening_type}"
        )

    @staticmethod
    def atomic_save(file_path, data):
        """
        Save data to file atomically to prevent corruption.

        Args:
            file_path (str): Path to save file
            data (any): Data to save
        """
        dir_name = os.path.dirname(file_path)
        with tempfile.NamedTemporaryFile('wb', delete=False, dir=dir_name) as tmp_file:
            pickle.dump(data, tmp_file)
            temp_name = tmp_file.name
        shutil.move(temp_name, file_path)

    def save_screening_results(self):
        """Save screening results to disk."""
        with threading.Lock():
            self.atomic_save(
                f'{self.screening_dir}/{self.screening_type}_predicted_criteria.pkl',
                self.record2answer
            )
            self.atomic_save(
                f'{self.screening_dir}/{self.screening_type}_missing_records.pkl',
                self.missing_records
            )

    def validate_mutual_exclusivity(self):
        """
        Validate that no record exists in both record2answer and missing_records.
        """
        overlapping_records = set(self.record2answer.keys()).intersection(self.missing_records)
        if overlapping_records:
            self.logger.error(f"Mutual exclusivity violated! Overlapping records: {overlapping_records}")
            # Resolve by removing from missing_records
            self.missing_records -= overlapping_records
            self.logger.info(f"Removed overlapping records from missing_records")
            # Save corrected data
            self.save_screening_results()
        else:
            self.logger.info("Mutual exclusivity validated successfully.")

    def parse_json_safely(self, json_string):
        """
        Parse JSON response safely with error handling.

        Args:
            json_string (str): JSON string to parse

        Returns:
            dict: Parsed JSON object or empty dict on failure
        """
        import re
        from json import JSONDecodeError

        # Use regex to extract the JSON object
        json_matches = re.findall(r'\{.*\}', json_string, re.DOTALL)
        if json_matches:
            json_data = json_matches[0]
            try:
                return json.loads(json_data)
            except JSONDecodeError:
                # Attempt to fix the JSON
                fixed_json_data = self._fix_json(json_data)
                try:
                    return json.loads(fixed_json_data)
                except JSONDecodeError as e:
                    self.logger.error(
                        f"Failed to parse JSON after attempting to fix: {e}\n"
                        f"Original Response was: {json_string}"
                    )
                    return {}
        else:
            self.logger.error(f"No JSON found in the response: {json_string}")
            return {}

    def _fix_json(self, json_string):
        """
        Attempt to fix malformed JSON.

        Args:
            json_string (str): Malformed JSON string

        Returns:
            str: Fixed JSON string
        """
        # Add missing closing braces
        open_braces = json_string.count('{')
        close_braces = json_string.count('}')
        json_string += '}' * (open_braces - close_braces)

        # Close unclosed quotes
        open_quotes = json_string.count('"') % 2
        if open_quotes:
            json_string += '"'

        return json_string

    @abstractmethod
    def prepare_prompt(self):
        """Prepare the screening prompt."""
        pass

    @abstractmethod
    async def process_record(self, record):
        """
        Process a single record.

        Args:
            record: Record to process

        Returns:
            tuple: (record_id, result)
            record = records_to_process[0]
        """

        pass

    # In academate/base/screener.py
    async def process_records_concurrently(self, records_to_process, max_concurrency=5):
        """
        Process records concurrently with improved error handling for event loop issues.

        Args:
            records_to_process (list): Records to process
            max_concurrency (int, optional): Maximum number of concurrent tasks. Defaults to 5.

        Returns:
            list: Results
        """
        # Initialize counters
        success_count = 0
        failure_count = 0

        # Create semaphore for concurrency control
        sem = asyncio.Semaphore(max_concurrency)

        # Wrap the process_record method with semaphore and better error handling
        async def process_with_semaphore(record):
            try:
                async with sem:
                    try:
                        # Set a timeout to prevent hanging operations
                        return await asyncio.wait_for(self.process_record(record), timeout=60)
                    except asyncio.TimeoutError:
                        self.logger.error(f"Timeout processing record {record.get('uniqueid', 'unknown')}")
                        return record.get('uniqueid', 'unknown'), None
                    except Exception as e:
                        self.logger.error(f"Error processing record {record.get('uniqueid', 'unknown')}: {str(e)}")
                        return record.get('uniqueid', 'unknown'), None
            except Exception as e:
                self.logger.error(f"Critical error with semaphore: {str(e)}")
                return record.get('uniqueid', 'unknown'), None

        # Create tasks one by one to avoid overloading
        all_results = []

        # Process in small batches to avoid overwhelming the event loop
        batch_size = 10
        for i in range(0, len(records_to_process), batch_size):
            batch = records_to_process[i:i + batch_size]
            self.logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(records_to_process) + batch_size - 1) // batch_size}")

            # Create tasks for this batch
            tasks = [asyncio.create_task(process_with_semaphore(record)) for record in batch]

            # Use tqdm if available, otherwise just process
            try:
                from tqdm.asyncio import tqdm_asyncio
                results = await tqdm_asyncio.gather(*tasks, desc=f"Processing Records ({self.screening_type})")
            except ImportError:
                results = await asyncio.gather(*tasks)

            # Process results from this batch
            for uniqueid, result in results:
                all_results.append((uniqueid, result))

                if result:
                    self.record2answer[uniqueid] = result
                    self.missing_records.discard(uniqueid)
                    success_count += 1
                else:
                    self.missing_records.add(uniqueid)
                    failure_count += 1

            # Explicitly clean up tasks
            for task in tasks:
                task.cancel()

            # Save results after each batch
            self.save_screening_results()

            # Force garbage collection to clean up resources
            import gc
            gc.collect()

            # Small delay between batches to allow event loop to process other tasks
            await asyncio.sleep(1)

        # Clean up GRPC resources that might be causing the errors
        self._cleanup_grpc_resources()

        return all_results

    def _cleanup_grpc_resources(self):
        """Clean up GRPC resources to prevent 'InterceptedCall' errors."""
        try:
            # Try to clean up GRPC resources
            import gc
            gc.collect()

            # Try to clean up specific GRPC channels if available
            try:
                import grpc.aio
                # This is a hacky solution but might help with the specific error
                import sys
                for obj in gc.get_objects():
                    if 'InterceptedCall' in str(type(obj)):
                        # Set the problematic attribute to avoid error in __del__
                        if not hasattr(obj, '_interceptors_task'):
                            setattr(obj, '_interceptors_task', None)
            except:
                pass
        except Exception as e:
            self.logger.warning(f"Error during GRPC cleanup: {str(e)}")

    @abstractmethod
    def screen(self, data):
        """
        Run the screening process with better event loop handling.

        Args:
            data: Data to screen

        Returns:
            pd.DataFrame: Screening results
        """
        # Prepare the prompt
        self.prompt = self.prepare_prompt()

        # Get records to process based on the specific screener implementation
        records_to_process = self._get_records_to_process(data, False)

        # Exit early if no records to process
        if not records_to_process:
            self.logger.info("All records have been processed.")
            return self._prepare_results(data)

        # Process records with proper event loop management
        try:
            # Create a new event loop to avoid nested loop issues
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the processing within this loop
            results = loop.run_until_complete(
                self.process_records_concurrently(records_to_process)
            )

            # Clean up the event loop
            loop.close()
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                self.logger.error("Event loop was closed unexpectedly. Creating a new one.")

                # Create a new event loop and try again
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Try again with a new loop
                results = loop.run_until_complete(
                    self.process_records_concurrently(records_to_process)
                )
                loop.close()
            else:
                self.logger.error(f"Runtime error in screening: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                results = []
        except Exception as e:
            self.logger.error(f"Error in screening: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            results = []

        # Save final results
        self.save_screening_results()

        # Check for mutual exclusivity
        self.validate_mutual_exclusivity()

        # Force garbage collection to clean up any remaining resources
        import gc
        gc.collect()

        # Prepare and return results
        return self._prepare_results(data)

    def _get_records_to_process(self, data, force_rescreen=False):
        """
        Determine which records need processing.
        Should be overridden by subclasses with specific implementation.

        Args:
            data: Data containing records
            force_rescreen (bool): Whether to force rescreening of already processed records

        Returns:
            list: Records that need processing
        """
        # Default implementation
        raise NotImplementedError("Subclasses must implement _get_records_to_process")