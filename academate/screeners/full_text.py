"""
Full-text screener implementation.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import pandas as pd
import json
import asyncio
import os
from academate.base.screener import AbstractScreener
import pickle

class FullTextScreener(AbstractScreener):
    """Screener for full-text documents."""

    def __init__(self, llm, embeddings, criteria_dict, vector_store_path, embeddings_path=None, verbose=False,
                 selection_method="all_criteria"):
        """
        Initialize the full-text screener.

        Args:
            llm (object): Language model instance
            embeddings (object): Embeddings model instance
            criteria_dict (dict): Screening criteria
            vector_store_path (str): Path to vector store
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        super().__init__(
            llm=llm,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            screening_type="screening2",
            verbose=verbose
        )

        self.embeddings = embeddings
        self.embeddings_path = embeddings_path
        self.selection_method = selection_method

        # Initialize error tracking
        self.pdf_embedding_error = set()
        self._load_error_tracking()
    def _load_error_tracking(self):
        """Load error tracking if available."""
        error_path = f'{self.embeddings_path}/screening2_PDFembedding_error.pkl'
        if os.path.exists(error_path):
            with open(error_path, 'rb') as file:
                self.pdf_embedding_error = pickle.load(file)
            self.logger.info(f"Loaded {len(self.pdf_embedding_error)} PDF embedding errors.")

        # Add this section to sync missing_records with the database
        if os.path.exists(f'{self.screening_dir}/{self.screening_type}_missing_records.pkl'):
            with open(f'{self.screening_dir}/{self.screening_type}_missing_records.pkl', 'rb') as file:
                self.missing_records = pickle.load(file)
            self.logger.info(f"Loaded {len(self.missing_records)} missing records for {self.screening_type}.")

    def prepare_prompt(self):
        """
        Prepare the screening prompt template.

        Returns:
            object: Prompt template
        """
        # Format criteria for the prompt
        formatted_criteria = "\n".join(f"- {key}: {value}"
                                       for key, value in self.criteria_dict.items())

        # Create JSON structure for the expected output
        json_structure = json.dumps(
            {key: {"label": "boolean", "reason": "string"} for key in self.criteria_dict.keys()},
            indent=2
        )

        # Escape curly braces for string formatting
        escaped_json_structure = json_structure.replace("{", "{{").replace("}", "}}")

        # Create the prompt template
        prompt_template = f"""
        Analyze the following sections extracted from a scientific article and determine if it meets the specified criteria.
        Only use the information provided in the context.

        Context: \n\n {{context}}\n\n

        Criteria:\n\n 
        {formatted_criteria}\n\n 

        For each criterion, provide a boolean label (true if it meets the criterion, false if it doesn't)
        and a brief reason for your decision.

        **Important**: Respond **only** with a JSON object matching the following structure, and do not include any additional text:

        {escaped_json_structure}

        Ensure your response is a valid JSON object.
        """

        # Store formatted criteria and JSON structure for later use
        self.formatted_criteria = formatted_criteria
        self.json_structure = json_structure

        return ChatPromptTemplate.from_template(prompt_template)

    async def validate_chroma_db(self, path_name, collection_name):
        """
        Validate that a Chroma database exists and is accessible.

        Args:
            path_name (str): Path to database
            collection_name (str): Collection name

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not (os.path.exists(os.path.join(path_name, 'chroma.sqlite3')) or
                    os.path.exists(os.path.join(path_name, 'index'))):
                return False

            # Try to open the database
            db = Chroma(
                collection_name=collection_name,
                persist_directory=path_name,
                embedding_function=self.embeddings
            )

            # Verify we can access the collection
            _ = db._collection.count()
            return True

        except Exception as e:
            self.logger.warning(f"Invalid Chroma database at {path_name}: {str(e)}")
            return False

    async def process_record(self, record):
        """
        Process a single record for full-text screening.

        Args:
            record (dict): Record to process

        Returns:
            tuple: (record_id, result)
            record = records_to_process[0]
            uniqueid = '3b85c58d9e0f478abc85'
        """
        max_retries = 5

        # Get the uniqueid
        uniqueid = record.get('uniqueid')

        # Ensure we have a valid uniqueid
        if not uniqueid:
            self.logger.error("Record missing uniqueid")
            return None, None

        # Try multiple times in case of failure
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Using embeddings path: {self.embeddings_path}")

                # Ensure we're using the correct embeddings path
                if not await self.validate_chroma_db(self.embeddings_path, "screening2"):
                    self.logger.error(f"No valid Chroma index found at {self.embeddings_path}")
                    return uniqueid, None

                # Create Chroma instance
                pdf_db = Chroma(
                    collection_name="screening2",
                    persist_directory=self.embeddings_path,
                    embedding_function=self.embeddings
                )

                # Retrieve relevant chunks for each criterion
                all_retrieved_docs = []
                for key, value in self.criteria_dict.items():
                    retriever = pdf_db.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            'k': 5,
                            'score_threshold': 0.0,
                            'filter': {'source': uniqueid}
                        }
                    )

                    retrieved_docs = retriever.invoke(value)

                    if not retrieved_docs:
                        self.logger.warning(f"No documents retrieved for criterion '{key}' in record {uniqueid}")
                        continue

                    all_retrieved_docs.extend(retrieved_docs)

                # If no documents were retrieved, we can't process this record
                if not all_retrieved_docs:
                    self.logger.error(f"No documents retrieved for any criteria in record {uniqueid}")
                    return uniqueid, None

                # Remove duplicate documents based on content
                unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()

                # Combine chunks into a single context
                context = "\n\n".join([doc.page_content for doc in unique_docs])

                # Prepare inputs
                inputs = {
                    "context": context
                }

                # Invoke the model
                # Create messages from the prompt template
                messages = self.prompt.format_messages(**inputs)
                # print(messages[0].content)

                # Invoke the LLM with messages
                result = await self.llm.ainvoke(messages)
                # result = self.llm.invoke(messages)
                # print(result.content)

                # Parse the result
                parsed_result = self.parse_json_safely(result.content if hasattr(result, 'content') else result)

                if parsed_result:
                    return uniqueid, parsed_result
                else:
                    self.logger.warning(f"Empty or invalid response for {uniqueid}. Retrying...")

            except Exception as e:
                # Check if event loop is closed
                if "Event loop is closed" in str(e):
                    self.logger.error(f"Event loop closed for {uniqueid}. Cannot continue.")
                    return uniqueid, None

                self.logger.error(f"Error processing {uniqueid}: {str(e)}")

                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying {uniqueid}... (Attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)
                else:
                    self.logger.error(f"Failed to process {uniqueid} after {max_retries} attempts.")
                    break

        return uniqueid, None

    def screen(self, data):
        """
        Run the full-text screening process.

        Args:
            data (pd.DataFrame): DataFrame with literature data

        Returns:
            pd.DataFrame: DataFrame with screening results

        data = filtered_df
        """
        # Prepare the prompt
        self.prompt = self.prepare_prompt()

        # Filter out records with PDF embedding errors
        if isinstance(data, pd.DataFrame):
            filtered_data = data[~data['uniqueid'].isin(self.pdf_embedding_error)]
            self.logger.info(f"Filtered out {len(data) - len(filtered_data)} records with PDF embedding errors.")
        else:
            filtered_data = [r for r in data if r['uniqueid'] not in self.pdf_embedding_error]
            self.logger.info(f"Filtered out {len(data) - len(filtered_data)} records with PDF embedding errors.")

        # Determine which records to process
        if isinstance(filtered_data, pd.DataFrame):
            records_to_process = []
            for _, row in filtered_data.iterrows():
                uniqueid = row['uniqueid']
                if uniqueid not in self.record2answer:
                    records_to_process.append(row.to_dict())
        else:
            records_to_process = [r for r in filtered_data if r['uniqueid'] not in self.record2answer]

        # Add missing records to be processed again
        for uniqueid in self.missing_records:
            if isinstance(filtered_data, pd.DataFrame):
                row = filtered_data[filtered_data['uniqueid'] == uniqueid]
                if not row.empty:
                    records_to_process.append(row.iloc[0].to_dict())
            else:
                record = next((r for r in filtered_data if r['uniqueid'] == uniqueid), None)
                if record:
                    records_to_process.append(record)

        # Exit early if no records to process
        if not records_to_process:
            self.logger.info("All records have been processed.")
            return self._prepare_results(data)

        # Process records
        try:
            results = asyncio.run(self.process_records_concurrently(records_to_process))
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Try again
                results = loop.run_until_complete(
                    self.process_records_concurrently(records_to_process))
                loop.close()
            else:
                raise

        # Summarize results
        total_records = len(records_to_process)
        success_count = len([r for r in results if r[1]])
        failure_count = len([r for r in results if not r[1]])

        self.logger.info(f"\nTotal records processed: {total_records}")
        self.logger.info(f"Successfully processed: {success_count}")
        self.logger.info(f"Failed to process: {failure_count}")

        # Save final results
        self.save_screening_results()

        # Check for mutual exclusivity
        self.validate_mutual_exclusivity()

        # Prepare and return results
        return self._prepare_results(data)

    # In academate/screeners/numeric_full_text.py
    def _prepare_results(self, data):
        """
        Prepare the final results DataFrame.

        Args:
            data (pd.DataFrame or list): Original data

        Returns:
            pd.DataFrame: Results DataFrame
        """
        from academate.utils.clustering import identify_teams, analyze_teams

        # Convert record2answer to DataFrame
        results_rows = []

        for uniqueid, answers in self.record2answer.items():
            row = {'uniqueid': uniqueid}

            # Extract boolean labels for each criterion
            for criterion, response in answers.items():
                if isinstance(response, dict):
                    row[criterion] = response.get('label', False)
                    # Convert string 'true'/'false' to boolean
                    if isinstance(row[criterion], str):
                        row[criterion] = row[criterion].lower() == 'true'
                else:
                    row[criterion] = bool(response)

            # Add row to results
            results_rows.append(row)

        # Create DataFrame
        results_df = pd.DataFrame(results_rows)

        if not results_df.empty:
            # Ensure columns are boolean
            for column in self.criteria_dict.keys():
                if column in results_df.columns:
                    results_df[column] = results_df[column].astype(bool)

            # Determine article selection based on method
            if self.selection_method == 'all_criteria':
                # Default behavior: all criteria must be true
                results_df['predicted_screening2'] = results_df[list(self.criteria_dict.keys())].all(axis=1)

            elif self.selection_method == 'clustering':
                # Clustering approach
                if len(results_df) > 1:  # Need at least 2 records for clustering
                    # Create boolean matrix
                    boolean_matrix = results_df[list(self.criteria_dict.keys())].values

                    # Apply clustering
                    team_labels, selected_features = identify_teams(boolean_matrix)

                    # Analyze clustering results
                    feature_names = list(self.criteria_dict.keys())
                    analysis = analyze_teams(boolean_matrix, team_labels, feature_names)

                    # Log analysis results
                    self.logger.info(f"Clustering Analysis:")
                    self.logger.info(f"Top discriminative features: {[f[0] for f in analysis['top_features'][:3]]}")
                    self.logger.info(
                        f"Team sizes: {analysis['winning_team_size']} winning, {analysis['losing_team_size']} losing")

                    # Assign team labels
                    results_df['predicted_screening2'] = team_labels
                else:
                    # Fall back to default if not enough records
                    self.logger.warning("Not enough records for clustering. Using all criteria instead.")
                    results_df['predicted_screening2'] = results_df[list(self.criteria_dict.keys())].all(axis=1)

        # Merge with original data
        if isinstance(data, pd.DataFrame):
            merged_df = data.merge(results_df, on='uniqueid', how='left')
        else:
            # Create DataFrame from list of dicts
            data_df = pd.DataFrame(data)
            merged_df = data_df.merge(results_df, on='uniqueid', how='left')

        # Store results
        self.results = merged_df

        return merged_df


    def _emergency_fallback_results(self, data):
        """
        Emergency fallback method when normal processing fails.
        Simply marks all records as rejected.

        Args:
            data (pd.DataFrame): Original data

        Returns:
            pd.DataFrame: Data with predicted column set to False
        """
        self.logger.warning("Using emergency fallback processing")
        result_df = data.copy()
        result_df[f'predicted_{self.screening_type}'] = False

        # Create minimal criteria columns if needed
        for criterion in self.criteria_dict:
            if f"{criterion}" not in result_df.columns:
                if 'numeric' in self.__class__.__name__.lower():
                    result_df[criterion] = 0
                else:
                    result_df[criterion] = False

        return result_df