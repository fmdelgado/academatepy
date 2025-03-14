"""
Title and abstract screener implementation.
"""

from langchain.prompts import ChatPromptTemplate
import pandas as pd
import json
import asyncio
from academate.base.screener import AbstractScreener


class TitleAbstractScreener(AbstractScreener):
    """Screener for titles and abstracts."""

    def __init__(self, llm, criteria_dict, vector_store_path, verbose=False, selection_method="all_criteria"):
        """
        Initialize the title/abstract screener.

        Args:
            llm (object): Language model instance
            criteria_dict (dict): Screening criteria
            vector_store_path (str): Path to vector store
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        super().__init__(
            llm=llm,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            screening_type="screening1",
            verbose=verbose
        )

        # Store content column for later use
        self.content_column = "Record"  # Default column name
        self.selection_method = selection_method

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
        Analyze the following scientific article and determine if it meets the specified criteria.
        Only use the information provided in the context.

        Context: {{context}}

        Criteria:
        {formatted_criteria}

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

    async def process_record(self, record):
        """
        Process a single record for title/abstract screening.

        Args:
            record (dict or pd.Series): Record to process

        Returns:
            tuple: (record_id, result)
        """
        max_retries = 5

        # Get the uniqueid
        if isinstance(record, dict):
            uniqueid = record.get('uniqueid')
            context = record.get(self.content_column, '')
        else:  # Assume pandas Series
            uniqueid = record['uniqueid']
            context = record[self.content_column]

        # Ensure we have a valid uniqueid
        if not uniqueid:
            self.logger.error("Record missing uniqueid")
            return None, None

        # Try multiple times in case of failure
        for attempt in range(max_retries):
            try:
                # Prepare inputs
                inputs = {
                    "context": context
                }

                # Create messages from the prompt template - THIS IS THE KEY CHANGE
                messages = self.prompt.format_messages(**inputs)

                # Invoke the chain with messages
                result = await self.llm.ainvoke(messages)

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
        Run the title/abstract screening process.

        Args:
            data (pd.DataFrame): DataFrame with literature data

        Returns:
            pd.DataFrame: DataFrame with screening results
            data = self.literature_df
        """
        # Store the content column name
        if isinstance(data, pd.DataFrame):
            for col in ['Record', 'record', 'content', 'text']:
                if col in data.columns:
                    self.content_column = col
                    break

        # Prepare the prompt
        self.prompt = self.prepare_prompt()

        # Determine which records to process
        if isinstance(data, pd.DataFrame):
            records_to_process = []
            for _, row in data.iterrows():
                uniqueid = row['uniqueid']
                if uniqueid not in self.record2answer:
                    records_to_process.append(row.to_dict())
        else:
            records_to_process = [r for r in data if r['uniqueid'] not in self.record2answer]

        # Add missing records to be processed again
        for uniqueid in self.missing_records:
            if isinstance(data, pd.DataFrame):
                row = data[data['uniqueid'] == uniqueid]
                if not row.empty:
                    records_to_process.append(row.iloc[0].to_dict())
            else:
                record = next((r for r in data if r['uniqueid'] == uniqueid), None)
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

    def _prepare_results(self, data):
        """
        Prepare the final results DataFrame.

        Args:
            data (pd.DataFrame or list): Original data

        Returns:
            pd.DataFrame: Results DataFrame
        """
        # Convert record2answer to DataFrame
        from academate.utils.clustering import identify_teams, analyze_teams

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

        # Add predicted_screening1 column (True if all criteria are met)
        if not results_df.empty:
            for column in self.criteria_dict.keys():
                if column in results_df.columns:
                    results_df[column] = results_df[column].astype(bool)

            if self.selection_method == 'all_criteria':
                results_df['predicted_screening1'] = results_df[list(self.criteria_dict.keys())].all(axis=1)
                # Default behavior: all criteria must be true

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
                    results_df['predicted_screening1'] = team_labels
                else:
                    # Fall back to default if not enough records
                    self.logger.warning("Not enough records for clustering. Using all criteria instead.")
                    results_df['predicted_screening1'] = results_df[list(self.criteria_dict.keys())].all(axis=1)

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