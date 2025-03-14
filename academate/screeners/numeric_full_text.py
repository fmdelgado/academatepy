"""
Full-text screener implementation.
"""

from langchain.prompts import ChatPromptTemplate
from academate.screeners.full_text import FullTextScreener
import json


class NumericFullTextScreener(FullTextScreener):
    """Screener for full-text documents with numeric ratings."""

    class NumericFullTextScreener(FullTextScreener):
        """Screener for full-text documents with numeric ratings."""

        def __init__(self, llm, embeddings, criteria_dict, vector_store_path, verbose=False,
                     selection_method="threshold"):
            """
            Initialize the numeric full-text screener.
            """
            super().__init__(
                llm=llm,
                embeddings=embeddings,
                criteria_dict=criteria_dict,
                vector_store_path=vector_store_path,
                screening_type="screening2",
                verbose=verbose
            )
            self.selection_method = selection_method
            self.threshold = 0.7  # Default threshold value

    def prepare_prompt(self):
        """
        Prepare a prompt template that asks for numeric ratings.

        Returns:
            object: Prompt template
        """
        # Format criteria for the prompt
        formatted_criteria = "\n".join(f"- {key}: {value}"
                                       for key, value in self.criteria_dict.items())

        # Create JSON structure for the expected output
        json_structure = json.dumps(
            {key: {"rating": "float between 0 and 1", "reason": "string"} for key in self.criteria_dict.keys()},
            indent=2
        )

        # Escape curly braces for string formatting
        escaped_json_structure = json_structure.replace("{", "{{").replace("}", "}}")

        # Create the prompt template with rating scale guidance
        prompt_template = f"""
                Analyze the following sections and evaluate how well it meets each criterion on a scale from 0 to 1.
                Only use the information provided in the context.

                Context: {{context}}

                Criteria:
                {formatted_criteria}

                Rating Scale Guidelines:
                - 0.0: The criterion is definitely not met
                - 0.25: The criterion is mostly not met, but there are minor elements that align
                - 0.5: The criterion is partially met (equal evidence for and against)
                - 0.75: The criterion is mostly met, with minor reservations
                - 1.0: The criterion is definitely met

                You may use any value between 0 and 1 (e.g., 0.3, 0.8) to provide more precise ratings.
                For each criterion, provide a numeric rating and a brief reason for your assessment.

                **Important**: Respond **only** with a JSON object matching the following structure:

                {escaped_json_structure}

                Ensure your response is a valid JSON object with ratings as floating-point numbers between 0 and 1.
                """

        # Store formatted criteria and JSON structure for later use
        self.formatted_criteria = formatted_criteria
        self.json_structure = json_structure

        return ChatPromptTemplate.from_template(prompt_template)

    # Reuse the _prepare_results method from NumericTitleAbstractScreener with minor adjustments
    # In academate/screeners/numeric_full_text.py

    def _prepare_results(self, data):
        """
        Prepare results DataFrame with numeric ratings.

        Args:
            data (pd.DataFrame or list): Original data

        Returns:
            pd.DataFrame: Results DataFrame
        """
        from academate.utils.clustering import identify_teams_continuous, analyze_continuous_teams
        import numpy as np
        # Convert record2answer to DataFrame
        results_rows = []

        for uniqueid, answers in self.record2answer.items():
            row = {'uniqueid': uniqueid}

            # Extract ratings for each criterion
            for criterion, response in answers.items():
                if isinstance(response, dict):
                    # Get rating (default to 0.0 if not found)
                    rating = response.get('rating', 0.0)

                    # Convert string ratings to float
                    if isinstance(rating, str):
                        try:
                            rating = float(rating)
                        except ValueError:
                            rating = 0.0

                    # Ensure rating is between 0 and 1
                    rating = max(0.0, min(1.0, rating))

                    row[criterion] = rating
                else:
                    row[criterion] = 0.0

            # Add row to results
            results_rows.append(row)

        # Create DataFrame
        import pandas as pd
        results_df = pd.DataFrame(results_rows)

        # Add predicted_screening column based on threshold (e.g., average rating ≥ 0.7)
        if not results_df.empty:
            criteria_columns = list(self.criteria_dict.keys())

            if self.selection_method == "threshold":
                # Default method: average rating ≥ threshold
                results_df['predicted_screening2'] = results_df[criteria_columns].mean(axis=1) >= self.threshold

            elif self.selection_method == "clustering":
                if len(results_df) > 1:  # Need at least 2 records for clustering
                    # Create rating matrix
                    rating_matrix = results_df[criteria_columns].values

                    # Apply continuous clustering
                    team_labels, selected_features = identify_teams_continuous(rating_matrix)

                    # Analyze clustering results
                    analysis = analyze_continuous_teams(rating_matrix, team_labels, criteria_columns)

                    # Log analysis results
                    self.logger.info(f"Continuous Clustering Analysis:")
                    self.logger.info(f"Top discriminative features: {[f[0] for f in analysis['top_features'][:3]]}")
                    self.logger.info(
                        f"Team sizes: {analysis['winning_team_size']} winning, {analysis['losing_team_size']} losing")

                    # Assign team labels
                    results_df['predicted_screening2'] = team_labels
                else:
                    # Fall back to threshold method if not enough records
                    self.logger.warning("Not enough records for clustering. Using threshold instead.")
                    results_df['predicted_screening2'] = results_df[criteria_columns].mean(axis=1) >= self.threshold

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