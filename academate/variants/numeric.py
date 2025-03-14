# academate/variants/numeric.py
from academate.variants.single_db import AcademateSingleDB
from academate.screeners.numeric_title_abstract import NumericTitleAbstractScreener
from academate.screeners.numeric_full_text import NumericFullTextScreener


class NumericAcademate(AcademateSingleDB):
    """
    Academate variant that uses numeric ratings (0-1) instead of boolean values.

    This allows for more nuanced assessment of how well articles meet criteria.
    """

    def __init__(self, topic, llm, embeddings, criteria_dict, vector_store_path,
                 literature_df=None, content_column="record", embeddings_path=None,
                 pdf_location=None, verbose=False, chunksize=25, rating_threshold=0.7,
                 selection_method="threshold"):
        """
        Initialize NumericAcademate.

        Args:
            topic (str): Topic of the systematic review
            llm (object): Language model instance
            embeddings (object): Embeddings model instance
            criteria_dict (dict): Criteria for screening
            vector_store_path (str): Path to vector store
            literature_df (pd.DataFrame, optional): Literature DataFrame. Defaults to None.
            content_column (str, optional): Content column name. Defaults to "record".
            embeddings_path (str, optional): Path for embeddings. Defaults to None.
            pdf_location (str, optional): Path for PDFs. Defaults to None.
            verbose (bool, optional): Verbose output. Defaults to False.
            chunksize (int, optional): Chunk size. Defaults to 25.
            rating_threshold (float, optional): Threshold for criteria. Defaults to 0.7.
        """
        # Initialize base class
        super().__init__(
            topic=topic,
            llm=llm,
            embeddings=embeddings,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            literature_df=literature_df,
            content_column=content_column,
            embeddings_path=embeddings_path,
            pdf_location=pdf_location,
            verbose=verbose,
            chunksize=chunksize
        )

        # Store options
        self.rating_threshold = rating_threshold
        self.selection_method = selection_method

        # Override screeners with numeric versions
        self.title_abstract_screener = NumericTitleAbstractScreener(
            llm=llm,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            verbose=verbose,
            selection_method=selection_method
        )
        self.title_abstract_screener.threshold = rating_threshold

        self.full_text_screener = NumericFullTextScreener(
            llm=llm,
            embeddings=embeddings,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            embeddings_path=self.embeddings_path2,  # Pass this explicitly
            verbose=verbose,
            selection_method=selection_method
        )
        self.full_text_screener.threshold = rating_threshold

        self.logger.info(
            f"Using NumericAcademate with {selection_method} selection (threshold: {self.rating_threshold})")


    def generate_excel_report(self, screening_type='screening1'):
        """
        Generate an Excel report with color-coded ratings.

        Args:
            screening_type (str, optional): Screening type. Defaults to 'screening1'.

        Returns:
            str: Path to Excel file
        """
        import xlsxwriter
        import pandas as pd

        filename = f"{self.vector_store_path}/{screening_type}_screening_report.xlsx"

        # Select the appropriate record2answer dictionary and DataFrame
        if screening_type == 'screening1':
            record2answer = self.title_abstract_screener.record2answer
            if self.results_screening1 is not None:
                article_df = self.results_screening1.copy()
            else:
                article_df = self.literature_df.copy()
        elif screening_type == 'screening2':
            record2answer = self.full_text_screener.record2answer
            if self.results_screening2 is not None:
                article_df = self.results_screening2.copy()
            else:
                raise ValueError("No screening2 results found. Please run screening2 first.")
        else:
            raise ValueError("screening_type must be 'screening1' or 'screening2'")

        # Prepare a list to hold the data
        data = []

        # Loop over each article and extract the ratings and reasons
        for uniqueid, answers in record2answer.items():
            row = {'uniqueid': uniqueid}
            for criterion in self.criteria_dict.keys():
                if criterion in answers:
                    # Get rating and reason
                    if isinstance(answers[criterion], dict):
                        rating = answers[criterion].get('rating', 0.0)
                        reason = answers[criterion].get('reason', None)
                    else:
                        rating = answers[criterion]
                        reason = None

                    # Convert string rating to float if needed
                    if isinstance(rating, str):
                        try:
                            rating = float(rating)
                        except ValueError:
                            rating = 0.0

                    # Ensure rating is between 0 and 1
                    rating = max(0.0, min(1.0, float(rating)))
                else:
                    rating = 0.0
                    reason = None

                row[f'{criterion}_rating'] = rating
                row[f'{criterion}_reason'] = reason
            data.append(row)

        if not data:
            self.logger.warning(f"No valid records found for {screening_type}")
            return None

        # Create a DataFrame from the data
        df_answers = pd.DataFrame(data)

        # Merge the answers DataFrame with the article DataFrame
        merged_df = pd.merge(article_df, df_answers, on='uniqueid', how='right')

        # Write to Excel with conditional formatting for ratings
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, sheet_name=screening_type, index=False)

            workbook = writer.book
            worksheet = writer.sheets[screening_type]

            # Create a color scale format for ratings
            # Green for high ratings, yellow for middle, red for low
            rating_format = workbook.add_format({'num_format': '0.00'})
            wrap_format = workbook.add_format({'text_wrap': True})

            # Set column widths and formats
            for i, col in enumerate(merged_df.columns):
                if col.endswith('_rating'):
                    worksheet.set_column(i, i, 12, rating_format)

                    # Add color scale conditional formatting
                    worksheet.conditional_format(
                        1, i, len(merged_df), i,
                        {'type': '3_color_scale',
                         'min_color': '#FF6666',  # Red
                         'mid_color': '#FFFF99',  # Yellow
                         'max_color': '#66FF66',  # Green
                         'min_type': 'num',
                         'mid_type': 'num',
                         'max_type': 'num',
                         'min_value': 0,
                         'mid_value': 0.5,
                         'max_value': 1}
                    )
                elif col.endswith('_reason'):
                    worksheet.set_column(i, i, 40, wrap_format)
                elif col == 'uniqueid':
                    worksheet.set_column(i, i, 15)
                elif col in ['Record', self.content_column]:
                    worksheet.set_column(i, i, 60, wrap_format)
                else:
                    worksheet.set_column(i, i, 12)

            # Freeze panes to keep headers visible
            worksheet.freeze_panes(1, 0)

        self.logger.info(f"Excel report with numeric ratings saved to {filename}")
        return filename


    def set_selection_method(self, method, threshold=None):
        """
        Set the article selection method and optionally update the threshold.

        Args:
            method (str): Selection method ('threshold' or 'clustering')
            threshold (float, optional): Rating threshold (0-1). Only used for 'threshold' method.
        """
        valid_methods = ["threshold", "clustering"]
        if method not in valid_methods:
            raise ValueError(f"Invalid selection method: {method}. Must be one of {valid_methods}")

        self.selection_method = method

        # Update screeners
        self.title_abstract_screener.selection_method = method
        self.full_text_screener.selection_method = method

        # Update threshold if provided
        if threshold is not None and 0 <= threshold <= 1:
            self.rating_threshold = threshold
            self.title_abstract_screener.threshold = threshold
            self.full_text_screener.threshold = threshold

        self.logger.info(f"Set selection method to '{method}'" +
                         (f" with threshold {threshold}" if threshold is not None else ""))

    def select_articles_based_on_criteria(self, df):
        """
        Select articles based on numeric criteria using the specified method.

        Args:
            df (pd.DataFrame): DataFrame with criteria rating columns

        Returns:
            pd.DataFrame: DataFrame with predicted_screening column
        """
        from academate.utils.clustering import identify_teams_continuous, analyze_continuous_teams

        self.logger.info(f"Selecting articles using '{self.selection_method}' method...")

        # Create matrix of numeric ratings
        rating_columns = [col for col in df.columns if col.endswith('_rating')]

        if not rating_columns:
            self.logger.warning("No rating columns found. Unable to select articles.")
            df['predicted_screening'] = False
            return df

        if self.selection_method == "threshold":
            # Calculate average rating across criteria
            df['avg_rating'] = df[rating_columns].mean(axis=1)

            # Select articles that meet or exceed the threshold
            df['predicted_screening'] = df['avg_rating'] >= self.rating_threshold

            if self.verbose:
                self.logger.info(
                    f"Selected {df['predicted_screening'].sum()} of {len(df)} articles (threshold: {self.rating_threshold})")

        elif self.selection_method == "clustering":
            if len(df) > 1:  # Need at least 2 articles for clustering
                # Create rating matrix (rows=articles, columns=criteria)
                rating_matrix = df[rating_columns].values

                # Apply continuous clustering
                team_labels, selected_features = identify_teams_continuous(
                    rating_matrix,
                    variance_percentile=75,
                    scale_features=True
                )

                # Analyze clustering results
                feature_names = [col.replace('_rating', '') for col in rating_columns]
                analysis = analyze_continuous_teams(rating_matrix, team_labels, feature_names)

                # Log analysis results
                self.logger.info(f"Clustering Analysis:")
                self.logger.info(
                    f"Team sizes: {analysis['winning_team_size']} winning, {analysis['losing_team_size']} losing")
                self.logger.info(f"Top discriminative features (feature, difference, effect size):")
                for feature, diff, effect in analysis["top_features"][:3]:
                    self.logger.info(f"  {feature}: diff={diff:.2f}, effect_size={effect:.2f}")

                # Assign team labels
                df['predicted_screening'] = team_labels
            else:
                # Fall back to threshold for single article
                self.logger.warning("Not enough articles for clustering. Using threshold instead.")
                df['avg_rating'] = df[rating_columns].mean(axis=1)
                df['predicted_screening'] = df['avg_rating'] >= self.rating_threshold

        return df