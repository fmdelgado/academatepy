"""
Single-database variant of Academate.
"""

import os
import time
import pandas as pd
from academate.base.core import AcademateBase
from academate.screeners.title_abstract import TitleAbstractScreener
from academate.screeners.full_text import FullTextScreener
from academate.embedders.literature import LiteratureEmbedder
from academate.embedders.pdf import PDFEmbedder
from academate.utils.pdf import PDFDownloader


class AcademateSingleDB(AcademateBase):
    """
    Single-database variant of Academate.

    This variant uses a single vector database to store embeddings.
    """

    def __init__(self, topic, llm, embeddings, criteria_dict, vector_store_path,
                 literature_df=None, content_column="record", embeddings_path=None,
                 pdf_location=None, verbose=False, chunksize=25, selection_method="all_criteria"):

        """
        Initialize AcademateSingleDB.

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
        """
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
        # Store selection method
        self.selection_method = selection_method
        self.logger.info(f"Using article selection method: {selection_method}")

        # Initialize components
        self.title_abstract_screener = TitleAbstractScreener(
            llm=llm,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            verbose=verbose,
            selection_method="all_criteria"  # Default method
        )

        self.literature_embedder = LiteratureEmbedder(
            embeddings=embeddings,
            embedding_type='screening1',
            vector_store_path=vector_store_path,
            embeddings_path=self.embeddings_path1,  # Pass the pre-configured path
            verbose=verbose
        )

        self.pdf_downloader = PDFDownloader(
            output_directory=self.pdf_location,
            email="user@example.com",  # You can make this configurable
            logger=self.logger,
            verbose=verbose
        )

        self.pdf_embedder = PDFEmbedder(
            embeddings=embeddings,
            embedding_type='screening2',
            vector_store_path=vector_store_path,
            embeddings_path=self.embeddings_path2,  # Pass the pre-configured path
            verbose=verbose
        )

        self.full_text_screener = FullTextScreener(
            llm=llm,
            embeddings=embeddings,
            criteria_dict=criteria_dict,
            vector_store_path=vector_store_path,
            embeddings_path=self.embeddings_path2,  # Pass this explicitly
            verbose=verbose,
            selection_method="all_criteria"  # Default method
        )
        #
        # self.full_text_screener = FullTextScreener(
        #     llm=google_llm,
        #     embeddings=google_embeddings,
        #     criteria_dict=criteria_dict,
        #     vector_store_path=outdir,
        #     verbose=True
        # )

        # Initialize tracking attributes
        self.screening1_record2answer = {}
        self.screening1_missing_records = set()
        self.uniqueids_withoutPDFtext = set()
        self.screening2_record2answer = {}
        self.screening2_PDFdownload_error = set()
        self.screening2_PDFembedding_error = set()
        self.screening2_missing_records = set()
        self.vectorDB_screening1 = None
        self.vectorDB_screening2 = None
        self.db_qa = None
        self.analysis_time = None

    def run_screening1(self):
        """
        Run the title/abstract screening process.

        Returns:
            pd.DataFrame: Screening results
        """
        if self.literature_df is None:
            raise ValueError("No literature DataFrame provided. Please provide one during initialization.")

        self.logger.info(f"Total number of records: {len(self.literature_df)}")

        # Embed literature if needed
        if not self.literature_embedder.embeddings_exist():
            self.logger.info("Embedding literature...")
            self.vectorDB_screening1 = self.literature_embedder.embed(
                self.literature_df,
                content_column=self.content_column
            )

        # Run screening
        start_time = time.time()
        self.results_screening1 = self.title_abstract_screener.screen(self.literature_df)

        # Validate mutual exclusivity
        self.title_abstract_screener.validate_mutual_exclusivity()

        # Record runtime
        self.analysis_time = time.time() - start_time
        self.results_screening1['runtime_scr1'] = self.analysis_time

        # Rename criteria columns to add '_scr1'
        self.results_screening1.rename(
            columns={key: key + '_scr1' for key in self.criteria_dict.keys()},
            inplace=True
        )

        # If 'predicted_screening' column exists, rename to 'predicted_screening1'
        if 'predicted_screening' in self.results_screening1.columns:
            self.results_screening1.rename(
                columns={'predicted_screening': 'predicted_screening1'},
                inplace=True
            )

        self.logger.info(
            f"Screening 1 complete. {self.results_screening1['predicted_screening1'].sum()} articles passed.")

        return self.results_screening1

    def run_screening2(self):
        """
        Run the full-text screening process.

        Returns:
            pd.DataFrame: Screening results
        """
        # Check if screening1 has been run
        if self.results_screening1 is None:
            # Check if PDF paths are provided
            if self.results_screening1 is not None and 'pdf_path' in self.results_screening1.columns:
                self.logger.info("Using provided PDF paths for screening2.")
                self.results_screening2 = self.results_screening1.copy()
            else:
                raise ValueError(
                    "Either run_screening1() must be called before run_screening2() "
                    "or 'pdf_path' column must be provided in the initial DataFrame."
                )
        else:
            # Filter articles that passed screening1
            if 'predicted_screening1' not in self.results_screening1.columns:
                self.logger.warning(
                    "Warning: 'predicted_screening1' column not found. Using all records for screening2.")
                self.results_screening2 = self.results_screening1.copy()
            else:
                self.results_screening2 = self.results_screening1[
                    self.results_screening1['predicted_screening1'] == True
                    ]

        self.logger.info(f"Records selected for PDF processing: {len(self.results_screening2)}")

        # Download PDFs if needed
        if 'pdf_path' not in self.results_screening2.columns or self.results_screening2['pdf_path'].isnull().any():
            self.logger.info("Downloading PDFs...")
            self.results_screening2 = self.pdf_downloader.process_dataframe(
                self.results_screening2,
                checkpoint_file=f"{self.vector_store_path}/download_progress.pkl"
            )
            # Update error tracking
            self.screening2_PDFdownload_error = self.pdf_downloader.pdf_download_error

        # Embed PDFs
        if len(self.results_screening2[self.results_screening2['pdf_path'].notna()]) > 0:
            self.logger.info("Embedding PDFs...")
            self.pdf_embedder.embed(self.results_screening2)
            # Update error tracking
            self.screening2_PDFembedding_error = self.pdf_embedder.pdf_embedding_error
            self.uniqueids_withoutPDFtext = self.pdf_embedder.uniqueids_withoutPDFtext

        # Log embedding errors
        if len(self.screening2_PDFembedding_error) > 0 or len(self.uniqueids_withoutPDFtext) > 0:
            self.logger.warning(f"Records with PDF embedding error: {len(self.screening2_PDFembedding_error)}")
            self.logger.warning(f"Records without PDF-selectable text: {len(self.uniqueids_withoutPDFtext)}")

        # Filter out records with PDF errors
        filtered_df = self.results_screening2[
            ~self.results_screening2['uniqueid'].isin(self.screening2_PDFembedding_error)]
        self.logger.info(f"Records with PDF proceeding for screening2: {len(filtered_df)}")

        # Run full-text screening if we have valid PDFs
        if len(filtered_df) > 0:
            # ADD CODE HERE TO RESET MISSING RECORDS SET BEFORE PROCESSING
            # Reset missing_records to prevent accumulation between runs
            self.full_text_screener.missing_records = set()

            # ALSO ADD SYNCHRONIZATION WITH EXISTING RECORD2ANSWER DATA
            # If we're reusing old data, synchronize the state
            if os.path.exists(f"{self.screening2_dir}/screening2_missing_records.pkl"):
                import pickle
                with open(f"{self.screening2_dir}/screening2_missing_records.pkl", 'rb') as file:
                    old_missing = pickle.load(file)
                    self.logger.info(f"Found {len(old_missing)} previously missing records, but will not use them")

            start_time = time.time()
            self.results_screening2 = self.full_text_screener.screen(filtered_df)

            # Validate mutual exclusivity
            self.full_text_screener.validate_mutual_exclusivity()

            # Record runtime
            self.analysis_time = time.time() - start_time
            self.results_screening2['runtime_scr2'] = self.analysis_time

            # Rename criteria columns to add '_scr2'
            self.results_screening2.rename(
                columns={key: key + '_scr2' for key in self.criteria_dict.keys()},
                inplace=True
            )

            # Rename predicted_screening to predicted_screening2 if it exists
            if 'predicted_screening' in self.results_screening2.columns:
                self.results_screening2.rename(
                    columns={'predicted_screening': 'predicted_screening2'},
                    inplace=True
                )

            self.logger.info(
                f"Screening 2 complete. {self.results_screening2.get('predicted_screening2', pd.Series()).sum()} articles passed.")
        else:
            self.logger.warning("No PDFs were successfully processed. Skipping screening2.")

        return self.results_screening2

    def generate_excel_report(self, screening_type='screening1'):
        """
        Generate an Excel report of screening results.

        Args:
            screening_type (str, optional): Screening type. Defaults to 'screening1'.

        Returns:
            str: Path to Excel file
        """
        import xlsxwriter

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

        # Loop over each article and extract the labels and reasons
        for uniqueid, answers in record2answer.items():
            row = {'uniqueid': uniqueid}
            for criterion in self.criteria_dict.keys():
                if criterion in answers:
                    # Directly access 'label' and 'reason' without .get()
                    label = answers[criterion]['label'] if isinstance(answers[criterion], dict) else answers[
                        criterion]
                    reason = answers[criterion]['reason'] if isinstance(answers[criterion], dict) else None
                else:
                    label = None
                    reason = None

                # Ensure label is boolean
                if isinstance(label, bool):
                    pass
                elif isinstance(label, str):
                    label_lower = label.lower().strip()
                    if label_lower in ['true', 't', 'yes']:
                        label = True
                    elif label_lower in ['false', 'f', 'no']:
                        label = False
                    else:
                        label = False  # Default to False if not recognized
                else:
                    label = False  # Default to False if not recognized

                row[f'{criterion}_label'] = label
                row[f'{criterion}_reason'] = reason
            data.append(row)

        if not data:
            self.logger.warning(f"No valid records found for {screening_type}")
            return None

        # Create a DataFrame from the data
        df_answers = pd.DataFrame(data)

        # Merge the answers DataFrame with the article DataFrame
        merged_df = pd.merge(article_df, df_answers, on='uniqueid', how='right')

        # Reorder columns to group logically: uniqueid, title/record, then alternating label and reason
        base_cols = ['uniqueid']
        if 'title' in merged_df.columns:
            base_cols.append('title')
        elif self.content_column in merged_df.columns:
            base_cols.append(self.content_column)

        # Get criteria columns without '_label' suffix for pairing
        criteria_cols = [col.replace('_label', '') for col in merged_df.columns
                         if col.endswith('_label') and col != 'predicted_screening_label']

        # Create pairs of label and reason columns
        paired_cols = []
        for criterion in criteria_cols:
            paired_cols.extend([f"{criterion}_label", f"{criterion}_reason"])

        # Add 'predicted_screening' at the end if it exists
        if 'predicted_screening' in merged_df.columns:
            paired_cols.append('predicted_screening')
        elif f'predicted_{screening_type}' in merged_df.columns:
            paired_cols.append(f'predicted_{screening_type}')

        # Combine all columns in desired order
        all_cols = base_cols + paired_cols
        # Filter to only include columns that actually exist
        cols_to_use = [col for col in all_cols if col in merged_df.columns]
        merged_df = merged_df[cols_to_use]

        # Now, write the merged_df to Excel with formatting
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, sheet_name=screening_type, index=False)

            workbook = writer.book
            worksheet = writer.sheets[screening_type]

            # Define formats
            true_format = workbook.add_format({'bg_color': '#75E69D'})  # Light green
            false_format = workbook.add_format({'bg_color': '#E66F60'})  # Light red
            wrap_format = workbook.add_format({'text_wrap': True})

            # Set column widths
            worksheet.set_column('A:A', 10)  # uniqueid

            # Set width for title or content column
            if 'title' in merged_df.columns:
                col_idx = merged_df.columns.get_loc('title')
                worksheet.set_column(col_idx, col_idx, 40)
            elif 'Title' in merged_df.columns:
                col_idx = merged_df.columns.get_loc('Title')
                worksheet.set_column(col_idx, col_idx, 40)
            elif self.content_column in merged_df.columns:
                col_idx = merged_df.columns.get_loc(self.content_column)
                worksheet.set_column(col_idx, col_idx, 80, wrap_format)

            # Apply conditional formatting to boolean columns and set column widths
            for col in merged_df.columns:
                col_idx = merged_df.columns.get_loc(col)
                col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)
                last_row = len(merged_df)

                if col.endswith('_label') or col == f'predicted_{screening_type}':
                    worksheet.set_column(col_idx, col_idx, 15)
                    worksheet.conditional_format(
                        f'{col_letter}2:{col_letter}{last_row + 1}',
                        {'type': 'cell',
                         'criteria': '==',
                         'value': True,
                         'format': true_format}
                    )
                    worksheet.conditional_format(
                        f'{col_letter}2:{col_letter}{last_row + 1}',
                        {'type': 'cell',
                         'criteria': '==',
                         'value': False,
                         'format': false_format}
                    )
                elif col.endswith('_reason'):
                    worksheet.set_column(col_idx, col_idx, 40, wrap_format)

            # Freeze panes to keep headers visible
            worksheet.freeze_panes(1, 0)

        self.logger.info(f"Excel report saved to {filename}")
        return filename

    def create_PRISMA_visualization(self, save_results=True):
        """
        Create a PRISMA flow diagram.

        Args:
            save_results (bool, optional): Save results to file. Defaults to True.

        Returns:
            object: PRISMA visualization object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            self.logger.error("Plotly is required for PRISMA visualization. Install with 'pip install plotly'.")
            return None

        # Prepare data
        df = self.literature_df.copy()

        # Add screening1 results
        if self.results_screening1 is not None:
            df['analyzed_screening1'] = ~df['uniqueid'].isin(self.title_abstract_screener.missing_records)
            screening1_results = dict(zip(
                self.results_screening1['uniqueid'],
                self.results_screening1.get('predicted_screening1', False)
            ))
            df['predicted_screening1'] = df['uniqueid'].map(screening1_results)
        else:
            df['analyzed_screening1'] = False
            df['predicted_screening1'] = False

        # Add PDF download results
        if hasattr(self, 'screening2_PDFdownload_error'):
            df['PDF_downloaded'] = (
                    (df['predicted_screening1'] == True) &
                    (~df['uniqueid'].isin(self.screening2_PDFdownload_error))
            )
        else:
            df['PDF_downloaded'] = False

        # Add PDF embedding results
        if hasattr(self, 'screening2_PDFembedding_error'):
            df['PDF_embedding_error'] = (
                    (df['predicted_screening1'] == True) &
                    (~df['uniqueid'].isin(self.screening2_PDFdownload_error)) &
                    (df['uniqueid'].isin(self.screening2_PDFembedding_error))
            )
        else:
            df['PDF_embedding_error'] = False

        # Add screening2 results
        if self.results_screening2 is not None:
            screening2_results = dict(zip(
                self.results_screening2['uniqueid'],
                self.results_screening2.get('predicted_screening2', False)
            ))
            df['predicted_screening2'] = df['uniqueid'].map(screening2_results)
        else:
            df['predicted_screening2'] = False

        # Calculate counts for PRISMA diagram
        total_records = len(df)
        records_screened = df['analyzed_screening1'].sum()
        records_included_scr1 = df['predicted_screening1'].sum()
        records_excluded_scr1 = records_screened - records_included_scr1

        pdfs_available = df['PDF_downloaded'].sum()
        pdfs_with_errors = df['PDF_embedding_error'].sum()
        pdfs_analyzed = pdfs_available - pdfs_with_errors

        records_included_scr2 = df['predicted_screening2'].sum()
        records_excluded_scr2 = pdfs_analyzed - records_included_scr2

        # Create PRISMA diagram with Plotly
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "sankey"}]])

        # Define nodes
        node_labels = [
            "Total Records",  # 0
            "Records Screened",  # 1
            "Records Excluded (Screening 1)",  # 2
            "Records for Full-Text",  # 3
            "PDFs Available",  # 4
            "PDFs with Errors",  # 5
            "PDFs Analyzed",  # 6
            "Records Excluded (Screening 2)",  # 7
            "Records Included"  # 8
        ]

        # Define links
        link_sources = [0, 1, 1, 3, 4, 4, 6, 6]
        link_targets = [1, 2, 3, 4, 5, 6, 7, 8]
        link_values = [
            records_screened,  # Total -> Screened
            records_excluded_scr1,  # Screened -> Excluded Scr1
            records_included_scr1,  # Screened -> For Full-Text
            pdfs_available,  # For Full-Text -> PDFs Available
            pdfs_with_errors,  # PDFs Available -> PDF Errors
            pdfs_analyzed,  # PDFs Available -> PDFs Analyzed
            records_excluded_scr2,  # PDFs Analyzed -> Excluded Scr2
            records_included_scr2  # PDFs Analyzed -> Included
        ]

        # Define link colors
        link_colors = [
            "rgba(0, 0, 255, 0.4)",  # Blue for flow
            "rgba(255, 0, 0, 0.4)",  # Red for exclusion
            "rgba(0, 255, 0, 0.4)",  # Green for inclusion
            "rgba(0, 0, 255, 0.4)",  # Blue for flow
            "rgba(255, 0, 0, 0.4)",  # Red for errors
            "rgba(0, 0, 255, 0.4)",  # Blue for flow
            "rgba(255, 0, 0, 0.4)",  # Red for exclusion
            "rgba(0, 255, 0, 0.4)"  # Green for inclusion
        ]

        # Create Sankey diagram
        fig.add_trace(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color="blue"
                ),
                link=dict(
                    source=link_sources,
                    target=link_targets,
                    value=link_values,
                    color=link_colors
                )
            )
        )

        # Update layout
        fig.update_layout(
            title_text="PRISMA Flow Diagram",
            font_size=12,
            height=800,
            width=1000
        )

        # Add counts as annotations
        annotations = [
            dict(
                x=0.01, y=0.99,
                xref="paper", yref="paper",
                text=f"Total Records: {total_records}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.92,
                xref="paper", yref="paper",
                text=f"Records Screened: {records_screened}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.85,
                xref="paper", yref="paper",
                text=f"Records Excluded (Screening 1): {records_excluded_scr1}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.78,
                xref="paper", yref="paper",
                text=f"Records for Full-Text: {records_included_scr1}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.71,
                xref="paper", yref="paper",
                text=f"PDFs Available: {pdfs_available}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.64,
                xref="paper", yref="paper",
                text=f"PDFs with Errors: {pdfs_with_errors}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.57,
                xref="paper", yref="paper",
                text=f"PDFs Analyzed: {pdfs_analyzed}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.50,
                xref="paper", yref="paper",
                text=f"Records Excluded (Screening 2): {records_excluded_scr2}",
                showarrow=False
            ),
            dict(
                x=0.01, y=0.43,
                xref="paper", yref="paper",
                text=f"Records Included: {records_included_scr2}",
                showarrow=False
            )
        ]
        fig.update_layout(annotations=annotations)

        # Save results if requested
        if save_results:
            html_path = os.path.join(self.vector_store_path, "prisma_flow_diagram.html")
            fig.write_html(html_path)
            self.logger.info(f"PRISMA flow diagram saved as {html_path}")

        return fig

    def set_selection_method(self, method):
        """
        Set the article selection method.

        Args:
            method (str): Selection method ('all_criteria' or 'clustering')
        """
        valid_methods = ["all_criteria", "clustering"]
        if method not in valid_methods:
            raise ValueError(f"Invalid selection method: {method}. Must be one of {valid_methods}")

        self.selection_method = method
        self.logger.info(f"Set selection method to '{method}'")

    def select_articles_based_on_criteria(self, df):
        """
        Select articles based on criteria using the specified method.

        Args:
            df (pd.DataFrame): DataFrame with criteria columns

        Returns:
            pd.DataFrame: DataFrame with predicted_screening column
        """
        from academate.utils.clustering import identify_teams, analyze_teams

        self.logger.info(f"Selecting articles using '{self.selection_method}' method...")

        # Ensure all criteria columns are boolean
        for column in self.criteria_dict.keys():
            if column in df.columns:
                df[column] = df[column].astype(bool)

        if self.selection_method == "all_criteria":
            # Default method: all criteria must be true
            df['predicted_screening'] = df[list(self.criteria_dict.keys())].all(axis=1)

        elif self.selection_method == "clustering":
            if len(df) > 1:  # Need at least 2 records for clustering
                # Create boolean matrix
                boolean_matrix = df[list(self.criteria_dict.keys())].values

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
                df['predicted_screening'] = team_labels
            else:
                # Fall back to default if not enough records
                self.logger.warning("Not enough records for clustering. Using all criteria instead.")
                df['predicted_screening'] = df[list(self.criteria_dict.keys())].all(axis=1)

        return df