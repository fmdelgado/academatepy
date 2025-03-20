#!/usr/bin/env python
"""
Extended version of ArticleMetadataRetriever with checkpointing and detailed reporting.
"""

import os
import hashlib
import pandas as pd
from collections import Counter
from academate.research_module.metadata_retriever import ArticleMetadataRetriever


class EnhancedArticleMetadataRetriever(ArticleMetadataRetriever):
    """Enhanced version with checkpointing and detailed reporting."""

    def summarize_results(self, df):
        """Generate a detailed summary of processing results."""
        summary = {
            'total_articles': len(df),
            'doi_success': 0,
            'doi_failure': 0,
            'metadata_success': 0,
            'metadata_failure': 0,
            'suspicious_entries': 0
        }

        # Count DOI successes/failures
        has_valid_doi = df['doi'].apply(lambda x: self.is_valid_doi(x) if pd.notna(x) else False)
        summary['doi_success'] = has_valid_doi.sum()
        summary['doi_failure'] = len(df) - summary['doi_success']

        # Count metadata successes/failures
        has_title = df['title'].notna()
        has_abstract = df['abstract'].notna()
        has_metadata = has_title | has_abstract
        summary['metadata_success'] = has_metadata.sum()
        summary['metadata_failure'] = len(df) - summary['metadata_success']

        # Count suspicious entries
        if 'metadata_validation' in df.columns:
            summary['suspicious_entries'] = (df['metadata_validation'] != 'ok').sum()

        # Additional details
        if 'doi_source' in df.columns:
            doi_sources = df['doi_source'].value_counts().to_dict()
            summary['doi_sources'] = doi_sources

        if 'metadata_sources' in df.columns:
            # Split the comma-separated values
            all_sources = []
            for sources in df['metadata_sources'].dropna():
                if isinstance(sources, str):
                    all_sources.extend([s.strip() for s in sources.split(',')])

            summary['metadata_sources'] = dict(Counter(all_sources))

        return summary

    def print_summary(self, summary):
        """Print a formatted summary to the console."""
        print("\n===== ARTICLE METADATA RETRIEVAL SUMMARY =====")
        print(f"Total articles processed: {summary['total_articles']}")
        print(f"DOI identification:")
        print(
            f"  ✓ Success: {summary['doi_success']} articles ({summary['doi_success'] / summary['total_articles'] * 100:.1f}%)")
        print(
            f"  ✗ Failure: {summary['doi_failure']} articles ({summary['doi_failure'] / summary['total_articles'] * 100:.1f}%)")
        print(f"Metadata retrieval:")
        print(
            f"  ✓ Success: {summary['metadata_success']} articles ({summary['metadata_success'] / summary['total_articles'] * 100:.1f}%)")
        print(
            f"  ✗ Failure: {summary['metadata_failure']} articles ({summary['metadata_failure'] / summary['total_articles'] * 100:.1f}%)")

        if 'suspicious_entries' in summary and summary['suspicious_entries'] > 0:
            print(f"Suspicious entries: {summary['suspicious_entries']}")

        if 'doi_sources' in summary:
            print("\nDOI sources:")
            for source, count in sorted(summary['doi_sources'].items(), key=lambda x: x[1], reverse=True):
                if pd.notna(source):
                    print(f"  {source}: {count}")

        if 'metadata_sources' in summary:
            print("\nMetadata sources:")
            for source, count in sorted(summary['metadata_sources'].items(), key=lambda x: x[1], reverse=True):
                if pd.notna(source):
                    print(f"  {source}: {count}")

        print("=================================================")

    def load_checkpoint(self, checkpoint_path):
        """Load a previously saved checkpoint."""
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_df = pd.read_excel(checkpoint_path)
                self.logger.info(
                    f"Loaded checkpoint with {len(checkpoint_df)} processed articles from {checkpoint_path}")
                return checkpoint_df
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")

        self.logger.info(f"No checkpoint found at {checkpoint_path}")
        return None

    def save_checkpoint(self, df, checkpoint_path):
        """Save the current state as a checkpoint."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            df.to_excel(checkpoint_path, index=False)
            self.logger.info(f"Saved checkpoint with {len(df)} processed articles to {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
            return False

    def process_dataframe_with_checkpoint(self, df, checkpoint_path=None):
        """Process a DataFrame with checkpointing to avoid reprocessing items."""
        # Create a unique identifier for each row if none exists
        if 'uniqueid' not in df.columns:
            self.logger.info("Creating uniqueid column for tracking")
            # Use title + authors + year if available, otherwise a hash of all columns
            if all(col in df.columns for col in ['Title', 'Authors', 'Year']):
                df['uniqueid'] = df.apply(
                    lambda row: hashlib.md5(
                        f"{str(row['Title'])}|{str(row['Authors'])}|{str(row['Year'])}".encode()).hexdigest(),
                    axis=1
                )
            else:
                # Use all columns to create a uniqueid
                df['uniqueid'] = df.apply(
                    lambda row: hashlib.md5(str(row.values).encode()).hexdigest(),
                    axis=1
                )

        # Load checkpoint if it exists
        checkpoint_df = None
        if checkpoint_path:
            checkpoint_df = self.load_checkpoint(checkpoint_path)

        # If we have a checkpoint, determine which rows need processing
        if checkpoint_df is not None and 'uniqueid' in checkpoint_df.columns:
            # Find rows that haven't been processed yet
            processed_ids = set(checkpoint_df['uniqueid'])
            to_process = df[~df['uniqueid'].isin(processed_ids)].copy()

            self.logger.info(f"Found {len(processed_ids)} already processed articles in checkpoint")
            self.logger.info(f"Need to process {len(to_process)} new articles")

            if len(to_process) == 0:
                self.logger.info("All articles already processed, returning checkpoint data")
                summary = self.summarize_results(checkpoint_df)
                self.print_summary(summary)
                return checkpoint_df, summary

            # Process new rows
            new_results = self.process_dataframe(to_process)

            # Combine with checkpoint data
            result_df = pd.concat([checkpoint_df, new_results], ignore_index=True)

            # Remove any duplicates (by uniqueid)
            result_df = result_df.drop_duplicates(subset='uniqueid', keep='first')

        else:
            # Process the entire DataFrame
            result_df = self.process_dataframe(df)

        # Save checkpoint
        if checkpoint_path:
            self.save_checkpoint(result_df, checkpoint_path)

        # Generate summary
        summary = self.summarize_results(result_df)
        self.print_summary(summary)

        return result_df, summary