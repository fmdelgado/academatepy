import pandas as pd
from app.pdf_downloading_module.PDF_downloader import ArticleDownloader  # Assuming your class is in a file named your_script.py
import logging
import os

# Example usage (replace with your actual paths and data)
workdir = "/Users/fernando/Documents/Research/academatepy/app/pdf_downloading_module/"
pdf_location = os.path.join(workdir, "downloaded_pdfs")
checkpoint_file = os.path.join(workdir, "download_progress.pkl")
email = "your_email@example.com"  # Replace with your email

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Instantiate the downloader
article_downloader = ArticleDownloader(
    output_directory=pdf_location,
    email=email,
    logger=logger,
    verbose=True  # Set to True if you want more verbose logging
)

# Load your DataFrame
df = pd.read_pickle(f"{workdir}withpdfs_preprocessed_articles_filtered.pkl")
df = df[df.screening1==True]
df = pd.read_pickle(f"{workdir}/final_deduplicated_dataframe.pkl")
df.record.duplicated().value_counts()
# Ensure values are strings and then split
print("Before processing", df['pdf_path'].isna().value_counts(), sum(df['pdf_path'].isna().value_counts()))


df['filename'] = df['pdf_path'].apply(lambda x: str(x).split("/")[-1] if isinstance(x, str) else None)
# Display the DataFrame
print(df)
df['pdf_path'] = pdf_location + '/' + df['filename']

# Process the DataFrame
downloadpdfs= ArticleDownloader(output_directory=pdf_location, email=email, logger=logger, verbose=True)
self = downloadpdfs

downloaded_articles_df = downloadpdfs.process_dataframe(df, checkpoint_file=checkpoint_file)

# Save the updated DataFrame
downloaded_articles_df.to_pickle(f"{workdir}/final_dataframe_with_pdfs.pkl")

# Deduplicate the final DataFrame (if necessary)
final_deduplicated_df = article_downloader.deduplicate_dataframe(downloaded_articles_df)
print("After processing", final_deduplicated_df['pdf_path'].isna().value_counts(), sum(final_deduplicated_df['pdf_path'].isna().value_counts()))
# Save the deduplicated DataFrame
final_deduplicated_df.to_pickle(f"{workdir}/final_deduplicated_dataframe.pkl")
final_deduplicated_df.to_excel(f"{workdir}/final_deduplicated_dataframe.xlsx")

