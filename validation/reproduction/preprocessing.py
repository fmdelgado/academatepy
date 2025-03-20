import os
import pandas as pd
import re
import shutil

# Set paths
input_file = "/Users/fernando/Documents/Research/academatepy/validation/reproduction/database_summary.xlsx"
output_dir = "/Users/fernando/Documents/Research/academatepy/validation/reproduction"
output_file = os.path.join(output_dir, "preprocessed_articles_filtered.pkl")
pdf_dir = "/Users/fernando/Documents/Research/academatepy/validation/reproduction/pdfs_renamed"



# Function to normalize filenames
def normalize_filename(filename):
    if not isinstance(filename, str):
        return filename

    # Remove file extension if present
    base_name = os.path.splitext(filename)[0]

    # Convert to lowercase, replace spaces with underscores
    normalized = base_name.lower()

    # Remove special characters except underscores and alphanumeric
    normalized = re.sub(r'[^\w]', '_', normalized)
    # Replace multiple underscores with a single one
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    return normalized


# Function to generate uniqueid consistently
def generate_uniqueid(row):
    """
    Generates a consistent row ID based on the normalized description.
    """
    if 'Record' in row and pd.notna(row['Record']):
        text = row['Record']
    elif 'title' in row and pd.notna(row['title']):
        text = row['title']
    else:
        return None

    # Normalize while preserving some punctuation
    normalized_description = re.sub(r'[^a-zA-Z0-9 \-():]', '', text)
    # Normalize whitespace
    normalized_description = ' '.join(normalized_description.split())

    key_string = f"{normalized_description}"
    import hashlib
    id_record = hashlib.sha256(key_string.encode()).hexdigest()[:20]
    return id_record


# Load the Excel file
df = pd.read_excel(input_file)

# Create Record column first (needed for uniqueid generation)
df['Record'] = df.apply(
    lambda row: f"{row['title']}\n\n{row['abstract']}" if pd.notna(row['title']) and pd.notna(
        row['abstract']) else None,
    axis=1
)
#
# # Generate uniqueid for each record
# df.loc[:, 'pdf_name'] = df['pdf_name'].fillna('')  # Use .loc to avoid the warning
# test = df[df.pdf_name.str.contains('NO')]
df.loc[:, 'pdf_irretrievable'] = df.pdf_name.fillna('').str.contains('NO')

df['uniqueid'] = df.apply(generate_uniqueid, axis=1)

# Create a mapping of original to normalized filenames
filename_mapping = {}
for idx, row in df.iterrows():
    if pd.notna(row['pdf_name']):
        original = row['pdf_name'] + '.pdf'
        # Use uniqueid as part of filename to ensure consistency
        if pd.notna(row['uniqueid']):
            normalized = f"{row['uniqueid']}_{normalize_filename(original)}.pdf"
        else:
            normalized = normalize_filename(original) + '.pdf'
        filename_mapping[original] = normalized
        df.at[idx, 'pdf_name'] = normalized

# Remove all rows that don't contain title or abstract
df.dropna(subset=['title', 'abstract'], how='any', inplace=True)

# Print a list of the problematic files mentioned in the error
problem_files = ['ji_et_al.pdf', 'xin_et_al.pdf', 'cao_et_al.pdf', 'hao_et_al.pdf']
print("Checking problematic files...")
for file in problem_files:
    matching_rows = df[df['pdf_name'].str.contains(normalize_filename(file), case=False, na=False)]
    if not matching_rows.empty:
        print(f"Found {len(matching_rows)} matches for {file}:")
        for _, row in matching_rows.iterrows():
            print(f"  uniqueid: {row['uniqueid']}, pdf_name: {row['pdf_name']}")
    else:
        print(f"No matches found for {file}")

# Rename actual PDF files in the directory
if os.path.exists(pdf_dir):
    print(f"Renaming PDF files in {pdf_dir}...")

    # Create a backup of the original directory
    backup_dir = os.path.join(output_dir, "pdfs_backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(pdf_dir, backup_dir)
    print(f"Original PDF files backed up to {backup_dir}")

    # Get list of files in the directory
    existing_files = os.listdir(pdf_dir)
    print(f"Found {len(existing_files)} files in {pdf_dir}")

    # Check for the problematic files
    for file in problem_files:
        if file in existing_files:
            print(f"Found problematic file in directory: {file}")
        else:
            print(f"Problematic file not found in directory: {file}")

    # Rename files
    renamed_count = 0
    for original, normalized in filename_mapping.items():
        original_path = os.path.join(pdf_dir, original)
        normalized_path = os.path.join(pdf_dir, normalized)

        if os.path.exists(original_path) and original != normalized:
            # If the destination file already exists, remove it first
            if os.path.exists(normalized_path):
                os.remove(normalized_path)

            # Rename the file
            try:
                os.rename(original_path, normalized_path)
                renamed_count += 1
                print(f"Renamed: {original} -> {normalized}")
            except Exception as e:
                print(f"ERROR renaming {original}: {str(e)}")
            print(f"Renamed: {original} -> {normalized}")

    print(f"Renamed {renamed_count} PDF files")
else:
    print(f"Warning: PDF directory {pdf_dir} not found. Files not renamed.")

print("Total number of records:", len(df))
print("Records that pass screening 1:", len(df[df.screening1 == True]))
print("Records that pass screening 2:", len(df[df.screening2 == True]))

# Save the processed DataFrame
df[['title', 'AUTHORS', 'Record', 'PMID', 'PMCID/ISSN', 'DOI', 'pdf_name', 'screening1', 'screening2', 'abstract',
    'journal', 'publication_date', 'uniqueid', 'pdf_irretrievable']].to_pickle(output_file)
print(f"Processed data saved to {output_file}")

# Create a mapping file for reference
mapping_df = pd.DataFrame(list(filename_mapping.items()), columns=['Original', 'Normalized'])
mapping_df.to_csv(os.path.join(output_dir, "filename_mapping.csv"), index=False)
print(f"Filename mapping saved to {os.path.join(output_dir, 'filename_mapping.csv')}")


# Load your database with the expected filenames
df = pd.read_pickle(
    '/Users/fernando/Documents/Research/academatepy/validation/reproduction/preprocessed_articles_filtered.pkl')

# PDF directory and DataFrame validation
# Add this after your main processing, before saving the final pickle
pdf_dir = '/Users/fernando/Documents/Research/academatepy/validation/reproduction/pdfs_renamed'
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files")

# Clear existing paths
df['pdf_path'] = None
screening1_passed = df[df['screening1'] == True].copy()

# Match using multiple strategies
matched = set()
for idx, row in screening1_passed.iterrows():
    uniqueid = row['uniqueid']

    # Try various matching strategies
    for pdf in pdf_files:
        if pdf in matched:
            continue

        # Match by uniqueid, author, or keywords from title
        if (uniqueid in pdf or
                (pd.notna(row['AUTHORS']) and row['AUTHORS'].split(',')[0].lower() in pdf.lower()) or
                (pd.notna(row['title']) and any(kw.lower() in pdf.lower() for kw in row['title'].split()[:3]))):
            df.loc[idx, 'pdf_path'] = os.path.join(pdf_dir, pdf)
            df.loc[idx, 'pdf_name'] = pdf
            matched.add(pdf)
            break

# Save results
print(f"Matched {len(matched)}/{len(screening1_passed)} records that passed screening1")
df.to_pickle(output_file)