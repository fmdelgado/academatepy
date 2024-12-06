
# Academate: Academic Literature Screening Tool


<p align="center">
  <img src="img/academate_logo.png" width="200"/>
</p>



## Overview
Academate is a Python-based tool for automating the screening and analysis of academic literature using Large Language Models (LLMs) and embeddings. It supports a two-stage screening process with PDF document processing capabilities, designed to streamline systematic reviews in academic research.

<p align="center">
  <img src="img/graphical_abstract.drawio.png" width="600"/>
</p>


## Features
- Two-stage literature screening
- Automated PDF downloading from various sources
- PDF text extraction and embedding
- Vector database integration using Chroma
- Concurrent processing for improved performance
- Progress tracking and error handling
- Excel report generation
- PRISMA flow diagram visualization

## Requirements
### Main Dependencies
```
langchain-community
langchain
pandas
tqdm
chromadb
PDFPlumber
habanero
requests
nest_asyncio (for Jupyter notebooks)
xlsxwriter
plotly (for visualizations)
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/fmdelgado/academatepy
cd academate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Setup
```python
from academate import academate
import nest_asyncio  # If using Jupyter notebook

# Apply nest_asyncio if using Jupyter
nest_asyncio.apply()

# Initialize the academate instance
screening = academate(
    topic=None,
    llm=your_llm_model,
    embeddings=your_embeddings_model,
    criteria_dict=your_criteria,
    vector_store_path="path/to/store",
    literature_df=your_dataframe,
    content_column="Record",
    pdf_location="path/to/pdfs",
    verbose=False
)
```

### Criteria Dictionary Format
```python
criteria_dict = {
    "criterion_1": "Description of criterion 1",
    "criterion_2": "Description of criterion 2",
    # ...additional criteria
}
```

### Running Screenings
```python
# Run first screening
results_screening1 = screening.run_screening1()

# Run second screening (PDF-based)
results_screening2 = screening.run_screening2()
```

### Required DataFrame Format
The input DataFrame should contain:
- A unique identifier column (will be created if not present)
- Content column (specified in content_column parameter)
- Optional: 'doi' column for PDF retrieval
- Optional: 'pdf_path' column for existing PDFs

## Directory Structure
```
vector_store_path/
├── embeddings/
│   ├── screening1_embeddings/
│   └── screening2_embeddings/
├── screening1/
│   ├── screening1_predicted_criteria.pkl
│   └── screening1_missing_records.pkl
├── screening2/
│   ├── screening2_predicted_criteria.pkl
│   └── screening2_missing_records.pkl
└── pdfs/
    └── downloaded_pdfs/
```

## Core Components

### Screening Process
1. **First Screening (`run_screening1`):**
   - Analyzes abstracts/titles
   - Creates initial embeddings
   - Generates preliminary results

2. **Second Screening (`run_screening2`):**
   - Downloads and processes PDFs
   - Creates PDF embeddings
   - Performs detailed analysis

### PDF Processing
- Automatic DOI detection
- Multi-source PDF downloading
- Text extraction and chunking
- Vector embeddings creation

### Output Generation
- Excel reports with color coding
- PRISMA flow diagrams
- Detailed logging
- Progress tracking

## Error Handling
- Automatic retries for failed operations
- Comprehensive error logging
- Progress persistence
- Recovery from interruptions

## Using in Jupyter Notebooks
When using in Jupyter notebooks, always include:
```python
import nest_asyncio
nest_asyncio.apply()
```

## Tips for Optimal Use
1. Ensure proper file permissions for storage directories
2. Monitor memory usage with large PDF collections
3. Use checkpointing for long-running processes
4. Review logs for error diagnosis
5. Validate PDF paths before running `screening2`

## Common Issues and Solutions
1. **Event Loop Errors in Jupyter:**
   - Solution: Use nest_asyncio

2. **PDF Download Failures:**
   - Check DOI validity
   - Verify access permissions
   - Review error logs

3. **Memory Issues:**
   - Adjust batch sizes
   - Monitor PDF chunk sizes
   - Use proper garbage collection

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
This project is licensed under the MIT License.

## Contact
For inquiries or support, please contact:  
Fernando Miguel Delgado Chaves  
[fernando.miguel.delgado-chaves@uni-hamburg.de](mailto:fernando.miguel.delgado-chaves@uni-hamburg.de)
