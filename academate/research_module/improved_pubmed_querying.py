"""
Improved PubMed query generator.
"""

import pandas as pd
from Bio import Entrez
import logging
from typing import Optional, Dict, List, Any, Union

from academate.research_module.query_extractor import PubMedQueryExtractor
from academate.research_module.llm_response_handler import LLMResponseHandler, detect_llm_type


class ImprovedPubMedQueryGenerator:
    """
    Improved generator for PubMed search queries.

    This class handles generation of PubMed queries using LLMs,
    with robust extraction and error handling.
    """

    def __init__(self, email: str, api_key: str, llm=None, two_step_search_query=False):
        """
        Initialize the PubMedQueryGenerator.

        Args:
            email: Your email address (required by NCBI)
            api_key: Your NCBI API key
            llm: An instance of the LLM for generating queries
            two_step_search_query: Whether to use two-step approach
        """
        self.email = email
        self.api_key = api_key
        self.llm = llm
        self.two_step_search_query = two_step_search_query

        # Set up Entrez
        Entrez.email = self.email
        Entrez.api_key = self.api_key

        # Set up date filters
        self.from_year = None
        self.to_year = None

        # Set up results storage
        self.results = None
        self.last_query = ""

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Set up response handler and query extractor
        self.response_handler = LLMResponseHandler(logger=self.logger)
        self.query_extractor = PubMedQueryExtractor(logger=self.logger)

        # Detect LLM type for prompt formatting
        self.llm_type = detect_llm_type(llm)

    def set_date_filter(self, from_year: Optional[int] = None, to_year: Optional[int] = None) -> None:
        """
        Set date filter for the PubMed query.

        Args:
            from_year: Starting year (inclusive)
            to_year: Ending year (inclusive)
        """
        self.from_year = from_year
        self.to_year = to_year

    def get_date_filter(self) -> str:
        """
        Construct the date filter string for PubMed query.

        Returns:
            str: Date filter in PubMed format
        """
        from_year = self.from_year if self.from_year else "1000"
        to_year = self.to_year if self.to_year else "3000"
        date_filter = f'("{from_year}/01/01"[Date - Publication] : "{to_year}/12/31"[Date - Publication])'
        return date_filter

    def generate_query(self, topic: str, known_pmids: Optional[List[str]] = None) -> str:
        """
        Generate a PubMed query for the given topic.

        Args:
            topic: Research topic or keywords
            known_pmids: Optional list of known PubMed IDs

        Returns:
            str: Generated PubMed query

        Raises:
            ValueError: If no LLM is provided
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        # Generate the query using the LLM
        if self.two_step_search_query:
            query = self._generate_query_with_llm_given_keywords(topic)
        else:
            query = self._generate_query_with_llm(topic)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

        self.last_query = query  # Store the last generated query
        return query

    def _generate_query_with_llm(self, topic: str) -> str:
        """
        Generate a PubMed query using the LLM.

        Args:
            topic: Research topic

        Returns:
            str: Generated query
        """
        # Create prompt template
        prompt_template = f"""You are an expert in creating PubMed search queries.

Generate a PubMed search query for the following research topic:

Topic:
{topic}

Requirements:
1. Identify the main concepts in the topic.
2. For each concept, include synonyms and related terms.
3. Use MeSH terms where appropriate.
4. Use OR to combine synonyms within a concept.
5. Use AND to combine different concepts.
6. Use field tags like [Title/Abstract], [MeSH Terms], etc.
7. Use parentheses to ensure the correct grouping of terms.
8. **Do not include any JSON, keys, or extra characters.**
9. **Provide ONLY the query, and no additional text or explanations.**
10. **Enclose your query between the markers `<START_QUERY>` and `<END_QUERY>`. Failure to do so will result in an invalid query.**

Example:

<START_QUERY>
((("term1"[Title/Abstract] OR "term1 synonym"[Title/Abstract] OR "Term1 MeSH"[MeSH Terms]) AND ("term2"[Title/Abstract] OR "term2 synonym"[Title/Abstract] OR "Term2 MeSH"[MeSH Terms])))
<END_QUERY>

Provide your query below:
"""
        # Format prompt based on LLM type
        formatted_prompt = self.response_handler.format_prompt(prompt_template, self.llm_type)

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)

            # Extract content from response
            content = self.response_handler.extract_content(response)

            # Extract query from content
            query = self.query_extractor.extract_query(content)

            return query

        except Exception as e:
            self.logger.error(f"Error generating query: {str(e)}")
            # Return a basic query as fallback
            return f'("{topic}"[Title/Abstract])'

    def _generate_query_with_llm_given_keywords(self, keywords: str) -> str:
        """
        Generate a PubMed query using the LLM based on predefined keywords.

        Args:
            keywords: Keywords and terms for the query

        Returns:
            str: Generated query
        """
        # Create prompt template
        prompt_template = f"""You are an expert in creating PubMed search queries.

Generate a PubMed search query for the following keywords:
{keywords}

Requirements:
1. Use MeSH terms where appropriate.
2. Use OR to combine synonyms within a concept.
3. Use AND to combine different concepts.
4. Use field tags like [Title/Abstract], [MeSH Terms], etc.
5. Use parentheses to ensure the correct grouping of terms.
6. **Do not include any JSON, keys, or extra characters.**
7. **Provide ONLY the query, and no additional text or explanations.**
8. **Enclose your query between the markers `<START_QUERY>` and `<END_QUERY>`. Failure to do so will result in an invalid query.**

Example:

<START_QUERY>
((("term1"[Title/Abstract] OR "term1 synonym"[Title/Abstract] OR "Term1 MeSH"[MeSH Terms]) AND ("term2"[Title/Abstract] OR "term2 synonym"[Title/Abstract] OR "Term2 MeSH"[MeSH Terms])))
<END_QUERY>

Provide your query below:
"""
        # Format prompt based on LLM type
        formatted_prompt = self.response_handler.format_prompt(prompt_template, self.llm_type)

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)

            # Extract content from response
            content = self.response_handler.extract_content(response)

            # Extract query from content
            query = self.query_extractor.extract_query(content)

            return query

        except Exception as e:
            self.logger.error(f"Error generating query: {str(e)}")
            # Return a basic query as fallback
            terms = [term.strip() for term in keywords.split('\n') if term.strip()]
            if terms:
                return ' AND '.join([f'"{term}"[Title/Abstract]' for term in terms])
            else:
                return f'("{keywords}"[Title/Abstract])'

    def execute_query(self, query: str, max_results: int = 10000) -> None:
        """
        Execute the PubMed query and retrieve results.

        Args:
            query: PubMed query string
            max_results: Maximum number of results to retrieve
        """
        try:
            self.logger.info(f"Executing PubMed query: {query}")

            # Search for PMIDs matching the query
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            pmids = record["IdList"]

            if not pmids:
                self.logger.info("No results found for the query.")
                self.results = pd.DataFrame()
                return

            # Fetch article details for the PMIDs
            self.logger.info(f"Found {len(pmids)} results. Fetching details...")
            articles = self._fetch_article_info(pmids)

            # Store results as DataFrame
            self.results = pd.DataFrame(articles)
            self.logger.info(f"Successfully retrieved {len(self.results)} articles.")

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get query results as a DataFrame.

        Returns:
            pd.DataFrame: Results DataFrame
        """
        return self.results

    def _fetch_article_info(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed information for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article information dictionaries
        """
        try:
            batch_size = 200  # NCBI recommends batches of 200 or less
            articles = []

            for start in range(0, len(pmids), batch_size):
                end = min(start + batch_size, len(pmids))
                batch_pmids = pmids[start:end]

                self.logger.debug(
                    f"Fetching batch {start // batch_size + 1}/{(len(pmids) + batch_size - 1) // batch_size}")

                # Fetch article details
                handle = Entrez.efetch(db="pubmed", id=",".join(batch_pmids), rettype="medline", retmode="xml")
                records = Entrez.read(handle)

                # Process each article
                for record in records['PubmedArticle']:
                    # Extract article information
                    medline_citation = record.get('MedlineCitation', {})
                    article = medline_citation.get('Article', {})

                    # Extract basic fields
                    pmid = str(medline_citation.get('PMID', ''))
                    title = article.get('ArticleTitle', '')

                    # Extract abstract
                    abstract_list = article.get('Abstract', {}).get('AbstractText', [])
                    if isinstance(abstract_list, list):
                        abstract = ' '.join(str(text) for text in abstract_list)
                    else:
                        abstract = str(abstract_list)

                    # Extract authors
                    authors_list = article.get('AuthorList', [])
                    authors = []
                    for author in authors_list:
                        last_name = author.get('LastName', '')
                        fore_name = author.get('ForeName', '')
                        full_name = ' '.join([fore_name, last_name]).strip()
                        if full_name:
                            authors.append(full_name)
                    authors_str = ', '.join(authors)

                    # Extract date
                    journal = article.get('Journal', {})
                    journal_issue = journal.get('JournalIssue', {})
                    pub_date = journal_issue.get('PubDate', {})
                    year = pub_date.get('Year', '')
                    if not year and 'MedlineDate' in pub_date:
                        # Extract year from MedlineDate if Year is not available
                        match = re.search(r'\d{4}', pub_date['MedlineDate'])
                        year = match.group(0) if match else ''

                    # Extract journal info
                    journal_title = journal.get('Title', '')
                    volume = journal_issue.get('Volume', '')
                    issue = journal_issue.get('Issue', '')

                    # Extract pagination
                    pagination = article.get('Pagination', {})
                    pages = pagination.get('MedlinePgn', '')

                    # Extract DOI
                    doi = ''
                    article_ids = record.get('PubmedData', {}).get('ArticleIdList', [])
                    for article_id in article_ids:
                        if article_id.attributes.get('IdType') == 'doi':
                            doi = str(article_id)
                            break

                    # Create article record
                    article_record = {
                        "PMID": pmid,
                        "Title": title,
                        "Abstract": abstract,
                        "Authors": authors_str,
                        "Year": year,
                        "Journal": journal_title,
                        "Volume": volume,
                        "Issue": issue,
                        "Pages": pages,
                        "DOI": doi
                    }

                    articles.append(article_record)

            return articles

        except Exception as e:
            self.logger.error(f"Error fetching article info: {str(e)}")
            return []