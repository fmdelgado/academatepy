import pandas as pd
from Bio import Entrez
import re
from pydantic import BaseModel
import logging
import json


class LLMResponse(BaseModel):
    content: str


class PubMedQueryGenerator:
    def __init__(self, email, api_key, ollama_llm=None, two_step_search_query=False):
        """
        Initializes the PubMedQueryGenerator.

        Parameters:
        - email (str): Your email address (required by NCBI).
        - api_key (str): Your NCBI API key.
        - ollama_llm: An instance of the LLM for generating and improving queries.
        - two_step_search_query (bool): Whether to use a two-step approach for query generation.
        """
        self.email = email
        self.api_key = api_key
        self.ollama_llm = ollama_llm
        self.two_step_search_query = two_step_search_query
        Entrez.email = self.email
        Entrez.api_key = self.api_key
        self.from_year = None
        self.to_year = None
        self.results = None
        self.last_query = ""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def set_date_filter(self, from_year=None, to_year=None):
        """
        Sets the date filter for the PubMed query.

        Parameters:
        - from_year (int): Start year for the date filter.
        - to_year (int): End year for the date filter.
        """
        self.from_year = from_year
        self.to_year = to_year

    def get_date_filter(self):
        """
        Constructs the date filter string for PubMed query.

        Returns:
        - date_filter (str): The date filter string.
        """
        from_year = self.from_year if self.from_year else "1000"
        to_year = self.to_year if self.to_year else "3000"
        date_filter = f'("{from_year}/01/01"[Date - Publication] : "{to_year}/12/31"[Date - Publication])'
        return date_filter

    def generate_query(self, topic, known_pmids=None):
        """
        Generates a PubMed query for the given topic using the LLM.

        Parameters:
        - topic (str): The research topic.
        - known_pmids (list): Optional list of known PubMed IDs to tailor the query.

        Returns:
        - query (str): The generated PubMed query.
        """
        if not self.ollama_llm:
            raise ValueError("An LLM instance is required to generate the query.")

        # Generate the query using the LLM
        if self.two_step_search_query:
            query = self.generate_query_with_llm_given_keywords(topic)
        else:
            query = self.generate_query_with_llm(topic)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

        self.last_query = query  # Store the last generated query
        return query

    def generate_query_with_llm(self, topic):
        """
        Generates a PubMed query for the topic using the LLM.
        """
        prompt = f"""You are an expert in creating PubMed search queries.

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
        response = self.ollama_llm.invoke(prompt)
        content = self.extract_content(response)
        query = self.extract_query_from_markers(content)
        query = self.clean_query(query)
        return query
    
    def generate_query_with_llm_given_keywords(self, keywords):
        """
        Generates a PubMed query for the topic using the LLM.
        """
        prompt = f"""You are an expert in creating PubMed search queries.

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
        response = self.ollama_llm.invoke(prompt)
        content = self.extract_content(response)
        query = self.extract_query_from_markers(content)
        query = self.clean_query(query)
        return query
    
    def extract_query_from_markers(self, content):
        """
        Extracts the query between the markers <START_QUERY> and <END_QUERY>.
        """
        match = re.search(r'<START_QUERY>(.*?)<END_QUERY>', content, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1).strip()
        else:
            self.logger.error("Markers not found in LLM output. Unable to extract query.")
            raise ValueError("Markers not found in LLM output.")
        return query

    def extract_query_from_response(self, response):
        """
        Extracts the 'query' from the LLM's JSON response.
        """
        if isinstance(response, dict):
            data = response
        else:
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON from LLM response.")
                return ""

        if 'query' in data:
            query = data['query']
            return query
        else:
            self.logger.error("No 'query' key found in LLM response.")
            return ""

    def clean_query(self, query):
        """
        Cleans and formats the generated query.
        """
        if not query:
            return ""

        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)

        # Remove leading/trailing whitespace and unwanted characters
        query = query.strip("`'\" \n\t")

        # Remove any accidental markers or extra text
        query = re.sub(r'<.*?>', '', query)

        # Replace multiple spaces with a single space
        query = re.sub(r'\s+', ' ', query).strip()

        # Ensure proper parentheses
        if not query.startswith("("):
            query = "(" + query
        if not query.endswith(")"):
            query = query + ")"

        return query

    def extract_content(self, response):
        """
        Extracts the text content from the LLM response.

        Parameters:
        - response: The response object from the LLM.

        Returns:
        - content (str): The extracted text content.
        """
        try:
            if isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            else:
                self.logger.warning("Unexpected response format. Attempting to convert to string.")
                return str(response)
        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            return ""

    def execute_query(self, query, max_results=10000):
        """
        Executes the PubMed query and retrieves the results.

        Parameters:
        - query (str): The PubMed query to execute.
        - max_results (int): Maximum number of results to retrieve.
        """
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            pmids = record["IdList"]
            if not pmids:
                self.results = pd.DataFrame()
                return
            articles = self.fetch_article_info(pmids)
            self.results = pd.DataFrame(articles)
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self):
        """
        Returns the query results as a pandas DataFrame.

        Returns:
        - DataFrame containing the results.
        """
        return self.results

    def fetch_article_info(self, pmids):
        """
        Fetches article information for given PMIDs.

        Parameters:
        - pmids (list): List of PubMed IDs.

        Returns:
        - articles (list): List of dictionaries containing article info.
        """
        try:
            batch_size = 200  # NCBI recommends fetching in batches of 200 or less
            articles = []
            for start in range(0, len(pmids), batch_size):
                end = min(start + batch_size, len(pmids))
                batch_pmids = pmids[start:end]
                handle = Entrez.efetch(db="pubmed", id=",".join(batch_pmids), rettype="medline", retmode="xml")
                records = Entrez.read(handle)
                for record in records['PubmedArticle']:
                    medline_citation = record.get('MedlineCitation', {})
                    article = medline_citation.get('Article', {})
                    journal = article.get('Journal', {})
                    journal_issue = journal.get('JournalIssue', {})
                    pagination = article.get('Pagination', {})
                    pub_date = journal_issue.get('PubDate', {})
                    authors_list = article.get('AuthorList', [])
                    article_ids = record.get('PubmedData', {}).get('ArticleIdList', [])
                    mesh_headings = medline_citation.get('MeshHeadingList', [])

                    # PMID
                    pmid = medline_citation.get('PMID', '')

                    # Title
                    title = article.get('ArticleTitle', '')

                    # Abstract
                    abstract_list = article.get('Abstract', {}).get('AbstractText', [])
                    abstract = ' '.join(abstract_list)

                    # Authors
                    authors = []
                    for author in authors_list:
                        last_name = author.get('LastName', '')
                        fore_name = author.get('ForeName', '')
                        full_name = ' '.join([fore_name, last_name]).strip()
                        if full_name:
                            authors.append(full_name)
                    authors_str = ', '.join(authors)

                    # Year
                    year = pub_date.get('Year', '')
                    if not year and 'MedlineDate' in pub_date:
                        year = pub_date['MedlineDate'].split(' ')[0]

                    # Journal
                    journal_title = journal.get('Title', '')

                    # Volume
                    volume = journal_issue.get('Volume', '')

                    # Issue
                    issue = journal_issue.get('Issue', '')

                    # Pages
                    pages = pagination.get('MedlinePgn', '')

                    # DOI
                    doi = ''
                    # First, check in the ArticleIdList in PubmedData
                    article_id_list = record.get('PubmedData', {}).get('ArticleIdList', [])
                    for article_id in article_id_list:
                        if article_id.attributes.get('IdType') == 'doi':
                            doi = str(article_id)
                            break
                    # If not found, check in ArticleIdList in MedlineCitation
                    if not doi:
                        article_id_list = medline_citation.get('ArticleIdList', [])
                        for article_id in article_id_list:
                            if article_id.attributes.get('IdType') == 'doi':
                                doi = str(article_id)
                                break
                    # If still not found, check in ELocationID
                    if not doi:
                        elocation_ids = article.get('ELocationID', [])
                        if not isinstance(elocation_ids, list):
                            elocation_ids = [elocation_ids]
                        for eloc in elocation_ids:
                            if eloc.attributes.get('EIdType') == 'doi':
                                doi = str(eloc)
                                break

                    # ISSN
                    issn = journal.get('ISSN', '')

                    # Keywords (if available)
                    keywords_list = medline_citation.get('KeywordList', [])
                    keywords = []
                    for keyword_group in keywords_list:
                        for keyword in keyword_group:
                            keywords.append(str(keyword))
                    keywords_str = ', '.join(keywords)

                    # Document Type (Publication Type)
                    publication_types = article.get('PublicationTypeList', [])
                    doc_type = ', '.join([pt for pt in publication_types])

                    articles.append({
                        "PMID": pmid,
                        "Title": title,
                        "Abstract": abstract,
                        "Authors": authors_str,
                        "Year": year,
                        "Journal": journal_title,
                        "Volume": volume,
                        "Issue": issue,
                        "Pages": pages,
                        "DOI": doi,
                        "ISSN": issn,
                        "Keywords": keywords_str,
                        "Document Type": doc_type
                    })
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching article info: {str(e)}")
            return []



