import logging

class KeywordsGenerator:
    def __init__(self, llm=None):
        """
        Initializes the KeywordsGenerator.

        Parameters:
        - llm: An instance of the LLM for generating and improving keywords.
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def generate_keywords(self, topic):
        """
        Generates keywords for the given topic.
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the keywords.")
        
        return self.generate_keyword_query(topic)


    def generate_keyword_query(self, topic: str) -> str:
        """
        Generates keywords using the LLM.
        """
        
        prompt = f"""You are an expert in creating keywords for search queries in public databases like PubMed, Semantic Scholar, or Scopus.

        Generate a comprehensive set of keywords describing for the following research topic to improve search results in these databases:

        Topic:
        {topic}

        **Instructions:**
        1. Identify the main concepts in the topic.
        2. For each concept, include synonyms and related terms.
        3. Use the exact template below to format your response. Do not deviate from this structure.
        4. Limit yourself to 3-5 main keywords.
        5. Include 3-5 synonyms for each main keyword.
        6. Do not include any additional information or context.

        **Example of a correct response:**
        <START_OUTPUT>
        Keyword 1: [Main Keyword]
        Synonyms: [Synonym 1], [Synonym 2], [Synonym 3]
        Keyword 2: [Main Keyword]
        Synonyms: [Synonym 1], [Synonym 2], [Synonym 3]
        <END_OUTPUT>

        **Please provide your response below:**
        """
        response = self.llm.invoke(prompt)
        keywords = self.extract_keywords(response)
        return keywords

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
                return response.strip()
            elif hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, dict) and 'content' in response:
                return response['content'].strip()
            else:
                self.logger.warning("Unexpected response format. Attempting to convert to string.")
                return str(response).strip()
        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            return ""
        
    def extract_keywords(self, response):
        """
        Extracts the keywords from the LLM response.

        Parameters:
        - response: The response object from the LLM.

        Returns:
        - keywords (list): The extracted keywords.
        """
        content = self.extract_content(response)
        return content
