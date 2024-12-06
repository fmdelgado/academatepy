import os
import json
import requests
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import logging
from langchain_core.documents import Document
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, llm, embeddings, chromadb, retriever_k=10, search_type='mmr'):
        """
        Initialize the RAGChatbot.

        Args:
            llm: The language model to use for generation and grading.
            embeddings: The embeddings model used for the vector store.
            chromadb: The vector store instance.
            retriever_k (int): Number of documents to retrieve.
            search_type (str): The search strategy for the retriever.
        """
        # Initialize LLM, embeddings, and vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = chromadb
        self.conversation_history = []  # Initialize an empty conversation history
        self.user_memories = {}  # Dictionary to store user memories

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={'k': retriever_k}
        )

        # Initialize prompts and chains
        self._initialize_chains()

    def save_memory(self, user_id, memory_text):
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        self.user_memories[user_id].append(memory_text)
        # Add the memory to the vector store
        doc = Document(
            page_content=memory_text,
            metadata={'user_id': user_id, 'type': 'memory', 'id': str(uuid.uuid4())}
        )
        self.vectorstore.add_documents([doc])

    def retrieve_memories(self, user_id):
        return self.user_memories.get(user_id, [])

    def extract_memory_from_text(self, text):
        # Use the LLM to extract important information
        prompt = f"Extract key information from the following text to remember for future interactions:\n\n{text}\n\nImportant Information:"
        extracted_info = self.llm.invoke(prompt)
        return extracted_info.content.strip()

    def _initialize_chains(self):
        # Retrieval Grader
        retrieval_grader_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["question", "document"],
        )
        self.retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

        # Generation chain
        generate_prompt_template = """You are an assistant that answers questions based on the provided context.

Context:
{context}

Question:
{question}

**It is crucial that you only answer based on the provided context.** Provide a concise answer to the question based solely on the context above. **After every sentence or factual statement, include a citation in the format [Source X].** Use the Source IDs provided in the context and sources list.

**If you cannot answer the question based on the context, respond with:** "I'm sorry, but the documents do not contain information about your question."

For example:
"A musculoskeletal condition affects the muscles, bones, and joints [Source 1]."

Answer:
"""

        generate_prompt = PromptTemplate(
            template=generate_prompt_template,
            input_variables=["context", "question"],
        )
        self.rag_chain = generate_prompt | self.llm | StrOutputParser()

        # Hallucination Grader
        hallucination_grader_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
Here are the facts:
\n ------- \n
{documents} 
\n ------- \n
Here is the answer: {generation}
Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "documents"],
        )
        self.hallucination_grader = hallucination_grader_prompt | self.llm | JsonOutputParser()

        # Answer Grader
        answer_grader_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
Here is the answer:
\n ------- \n
{generation} 
\n ------- \n
Here is the question: {question}
Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "question"],
        )
        self.answer_grader = answer_grader_prompt | self.llm | JsonOutputParser()


    # Utility methods
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def map_citations_to_sources(self, generation, documents):
        import re

        generation_text = generation.strip()

        citation_pattern = r'\[Source\s*(\d+)\]'
        citations = re.findall(citation_pattern, generation_text)
        unique_citations = set(citations)

        source_mapping = {}
        for citation in unique_citations:
            index = int(citation) - 1  # Adjust for zero-based index
            if 0 <= index < len(documents):
                source_mapping[f"Source {citation}"] = documents[index]
            else:
                logger.warning(f"Invalid citation [Source {citation}]")
        return source_mapping

    def retrieve(self, question, user_id=None):
        logger.info("---RETRIEVE---")

        # Retrieve documents
        def filter_fn(doc):
            return doc.metadata.get('user_id') == user_id or doc.metadata.get('type') != 'memory'

        documents = self.retriever.invoke(question, filter=filter_fn)
        return documents

    def grade_documents(self, question, documents):
        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        filtered_docs = []
        for doc in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            grade = score.get("score", "").lower()
            if grade == "yes":
                logger.info(
                    f"---GRADE: DOCUMENT RELEVANT--- (Source: {doc.metadata.get('source', 'N/A')}, Chunk: {doc.metadata.get('chunk_index', 'N/A')})")
                filtered_docs.append(doc)
            else:
                logger.info(
                    f"---GRADE: DOCUMENT NOT RELEVANT--- (Source: {doc.metadata.get('source', 'N/A')}, Chunk: {doc.metadata.get('chunk_index', 'N/A')})")
        return filtered_docs

    def generate(self, full_question, documents):
        logger.info("---GENERATE---")
        if not documents:
            logger.info("---NO DOCUMENTS AVAILABLE IN GENERATE---")
            generation = "I'm sorry, but the documents do not contain information about your question."
            return generation, {}
        # Prepare context with source information and unique identifiers
        context = "\n\n".join(
            [
                f"[Source {i + 1}] {doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )
        # Prepare sources list (if needed)
        sources = "\n".join(
            [
                f"[Source {i + 1}]: {doc.metadata.get('source', 'N/A')} (Chunk {doc.metadata.get('chunk_index', 'N/A')})"
                for i, doc in enumerate(documents)
            ]
        )
        full_context = f"{context}\n\nSources:\n{sources}"

        # RAG generation
        generation = self.rag_chain.invoke({"context": full_context, "question": full_question})
        logger.debug(f"Generated answer: {generation}")

        # Map citations to sources
        source_mapping = self.map_citations_to_sources(generation, documents)

        return generation, source_mapping

    def grade_generation(self, full_question, generation, source_mapping):
        logger.info("---CHECK HALLUCINATIONS---")
        if not source_mapping:
            logger.info("---DECISION: NO VALID CITATIONS FOUND IN GENERATION---")
            return False

        # Prepare documents for grading
        mapped_documents = list(source_mapping.values())
        documents_text = self.format_docs(mapped_documents)

        # Grade for hallucinations
        score = self.hallucination_grader.invoke(
            {"documents": documents_text, "generation": generation}
        )
        grade = score.get("score", "").lower()
        if grade != "yes":
            logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return False

        # Grade for usefulness
        score = self.answer_grader.invoke({"question": full_question, "generation": generation})
        grade = score.get("score", "").lower()
        if grade == "yes":
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            return True
        else:
            logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return False

    def get_answer(self, question, conversation_history=None, user_id=None):
        if conversation_history is None:
            conversation_history = []

        # Append the current question to the conversation history
        conversation_history.append({"role": "user", "content": question})

        # Build the full question using conversation history (for generation)
        full_question = self._build_full_question(conversation_history)

        # Use only the current question for retrieval
        retrieval_question = question

        # Retrieve documents based on the current question
        documents = self.retrieve(retrieval_question)

        # Grade documents
        filtered_docs = self.grade_documents(retrieval_question, documents)

        # Decide to generate or handle no answer
        if not filtered_docs:
            generation = "I'm sorry, but the documents do not contain information about your question."
            # Append the assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": generation})

            # Extract memory from the user's input
            if user_id:
                new_memory = self.extract_memory_from_text(question)
                self.save_memory(user_id, new_memory)
            return generation, None

        # Generate answer using the full question and filtered documents
        generation, source_mapping = self.generate(full_question, filtered_docs)

        # Grade generation
        is_valid = self.grade_generation(full_question, generation, source_mapping)

        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": generation})

        # Save memory (extract from assistant's response)
        if user_id:
            new_memory = self.extract_memory_from_text(generation)
            self.save_memory(user_id, new_memory)

        if is_valid:
            return generation, source_mapping
        else:
            generation = "I'm sorry, but the documents do not contain information about your question."
            return generation, None

    def _build_full_question(self, conversation_history):
        # Combine the conversation history into a single string
        history = ""
        for turn in conversation_history:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                history += f"User: {content}\n"
            elif role == "assistant":
                history += f"Assistant: {content}\n"
        return history.strip()


if __name__ == "__main__":
    # Load environment variables
    dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
    load_dotenv(dotenv_path)

    # Authentication (if needed)
    auth_response = requests.post(
        os.getenv('AUTH_URL'),
        json={'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
    )
    jwt = json.loads(auth_response.text)["token"]
    headers = {"Authorization": "Bearer " + jwt}

    # Initialize LLM
    llm = ChatOllama(
        base_url=os.getenv('OLLAMA_API_URL'),
        model='mistral:7b',
        temperature=0.1,
        num_predict=-1,
        seed=20,
        stop=["Answer:"],
        client_kwargs={'headers': headers}
    )

    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        base_url=os.getenv('OLLAMA_API_URL'),
        model="nomic-embed-text",
        headers=headers
    )

    # Load vectorstore
    persist_directory = "/Users/fernando/Documents/Research/academatepy/test_results/results_llama3.1_8b/embeddings/screening1_embeddings"
    vectorstore = Chroma(
        collection_name="literature",
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Instantiate chatbot
    chatbot = RAGChatbot(
        llm=llm,
        embeddings=embeddings,
        chromadb=vectorstore
    )

    # Initialize conversation history
    conversation_history = []
    user_id = "user123"  # Assign a unique ID for the user

    # First user input
    question = "What is CORKA?"
    answer, source_mapping = chatbot.get_answer(question, conversation_history, user_id)

    print("Answer 1:")
    print(answer)
    if source_mapping:
        print("Final Documents (Supporting Chunks):")
        for source_id, doc in source_mapping.items():
            print(f"{source_id}:")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...\n")
    else:
        print("No supporting documents found.")
    del answer, source_mapping, question

    # Continue the conversation
    question = "Was there a statistically significant difference with that program?"
    answer, source_mapping = chatbot.get_answer(question, conversation_history, user_id)

    print("Answer 2:")
    print(answer)
    if source_mapping:
        print("Final Documents (Supporting Chunks):")
        for source_id, doc in source_mapping.items():
            print(f"{source_id}:")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...\n")
    else:
        print("No supporting documents found.")
    del answer, source_mapping, question

    # First user input
    question = "My name is Alice."
    answer, source_mapping = chatbot.get_answer(question, conversation_history, user_id)
    print("Answer 3:")
    print(answer)
    if source_mapping:
        print("Final Documents (Supporting Chunks):")
        for source_id, doc in source_mapping.items():
            print(f"{source_id}:")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...\n")
    else:
        print("No supporting documents found.")
    del answer, source_mapping, question

    # Second user input
    question = "What's my name?"
    answer, source_mapping = chatbot.get_answer(question, conversation_history, user_id)
    print("Answer 4:")
    print(answer)
    if source_mapping:
        print("Final Documents (Supporting Chunks):")
        for source_id, doc in source_mapping.items():
            print(f"{source_id}:")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...\n")
    else:
        print("No supporting documents found.")
    del answer, source_mapping, question
