from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import json, requests
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## Inititalize LLMs ---------------
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

auth_response = requests.post(os.getenv('AUTH_URL'), json={'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')})
jwt = json.loads(auth_response.text)["token"]
headers = {"Authorization": "Bearer " + jwt}

model_name = 'mistral:7b'
# try:
llm = ChatOllama(
    base_url=os.getenv('OLLAMA_API_URL'),
    model=model_name,
    temperature=0.1,
    num_predict=-1,
    # Remove format="json" to get plain text output
    # format="json",
    seed=20,
    stop=["Answer:"],
    client_kwargs={'headers': headers}
)

ollama_embeddings = OllamaEmbeddings(base_url=os.getenv('OLLAMA_API_URL'), model="nomic-embed-text", headers=headers)

## CHECKING  the PDFs,  splitting and embedding them---------------
persist_directory = "/Users/fernando/Documents/Research/academatepy/test_results/results_llama3.1_8b/embeddings/screening1_embeddings"  # Update this path accordingly
collection_name = "vectorDB_screening2"  # Update if your collection name is different

# Load the existing Chroma vectorstore
vectorstore = Chroma(
                collection_name="literature",
                persist_directory=persist_directory,
                embedding_function=ollama_embeddings
            )
# Get all documents from the vectorstore
all_docs = vectorstore.similarity_search("", k=1000)
print(f"Number of documents in vectorstore: {len(all_docs)}")
retriever = vectorstore.as_retriever(search_type='mmr',  # Use 'mmr' as per your preference
                            search_kwargs={'k': 10}  # Adjust k as needed
                        )

## LLM connection
### Retrieval Grader

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
# question = "Hyperaldosteronism"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


### Generate

#
# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Adjust the prompt to include citation instructions and example
prompt_template = """You are an assistant that answers questions based on the provided context.

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



prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

# Rebuild the chain with the new prompt
rag_chain = prompt | llm | StrOutputParser()
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

def map_citations_to_sources(generation, documents):
    import re

    # Extract the actual answer if the assistant outputs JSON
    generation_text = generation.strip()  # Ensure there are no leading/trailing whitespaces

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



### Hallucination Grader

# Prompt
prompt = PromptTemplate(
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

hallucination_grader = prompt | llm | JsonOutputParser()
# hall_grade = hallucination_grader.invoke({"documents": docs, "generation": generation})
# print(hall_grade)


### Answer Grader

# Prompt
prompt = PromptTemplate(
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

answer_grader = prompt | llm | JsonOutputParser()
# answer_grade = answer_grader.invoke({"question": question, "generation": generation})
# print(answer_grade)


### Question Re-writer
# Prompt
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
# question_rewritten = question_rewriter.invoke({"question": question})
# print(question_rewritten)



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def retrieve(state):
    logger.info("---RETRIEVE---")
    question = state["question"]

    # Retrieve documents
    documents = retriever.invoke(question, k=1000)

    return {"documents": documents, "question": question}


def generate(state):
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        # No documents to generate an answer from
        logger.info("---NO DOCUMENTS AVAILABLE IN GENERATE---")
        # Generate a response indicating the answer is not found
        generation = "I'm sorry, but the documents do not contain information about your question."
        state["generation"] = generation
        return state

    # Prepare context with source information and unique identifiers
    context = "\n\n".join(
        [
            f"[Source {i+1}] {doc.page_content}"
            for i, doc in enumerate(documents)
        ]
    )

    # Prepare a mapping of Source IDs to metadata
    sources = "\n".join(
        [
            f"[Source {i+1}]: {doc.metadata.get('source', 'N/A')} (Chunk {doc.metadata.get('chunk_index', 'N/A')})"
            for i, doc in enumerate(documents)
        ]
    )

    full_context = f"{context}\n\nSources:\n{sources}"

    # RAG generation
    generation = rag_chain.invoke({"context": full_context, "question": question})

    logger.debug(f"Generated answer: {generation}")

    # Update state with the generation
    state["generation"] = generation

    return state


def grade_documents(state):
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            logger.info(f"---GRADE: DOCUMENT RELEVANT--- (Source: {doc.metadata.get('source', 'N/A')}, Chunk: {doc.metadata.get('chunk_index', 'N/A')})")
            filtered_docs.append(doc)
        else:
            logger.info(f"---GRADE: DOCUMENT NOT RELEVANT--- (Source: {doc.metadata.get('source', 'N/A')}, Chunk: {doc.metadata.get('chunk_index', 'N/A')})")
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

### Edges

def decide_to_generate(state):
    logger.info("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # No documents to generate an answer from
        logger.info("---DECISION: NO DOCUMENTS AVAILABLE, HANDLE NO ANSWER---")
        return "handle_no_answer"
    else:
        # We have relevant documents, proceed to generate
        logger.info("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    logger.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    source_mapping = map_citations_to_sources(generation, documents)

    # Now you can retrieve the original fragments
    for source_id, doc in source_mapping.items():
        logger.info(f"{source_id}: {doc.page_content[:100]}...")  # Log the first 100 characters

    # Store the source mapping in the state
    state["source_mapping"] = source_mapping

    # Prepare the documents for grading by converting mapped documents back into a list
    mapped_documents = list(source_mapping.values())

    # Use the mapped_documents for grading
    score = hallucination_grader.invoke(
        {"documents": format_docs(mapped_documents), "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        logger.info("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            # Update state with mapped documents
            state["documents"] = mapped_documents
            return "useful"
        else:
            logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "no_answer"
    else:
        logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "no_answer"



def handle_no_answer(state):
    logger.info("---HANDLE NO ANSWER---")
    # Generate a response indicating the answer is not found
    generation = "I'm sorry, but the documents do not contain information about your question."
    state["generation"] = generation
    return state


# Adjust the workflow

# Initialize the workflow
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
# Do not add 'decide_to_generate' as a node
workflow.add_node("generate", generate)
workflow.add_node("handle_no_answer", handle_no_answer)

# Build the graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Use 'decide_to_generate' as a condition function without adding it as a node
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "handle_no_answer": "handle_no_answer",
        "generate": "generate",
    },
)
workflow.add_edge("handle_no_answer", END)

# Similarly, use 'grade_generation_v_documents_and_question' as a condition function
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "no_answer": "handle_no_answer",
    },
)

# Compile the workflow
app = workflow.compile()


from pprint import pprint
# Run

# Run
inputs = {"question": "What can you say about a home-based rehabilitation program (CORKA)?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
    pprint("\n---\n")

print("Final Answer:")
pprint(value["generation"])
print("Final Question:")
pprint(value["question"])
print("Final Documents (Supporting Chunks):")
source_mapping = value.get("source_mapping", {})
if source_mapping:
    for source_id, doc in source_mapping.items():
        print(f"{source_id}:")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
        print(f"Content: {doc.page_content}\n")
else:
    print("No supporting documents found.")

