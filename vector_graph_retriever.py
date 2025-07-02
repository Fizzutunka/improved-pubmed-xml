import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize LLM
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say "I don't know." and explain why you couldn't answer.
In your answer qoute the source of the information, e.g. "According to article with PMID 12345678...".

{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

# Define state for app
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define custom retrieval Cypher query for PubMed
retrieval_query = """
MATCH (a:Article) WHERE id(a) = id(node) AND a.abstract IS NOT NULL
OPTIONAL MATCH (auth:Author)-[:WROTE]->(a)
WITH a, score, collect(auth.name) AS authors
RETURN 
    a.abstract AS text,
    score,
    {
        pmid: a.pmid,
        pub_date: a.pub_date,
        title: a.title,
        j_title: a.j_title,
        authors: authors
    } AS metadata
"""

# Create vector store object using index and custom query
abstract_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="vector",  # your actual Neo4j vector index name
    embedding_node_property="abstract_vectors",
    text_node_property="abstract",
    retrieval_query=retrieval_query,
)

# Function to retrieve context
def retrieve(state: State):
    context = abstract_vector.similarity_search(
        state["question"],
        k=6,
    )
    return {"context": context}

# Function to generate answer
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}

# Define workflow
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Example question
question = "Whats is acetylsalicylic acid?"
response = app.invoke({"question": question})
print("\n Answer:", response["answer"])
print("\n Context:", response["context"])
