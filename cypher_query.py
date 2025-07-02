import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize the LLM
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Try to quote the source of the information where neccesary, e.g. "According to article with PMID 12345678...".
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[dict]
    answer: str

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# provide a schema for the cypher query and examples
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Exclude NULL values when finding the highest value of a property.

Schema:
{schema}
Examples:
1. Question: Get authors of an article?
   Cypher:MATCH (a:Article)<-[:WROTE]-(auth:Author) WHERE a.pmid = "your_pmid_here" RETURN collect(auth.name) AS authors
2. Question: Get PMID of an article?
   Cypher: MATCH (a:Article) WHERE a.title = "Your Article Title Here" RETURN a.pmid AS pmid

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)
from langchain_neo4j import GraphCypherQAChain

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    allow_dangerous_requests=True,
    return_direct=True, # this teturns the cypher query, not the answer 
)

# Define functions for each step in the application

# Retrieve context, i.e. passes the cypher querey to the graph and retrieves the context
def retrieve(state: State):
    context = cypher_qa.invoke(
        {"query": state["question"]}
    )
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = model.invoke(messages)
    return {"answer": response.content}

# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Run the application
question = "Whats is acetylsalicylic acid?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])
print("Context:", response["context"])


