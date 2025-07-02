import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Initialize the LLM
from langchain_openai import OpenAIEmbeddings
#model = init_chat_model("gemini-1.5-flash", model_provider="google_genai")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context just say "I don't know." and why you coun't answer.


{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


from langchain_neo4j import Neo4jGraph
# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the embedding model
#from langchain_openai import OpenAIEmbeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create Vector
from langchain_neo4j import Neo4jVector
abstract_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="vector", # The name of the index to use
    embedding_node_property="abstract_vectors",# The property containing the embedding vector
    text_node_property="abstract",# The property containing the text to embed
)
# Define functions for each step in the application

# Retrieve context from the graph
def retrieve(state: State):
    # Use the vector to find relevant documents
    context = abstract_vector.similarity_search(
        state["question"], 
        k=4,  # Number of documents to retrieve
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




# Answer Provided from vector retrieval: 
# Answer: I don't know.  The provided text mentions aspirin and salicylates, but doesn't define acetylsalicylic acid.  
# While aspirin is a common name for acetylsalicylic acid, the provided text doesn't explicitly state this.


# Answer provided from Vector_graph_retrival.py 
#  Answer: I don't know.  While several documents mention salicylates 
# (e.g.,  "Analysis of the results shows that salicylates were responsible for some of this difference...",  
# from the article with PMID 1198013), and one discusses aspirin intolerance (PMID 1199905), 
# none of the provided text explicitly defines acetylsalicylic acid.


# Answer provided from cypher_query.py
# Answer: Based on the provided text, acetylsalicylic acid (ASA) is a drug that inhibits platelet 
# aggregation and other platelet functions.  
# The abstract from the study "Influence of intravenously administered acetylsalicylic acid on platelet functions. 
# A pharmacodynamic and pharmacokinetic study." states that  "A significant inhibition of platelet aggregation as 
# well as PF 3 and PF 4 availabilities could be demonstrated 2 min after injection."  
# Additionally, another abstract mentions its use in preventing thrombosis.  
# The exact nature of acetylsalicylic acid beyond its effects on platelets is not described in these abstracts.