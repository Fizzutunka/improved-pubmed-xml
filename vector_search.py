import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Neo4j driver
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# creates embeddings of abstracts and saves as aproperrty in the Article nodes
neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    node_label="Article",
    text_node_properties=["abstract"],
    embedding_node_property="abstract_vectors"
)
print("Abstracts embedded and saved.")

# Create vector index from the abstract embeddings. Useful for faster indexing and searching and similarity searches. Use the idex to query. 
create_index_query = """
CREATE VECTOR INDEX abstract_embedding_index
FOR (a:Article)
ON (a.abstract_vectors)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
}
"""

with driver.session() as session:
    session.run(create_index_query)
    print("Vector index created.")

# Search
result = neo4j_vector.similarity_search("cancer", k=3)
for doc in result:
    print(f"PMID: {doc.metadata.get('pmid')}")
    print(f"Match: {doc.page_content}\n")



