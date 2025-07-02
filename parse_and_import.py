import xml.etree.ElementTree as ET
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

#neo4j credentials
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)
# test neo4j connection
def test_connection(tx):
    result = tx.run("RETURN 'Connection successful' AS message")
    for record in result:
        print(record["message"])

with driver.session() as session:
    session.execute_write(test_connection)

# Parsing articles (xml)
def parse_pubmed_article(article_elem):
    def get_text(elem, path):
        node = elem.find(path)
        return node.text.strip() if node is not None and node.text else None

    pmid = get_text(article_elem, ".//PMID")
    title = get_text(article_elem, ".//ArticleTitle")
    abstract = get_text(article_elem, ".//AbstractText")
    language = get_text(article_elem, ".//Language")

    journal = article_elem.find(".//Journal")
    j_issn = get_text(journal, "ISSN")
    if not j_issn or j_issn.strip() == "":
            return None  # Skip this article entirely from the neo4j import
    j_title = get_text(journal, "Title") if journal is not None else None
    j_volume = get_text(journal, "JournalIssue/Volume") if journal is not None else None
    j_issue = get_text(journal, "JournalIssue/Issue") if journal is not None else None

    pub_date_elem = journal.find("JournalIssue/PubDate") if journal is not None else None
    pub_date = None
    if pub_date_elem is not None:
        year = get_text(pub_date_elem, "JournalIssue/Year")
        month = get_text(pub_date_elem, "JournalIssue/Month") or "00"
        day = get_text(pub_date_elem, "JournalIssue/Day") or "00"
        try:
            pub_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            pass

    authors = []
    for author in article_elem.findall(".//Author"):
        fore = get_text(author, "ForeName")
        last = get_text(author, "LastName")
        if last or fore:
            name = f"{fore or ''}. {last or ''}".strip()
            authors.append(name)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "language": language,
        "pub_date": pub_date,
        "j_issn": j_issn,
        "j_title": j_title,
        "j_volume": j_volume,
        "j_issue": j_issue,
        "authors": authors
    }

# Load and parse the XML file
tree = ET.parse("/Users/louisgarrett/Desktop/THM_Project/XML_pubmed_parser_and_embedd/pubmed25n0040.xml")
root = tree.getroot()

articles = [a for a in (parse_pubmed_article(elem) for elem in root.findall(".//PubmedArticle")) if a]


# Clear existing data in Neo4j for a fresh graph
def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

with driver.session() as session:
    session.execute_write(clear_database)

# Insert data into Neo4j
def insert_article(tx, article):
    tx.run("""
        MERGE (a:Article {pmid: $pmid})
        SET a.title = $title,
            a.abstract = $abstract,
            a.language = $language,
            a.pub_date = $pub_date,
            a.j_issn = $j_issn,
            a.j_title = $j_title,
            a.j_volume = $j_volume,
            a.j_issue = $j_issue

        MERGE (j:Journal {issn: $j_issn})
        SET j.title = $j_title

        MERGE (a)-[:FOUND_IN]->(j)

        WITH a
        UNWIND $authors AS name
            MERGE (au:Author {name: name})
            MERGE (au)-[:WROTE]->(a)
    """, article)

with driver.session() as session:
    for article in articles:
        session.execute_write(insert_article, article)

print(f"Import to NEO4j completed.")
driver.close()
