# pubmed_XML_parser
Creating a neo4j database and XML parser in python for PubMED abstracts. 


Create an .env file with the following:

OPENAI_API_KEY="..."

NEO4J_URI="..." (local host usually: "bolt://localhost:7687")

NEO4J_USERNAME="..."

NEO4J_PASSWORD="..."

# The objective

Parse_and_import will read an XML file such as the one from the https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ 

A neo4j graphDB must be an existiing intstance before running the file. The file imports Article and Author and Journal Nodes (see DB schema below). 

Uses google_api_llm. 

Loads, Embedds, creates vector index and quieries the abstracts of the PubMed articles in natural language. 





# NEO4J Schema DB

Article-[:FOUND_IN}->Journal

Author -[:WROTE]-> Article 

Article: 
title
pmid
pub_date
language
journal volume
Journal_issue
journal_issn

Journal:
issn
title

Author:
name [format: F. Last]


