
import logging
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_community.llms.llamafile import Llamafile
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from os.path import expanduser
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain_community.docstore.document import Document




# 1. Load and Prepare Your CSV File
loader = CSVLoader(file_path='Operation_Counts.csv')
documents = loader.load()



     
# 2. Split documents for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = []
with open("Operation_Counts.csv", "r") as f:
    for line in f:
        for chunk in text_splitter.split_text(line):
            documents.append(Document(page_content=chunk))
print(documents)
#docs = text_splitter.split_documents(documents)
#print(docs)



# 3. Sentence Transformers Embedding Setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the database
persist_directory = 'db'  # Directory to store the database
vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)


# 5. Set Up Your LLM Model
llm = Ollama()

# Retrieval QA Chain with Prompt
prompt_template = """You are reviewing airport flight data from a CSV file. 
The columns in the file are:
*operation: type of air traffic control operation. Operation types are overflight, take off, touch and go, full stop and low approach.
*count: number of times the corresponding operation occured.

Use the following pieces of context to answer the question at the end. Read every column and row. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
{context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


# 6. Create the RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# 7. Interact and Get Answers
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        #Clear Chroma DB
        vectordb.delete_collection() 
        vectordb.persist()  # This makes the changes permanent by saving to disk
        print("Chroma DB cleared. Exiting...")
        break
    answer = qa.run(query)
    print(answer)
