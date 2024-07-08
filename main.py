
import logging
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveJsonSplitter
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




# 1. Load and Prepare Your Knowledge Base
loader = CSVLoader(file_path='Operation_Counts.csv')
documents = loader.load()



     
# 2. Split documents for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(docs)



# 3. Sentence Transformers Embedding Setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the database
persist_directory = 'db'  # Directory to store the database
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)


# 5. Set Up Your LLM Model
llm = Ollama()

# 6. Create the RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever()
)

# 7. Interact and Get Answers
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print(answer)
