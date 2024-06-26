
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_community.llms.llamafile import Llamafile
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import LlamaCpp
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer



# 1. Load and Prepare Your Knowledge Base
loader = TextLoader('your_knowledge_base.txt')
documents = loader.load()

# Split documents for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)


# 2. Create a Vector Database (Chroma), and Embedding LLM 
persist_directory = 'db'  # Directory to store your database
#embedding = OpenAIEmbeddings()  
# Replace OpenAI embeddings with Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

# 3. Set Up Your LLM Model (LlamaCpp) or Llamafile
#llm = LlamaCpp(model_path="./models/llama-2-7b-chat.Q4_0.gguf")  # Update model_path as needed
llm= Llamafile()

# 4. Create the RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever()
)

# 5. Interact and Get Answers
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print(answer)