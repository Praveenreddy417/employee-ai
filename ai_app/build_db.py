# import os
# from dotenv import load_dotenv
# load_dotenv()

# # from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# openai_api_key = os.getenv("OPENAI_API_KEY")

# # loader = PyPDFLoader("data/employee_data.pdf")

# # import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# pdf_path = os.path.join(BASE_DIR, "data", "employee_data.pdf")

# loader = PyPDFLoader(pdf_path)
# documents = loader.load()
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# docs = splitter.split_documents(documents)

# # embeddings = OpenAIEmbeddings(
# #     model="text-embedding-3-small",
# #     api_key=openai_api_key
# # )

# # vectorstore = Chroma.from_documents(
# #     docs,
# #     embeddings,
# #     persist_directory="chroma_db"
# # )

# # vectorstore.persist()

# # print("✅ Employee DB created successfully")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     persist_directory="chroma_db"
# )

# print("Vector DB created successfully")


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "data", "employee_data.pdf")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# ✅ Use OpenAI embeddings (no torch)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Vector DB created successfully")
