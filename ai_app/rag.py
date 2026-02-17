# import os
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_groq import ChatGroq
# # from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# vectorstore = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embeddings
# )


# groq_api_key = os.getenv("GROQ_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # embeddings = OpenAIEmbeddings(
# #     model="text-embedding-3-small",
# #     api_key=openai_api_key
# # )

# # vector_store = Chroma(
# #     persist_directory="chroma_db",
# #     embedding_function=embeddings
# # )

# retriever = vectorstore.as_retriever()

# # llm = ChatGroq(
# #     groq_api_key=groq_api_key,
# #     model_name="llama3-8b-8192",
# #     temperature=0.3
# # )
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0
# )


# SYSTEM_PROMPT = """
# You are an Employee Data Assistant.

# Answer ONLY from given context.
# If not found, say: "Employee data not available."

# Context:
# {context}
# """

# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", "{question}")
# ])

# rag_chain = (
#     {"context": retriever, "question": lambda x: x}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# def ask_question(query):
#     return rag_chain.invoke(query)



import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ Use OpenAI embeddings (no torch)
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

# ✅ Groq LLM (lightweight)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

SYSTEM_PROMPT = """
You are an Employee Data Assistant.

Answer ONLY from given context.
If not found, say: "Employee data not available."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(query):
    return rag_chain.invoke(query)
