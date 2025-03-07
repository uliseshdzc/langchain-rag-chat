import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

model = ChatOllama(model="deepseek-r1:1.5b")

# 1. Split data into chunks
files_folder = "./knowledge-files"
docs = [PyPDFLoader(os.path.join(files_folder, file)).load() for file in os.listdir(files_folder)]
docs_items = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
docs_splitted = text_splitter.split_documents(docs_items)

# 2. Convert documents to embeddings and store them
vector_store = Chroma.from_documents(
    documents=docs_splitted,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="bge-m3:latest")
)
retriever = vector_store.as_retriever()

# 3. Before RAG
print("Before RAG:\n")
question = "What is ollama"
before_rag_prompt = ChatPromptTemplate.from_template(question)
before_rag_chain = before_rag_prompt | model | StrOutputParser()

print(before_rag_chain.invoke({}))

# 4. After RAG
print("\n\nAfter RAG:\n")
after_rag_prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context}
    Question: {question}
    """
)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model
    | StrOutputParser()
)
print(before_rag_chain.invoke({"question": question}))