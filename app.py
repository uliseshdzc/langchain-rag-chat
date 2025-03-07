import gradio
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter


llm_model = "deepseek-r1:1.5b"
embeddings_model = "bge-m3:latest"
persist_vector_directory = "./vector-db"

model = ChatOllama(model=llm_model)
embeddings = OllamaEmbeddings(model=embeddings_model)

if os.path.exists(persist_vector_directory):
    vector_store = Chroma(persist_directory=persist_vector_directory, embedding_function=embeddings)
else:
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
        embedding=embeddings,
        persist_directory=persist_vector_directory
    )
    vector_store.persist()

retriever = vector_store.as_retriever()

def process_input(question: str):
    # 3. Before RAG
    before_rag_prompt = ChatPromptTemplate.from_template(question)
    before_rag_chain = before_rag_prompt | model | StrOutputParser()

    # 4. After RAG
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
    return before_rag_chain.invoke(), after_rag_chain.invoke(question)

iface = gradio.Interface(fn=process_input,
                         inputs=[gradio.Textbox(label="Question")],
                         outputs=[gradio.TextArea(label="Without RAG"), gradio.TextArea(label="With RAG")],
                         title="Document Query",
                         description="Enter a question to query the documents.")
iface.launch()