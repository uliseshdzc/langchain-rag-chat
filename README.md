# Document Query with LangChain and Gradio

This project demonstrates how to use the Ollama Chat model in LangChain to query documents stored in PDF files. The application uses Gradio for the user interface and supports both retrieval-augmented generation (RAG) and non-RAG responses. It is based on the [example](https://mer.vin/2024/02/ollama-embedding/) given by Mervin Praison.

## Setup

### Prerequisites

- Python 3.13
- Virtual environment tool (e.g., `venv`)

### Installation

1. Installing Python 3.13

   **Windows:**
   - Download the installer from the official [Python website](https://www.python.org/downloads/).
   - Run the installer and follow the instructions. Make sure to check the box that says "Add Python to PATH".

    **macOS:**
   - You can use Homebrew to install Python 3.13:
     ```sh
     brew install python@3.13
     ```

    **Linux:**
   - Use your package manager to install Python 3.13. For example, on Ubuntu:
     ```sh
     sudo apt update
     sudo apt install python3.13
     ```

2. **Create a virtual environment:**

   ```
   python3.13 -m venv venv
   ```
3. **Activate the virtual environment:**
   - On Windows:
   ```
   .\venv\Scripts\activate
   ```
   - On macOS and Linux:
   ```
   source venv/bin/activate
   ```
4. **Install the required packages:**

   ```
   pip install -r requirements.txt
   ```
5. **Install Ollama and the required models:** Follow the instructions on the Ollama website to install the [Ollama package](https://ollama.com/download) and the required [models](https://ollama.com/search) (**deepseek-r1:1.5b** and **bge-m3:latest**). You can edit the code to use a different embedding model and LLM.

### Usage

1. **Prepare your PDF files:**
Place the PDF files used for the RAG in the `./knowledge-files` directory.

2. **Run the application:**
   ```
   python app.py
   ```
   This will start the Gradio interface.

3. **Accessing the interface:**
   Go to http://127.0.0.1:7860/ to access the interface. You may now enter your question.

### Code Explanation
The main components of the application are:

- **PDF Text Extraction**: Extracts text from PDF files using PyPDFLoader.
- **Text Chunking**: Splits the extracted text into manageable chunks using CharacterTextSplitter.
- **Vector Store**: Converts the text chunks into embeddings and stores them using Chroma.
- **Retrieval-Augmented Generation (RAG)**: Uses the stored embeddings to retrieve relevant context for answering questions.
- **Gradio Interface**: Provides a user-friendly interface for querying the documents.