# import nltk
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()

from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader  # For reading PDF files Need to put in requirements
from docx import Document      # For reading Word files Need to put in requirements


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def read_files_from_directory(directory):
    # Get all the document files in the directory
    extensions = ['*.txt', '*.pdf', '*.docx']  # Add other formats as needed
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)

def read_docx_file(file_path):
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def read_document(file_path):
    if file_path.endswith('.txt'):
        return read_text_file(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf_file(file_path)
    elif file_path.endswith('.docx'):
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def create_embeddings(file_path):
    content = read_document(file_path)

    if not content.strip():  # Check if content is empty
        print(f"No content to process for {file_path}. Skipping.")
        return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_text(content)

    # all_splits = text_splitter.split_documents(content)
    return all_splits

# load the pdf and split it into chunks
# loader = OnlinePDFLoader("https://cpb-us-w2.wpmucdn.com/sites.udel.edu/dist/a/855/files/2020/08/Rebalancing-Strategies.pdf")
# data = loader.load()

def store_in_chroma(all_splits):
    with SuppressStdout():
        vectorstore = Chroma.from_texts(texts=all_splits, embedding=GPT4AllEmbeddings())
        # vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    return vectorstore

def start_chat(vectorstore):
    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Prompt
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        llm = Ollama(model="gemma2:2b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        qa_chain({"query": query})


def main(folder_path):
    files = read_files_from_directory(folder_path)
    splits = list()
    for file_path in files:
        print(f'Processing {file_path}...')
        try:
            splits = splits + create_embeddings(file_path)
            print(f'Successfully stored embeddings for {file_path}.')
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
    vector_store = store_in_chroma(splits)
    start_chat(vector_store)

if __name__ == "__main__":
    folder_path = "Folder_pathI"  # Change this to your folder path
    main(folder_path)
