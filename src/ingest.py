import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def ingestar_documentos():
    data_dir='data'
    persist_directory = 'db'

    if not os.path.exists(data_dir):
        print('Not found data')
        return

    print('Cargando documentos')

    documents = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        if filename.lower().endswith('.txt'):
            try:
                loader= TextLoader(file_path,encoding='utf-8')
                documents.extend(loader.load())
                print(f'Cargado TXT: {filename}')
            except Exception as e:
                print(f' No se pudo leer el TXT {filename}:{e}')
        if filename.lower().endswith('.pdf'):
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f'Cargado PDF: {filename}')
            except Exception as e:
                print(f' No se pudo leer el TXT {filename}:{e}')
    
    if not documents:
        print('not found txt or pdf')
        return

    print(f'Se han cargado {len(documents)} documentos')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)
    print(f'Documentos divididos en {len(chunks)} fragmentos')

    print('Generando embeddings numericos con hugging face')

    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    print('Guardando/actualizando la base de datos vectorial en db')

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print('Ingesta completada')
   
if __name__ == '__main__':
    ingestar_documentos()
