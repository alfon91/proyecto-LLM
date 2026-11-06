import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic


load_dotenv()

def consultar_rag(pregunta):
    persist_directory ='db'

    if not os.path.exists(persist_directory):
        print('Not found')
        return

    print('Searching')

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    documentos_relevantes = vectorstore.similarity_search(pregunta,k=2)

    contexto = '\n\n'.join([doc.page_content for doc in documentos_relevantes])

    print(f'fragmentos encontrados: {len(documentos_relevantes)}')

    llm = ChatAnthropic(
        model = 'claude-sonnet-5-5'
    )


    prompt_completo = f"""eres un asistente util y preciso.Responde a la pregunta del usuario basándote exclusivamente en el siguiente contexto recuperado:
    
    Contexto: {contexto}
    
    Pregunta: {pregunta}
    
    Respuesta:"""

  

    print(f'consultando a Claude:{pregunta}\n')
    response = llm.invoke(prompt_completo)

    print('Respues de claude:')
    print('-'*40)
    print(response.content)
    print('-'*40)

if __name__ == '__main__':
    print('Asistente RAg local con calude iniciado')
    print('Escribe salir o exit cuando quieras terminar.\n')

    while True:
        pregunta_usuario= input('¿Que quieres saber de tus documentos?')

        if pregunta_usuario.lower() in ['salir','exit','quit']:
            print('Adios')
            break
        if not pregunta_usuario.strip():
            continue

        consultar_rag(pregunta_usuario)