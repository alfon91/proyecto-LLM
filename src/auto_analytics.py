import os
import glob
import json
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()
def extraer_datos_documentos():
    data_dir='data'

    if not os.path.exists(data_dir):
        print('Not found')
        return None

    print(f'Leyendo documentos de la carpeta {data_dir}....')

    documentos_procesados = []

    for archivo in os.listdir(data_dir):
        ruta_archivo= os.path.join(data_dir,archivo)
        texto_completo=''

        if archivo.lower().endswith('.pdf'):
            loader = PyPDFLoader(ruta_archivo)
            pages = loader.load()
            texto_completo= '\n'.join([p.page_content for p in pages])
        elif archivo.lower().endswith('.txt'):
            with open(ruta_archivo,'r',encoding='utf-8') as f:
                texto_completo=f.read()
        else:
            continue

        print(f'Analizando y extrayendo datos de: {archivo}')

        llm_extractor = ChatAnthropic(model='claude-sonnet-5-5')

        prompt_extraccion = f"""Eres un asistente contable experto. Analiza el texto de la siguiente factura y extrae los datos clave en formato JSON estricto (una lista de objetos con estas claves exactas usando guiones bajos: id_factura, concepto, fecha, empresa_expedidora, cliente, base_imponible, iva, importe_iva, total).

CRITERIOS IMPORTANTES PARA NO CONFUNDIR EMPRESAS:
- empresa_expedidora: Es la empresa que EMITE la factura (el proveedor o emisor, el que cobra). Suele estar arriba del todo, en el membrete o en el logotipo.
- cliente: Es la empresa o persona que RECIBE la factura (el comprador, al que se le factura). Siempre es Alfonso Huertas Jimenez.Suele aparecer junto a etiquetas como "Cliente:", "Facturar a:" o "Datos del cliente:".

Si algún dato no aparece, pon null o 0.

Texto del documento:
{texto_completo}

Devuelve ÚNICAMENTE el JSON válido, sin texto adicional ni bloques de markdown si es posible, o solo el bloque JSON:"""

        respuesta = llm_extractor.invoke(prompt_extraccion)

        try:
            contenido = respuesta.content.strip()
            if "```json" in contenido:
                contenido = contenido.split("```json")[1].split("```")[0].strip()
            elif"```" in contenido:
                contenido = contenido.split("```")[1].split("```")[0].strip()

            datos_json = json.loads(contenido)
            if isinstance(datos_json,dict):
                datos_json = [datos_json]

            documentos_procesados.extend(datos_json)
            print(f'Datos extraidos correctamente de:{archivo}')
        except Exception as e:
            print(f' No se pudo extraer estructura tabular de {archivo}:{e}')

    if not documentos_procesados:
        print('No se puedieron extraer datos estructurados de ningun documento')
        return None

    df = pd.DataFrame(documentos_procesados)
    df.to_csv('data/dataframe_extraido.csv', index = False, encoding='utf-8')
    
    return df

def iniciar_sistema():
    df = extraer_datos_documentos()
    if df is None or df.empty:
        return

    print('\n DAtaframe consolidado con tus documentos reales:')
    print(df)
    print('\n'+'='*50 + '\n')

    llm = ChatAnthropic(model='claude-sonnet-5-5')

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code = True
    )

    print('Asistente analitico inteligente sobre tus documentos iniciado')
    print('Pregunta lo que quieras')
    print("escribe 'salir'para terminar.\n")

    while True:
        pregunta=input('Que quieres analizar?')
        if pregunta.lower() in ['salir', 'quitar','exit']:
            print('Bye bye')
            break
        if not pregunta.strip():
            continue
        try:
            print("\n🤖 Procesando consulta analítica...")
            respuesta = agent.invoke(pregunta)
            print("\n💡 Resultado:")
            print("-" * 50)
            print(respuesta["output"])
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"⚠️ Error en la consulta: {e}\n")

if __name__ == "__main__":
    iniciar_sistema()
