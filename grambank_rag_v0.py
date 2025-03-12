import rdflib
import pandas as pd
from tabulate import tabulate
from langchain_huggingface import HuggingFaceEndpoint

# 📌 Cargar el archivo TTL en un grafo RDF
ttl_file_path = "linguistic_kg.ttl"  # Asegúrate de tener el archivo en la misma carpeta
g = rdflib.Graph()
g.parse(ttl_file_path, format="turtle")

# 📌 Configurar Mistral
mistral = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.6,
    max_new_tokens=500
)

# 📌 Función para ejecutar consultas SPARQL en el archivo TTL local
def ejecutar_sparql_local(consulta):
    try:
        qres = g.query(consulta)
        
        # Obtener nombres de columnas
        columnas = [str(var) for var in qres.vars]

        # Convertir resultados a lista
        filas = [list(row) for row in qres]

        # Convertir a DataFrame
        df = pd.DataFrame(filas, columns=columnas)

        return df if not df.empty else "⚠️ No se encontraron datos."
    
    except Exception as e:
        return f"⚠️ Error en la consulta: {str(e)}"

# 📌 Lista de preguntas sugeridas con sus consultas SPARQL asociadas
preguntas_sparql = {
    1: {
        "pregunta": "📌 ¿Cuáles son todas las lenguas disponibles en la base de datos?",
        "consulta": """
            PREFIX glotto: <https://glottolog.org/resource/languoid/id/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?lang ?name WHERE {
                ?lang a <http://example.org/linguistics#Language> ;
                      rdfs:label ?name .
            } LIMIT 50
        """
    },
    2: {
        "pregunta": "📌 ¿Cuáles son las características lingüísticas de Mapudungun?",
        "consulta": """
            PREFIX glotto: <https://glottolog.org/resource/languoid/id/>
            PREFIX ling: <http://example.org/linguistics#>

            SELECT ?feature WHERE {
                glotto:mapu1245 ling:hasFeature ?feature .
            }
        """
    },
    3: {
        "pregunta": "📌 ¿Cuáles son las diferencias entre Aymara y Mapudungun?",
        "consulta": """
            PREFIX glotto: <https://glottolog.org/resource/languoid/id/>
            PREFIX ling: <http://example.org/linguistics#>

            SELECT DISTINCT ?feature ?aymara ?mapudungun WHERE {
                {
                    glotto:cent2142 ling:hasFeature ?feature .
                    BIND("Sí" AS ?aymara)
                }
                UNION
                {
                    glotto:mapu1245 ling:hasFeature ?feature .
                    BIND("Sí" AS ?mapudungun)
                }
            }
        """
    },
    4: {
        "pregunta": "📌 ¿Cuáles son las coordenadas geográficas de Cusco Quechua?",
        "consulta": """
            PREFIX glotto: <https://glottolog.org/resource/languoid/id/>
            PREFIX geo: <http://www.opengis.net/ont/geosparql#>

            SELECT ?lat ?long WHERE {
                glotto:cusc1236 geo:lat ?lat ;
                                geo:long ?long .
            }
        """
    }
}

# 📌 Función para generar una explicación usando Mistral
def generar_explicacion(df, pregunta):
    if isinstance(df, str):  # Si no hay datos, devolvemos el mensaje de error
        return df
    
    # Convertir los resultados a texto estructurado
    contexto = df.to_string(index=False)

    # Construir el prompt para Mistral
    prompt = f"""<s>[INST] Eres un lingüista experto en lenguas indígenas.
Usa los siguientes datos para responder de manera clara y estructurada:

**Contexto:**
{contexto}

📌 **Pregunta:** {pregunta}

✅ **Genera una respuesta ordenada y detallada.** [/INST]"""

    try:
        respuesta = mistral(prompt)
        return respuesta.split("[/INST]")[-1].strip()
    except Exception as e:
        return f"⚠️ Error generando respuesta: {str(e)}"

# 📌 Menú interactivo mejorado con generación de explicación
def menu_interactivo():
    while True:
        print("\n🔍 **Preguntas Sugeridas:**")
        for num, data in preguntas_sparql.items():
            print(f"{num}. {data['pregunta']}")

        seleccion = input("\n🔹 **Elige un número (o escribe 'salir' para terminar):** ")

        if seleccion.lower() == "salir":
            print("👋 Saliendo del sistema.")
            break

        try:
            seleccion = int(seleccion)
            if seleccion in preguntas_sparql:
                pregunta = preguntas_sparql[seleccion]['pregunta']
                print(f"\n🔎 Ejecutando consulta para: {pregunta}\n")

                # Ejecutar consulta y obtener datos
                df_resultados = ejecutar_sparql_local(preguntas_sparql[seleccion]["consulta"])

                # 📌 Generar explicación con Mistral
                explicacion = generar_explicacion(df_resultados, pregunta)

                # 📌 Mostrar resultados
                if isinstance(df_resultados, pd.DataFrame):
                    print("\n📊 **Resultados:**")
                    print(tabulate(df_resultados, headers="keys", tablefmt="grid"))
                
                print("\n📝 **Explicación:**")
                print(explicacion)

            else:
                print("⚠️ Número inválido. Por favor, elige un número de la lista.")
        except ValueError:
            print("⚠️ Entrada inválida. Ingresa un número.")

# Ejecutar menú interactivo
menu_interactivo()
