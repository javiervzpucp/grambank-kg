# -*- coding: utf-8 -*-
"""
Pipeline completo mejorado: GramBank + Wikidata -> GraphDB -> RAG inteligente
"""

import os
import pandas as pd
import requests
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, GEO, XSD
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import re
from langchain_huggingface import HuggingFaceEndpoint

# Configuración inicial
load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GRAPHDB_SERVER = os.getenv("GRAPHDB_SERVER")
REPO_NAME = "linguistic_knowledge_graph"
GRAPHDB_USER = "grambank"
GRAPHDB_PASSWORD = os.getenv("GRAPHDB_PASSWORD")

# 1. Carga y limpieza de datos de GramBank
print("📥 Cargando y limpiando datos...")

# URLs de los datasets de GramBank
languages_url = "https://raw.githubusercontent.com/grambank/grambank/master/cldf/languages.csv"
parameters_url = "https://raw.githubusercontent.com/grambank/grambank/master/cldf/parameters.csv"
values_url = "https://raw.githubusercontent.com/grambank/grambank/master/cldf/values.csv"

# Procesamiento de datos con validación estricta
values = pd.read_csv(values_url)
values = values[values['Value'].apply(lambda x: str(x).isnumeric() and str(x) != '?')]
values['Value'] = values['Value'].astype(int)

languages = pd.read_csv(languages_url)
parameters = pd.read_csv(parameters_url)

# Filtrar lenguas objetivo con manejo de valores nulos
target_families = ["Quechuan", "Aymaran", "Araucanian"]
languages = languages[languages['Family_name'].isin(target_families)].dropna(subset=['Glottocode'])

# 2. Construcción del Knowledge Graph
print("🧠 Construyendo grafo RDF...")

# Configuración de namespaces
g = Graph()
LING = Namespace("http://example.org/linguistics#")
GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
WIKIDATA = Namespace("http://www.wikidata.org/entity/")

g.bind("ling", LING)
g.bind("glotto", GLOTTO)
g.bind("wd", WIKIDATA)
g.bind("geo", GEO)

def obtener_id_wikidata(glottocode):
    """Obtiene ID de Wikidata con manejo robusto de errores"""
    try:
        query = f'SELECT ?item WHERE {{ ?item wdt:P2208 "{glottocode}" }}'
        response = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers={"User-Agent": "LinguisticKG/1.0"},
            timeout=15
        )
        if response.status_code == 200:
            resultados = response.json().get('results', {}).get('bindings', [])
            return resultados[0]['item']['value'].split('/')[-1] if resultados else None
    except Exception as e:
        print(f"⚠️ Error en Wikidata para {glottocode}: {str(e)}")
    return None

# Procesamiento de lenguas con múltiples validaciones
for _, lang in languages.iterrows():
    try:
        glottocode = lang['Glottocode'].strip()
        if not glottocode:
            continue
            
        lang_uri = URIRef(GLOTTO[glottocode])
        
        # Metadatos básicos
        g.add((lang_uri, RDF.type, LING.Language))
        g.add((lang_uri, RDFS.label, Literal(lang['Name'].strip())))
        
        # Coordenadas geográficas con validación
        if pd.notna(lang['Latitude']) and pd.notna(lang['Longitude']):
            g.add((lang_uri, GEO.lat, Literal(float(lang['Latitude']), datatype=XSD.float)))
            g.add((lang_uri, GEO.long, Literal(float(lang['Longitude']), datatype=XSD.float)))
        
        # Enlace a Wikidata
        if wikidata_id := obtener_id_wikidata(glottocode):
            g.add((lang_uri, OWL.sameAs, URIRef(WIKIDATA[wikidata_id])))
        
        # Características gramaticales con triple validación
        lang_values = values[values['Language_ID'] == lang['ID']]
        for _, val in lang_values.iterrows():
            if val['Parameter_ID'] not in parameters['ID'].values:
                continue
                
            param = parameters[parameters['ID'] == val['Parameter_ID']].iloc[0]
            feature_uri = URIRef(LING[f"f{param['ID']}"])  # Mejor formato para URIs
            
            # Triples principales
            g.add((feature_uri, RDF.type, LING.Feature))
            g.add((feature_uri, RDFS.label, Literal(param['Name'].strip())))
            g.add((lang_uri, LING.hasFeature, feature_uri))
            
            # Valor con triple validación
            try:
                valor = int(val['Value'])
                g.add((feature_uri, LING.featureValue, Literal(valor, datatype=XSD.integer)))
            except ValueError:
                print(f"🚨 Valor inválido omitido: {val['Value']} para {lang['Name']}")
                
    except Exception as e:
        print(f"🔥 Error procesando {lang.get('Name', 'lengua desconocida')}: {str(e)}")

# 3. Almacenamiento y carga a GraphDB
print("💾 Guardando TTL localmente...")
g.serialize("linguistic_kg.ttl", format="turtle")

def cargar_graphdb():
    """Carga segura del KG a GraphDB con reintentos"""
    headers = {"Content-Type": "application/x-turtle"}
    endpoint = f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}/statements"
    
    try:
        with open("linguistic_kg.ttl", "rb") as f:
            for intento in range(3):
                response = requests.post(
                    endpoint,
                    data=f.read(),
                    headers=headers,
                    auth=(GRAPHDB_USER, GRAPHDB_PASSWORD),
                    timeout=45
                )
                if response.status_code == 204:
                    print("✅ KG cargado exitosamente en GraphDB")
                    return True
                print(f"⚠️ Intento {intento+1}/3 fallido: {response.status_code}")
        print(f"❌ Error persistente en la carga: {response.text}")
        return False
    except Exception as e:
        print(f"🚨 Error de conexión: {str(e)}")
        return False

# 4. Sistema RAG Avanzado
def construir_consulta_segura(lenguas):
    """Construye consultas SPARQL válidas y seguras con comparación exacta"""

    # Construimos los filtros correctamente sin `||` inválido
    filtros = " || ".join([f'LCASE(STR(?name)) = "{lengua.lower()}"' for lengua in lenguas])

    return f"""
        PREFIX ling: <http://example.org/linguistics#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?lang ?feature ?value ?name WHERE {{
            ?lang a ling:Language ;
                  rdfs:label ?name ;
                  ling:hasFeature ?feature ;
                  ling:hasFeatureValue ?value .
            FILTER({filtros})
        }} LIMIT 20
    """



def detectar_entidades(pregunta):
    """Detección mejorada de entidades lingüísticas"""
    patrones = {
        'quechua': r'\b(quechua|quichua|qichwa|qheshwa)\b',
        'aymara': r'\b(aymara|aimara|aymar)\b',
        'mapudungun': r'\b(mapudungun|mapuche|araucano|mapuzugun)\b'
    }
    
    detectadas = []
    pregunta = pregunta.lower()
    for lengua, patron in patrones.items():
        if re.search(patron, pregunta, flags=re.IGNORECASE):
            detectadas.append(lengua.capitalize())
    
    return detectadas

def generar_consulta_ia(pregunta):
    """Genera una consulta SPARQL optimizada usando IA"""

    plantilla = """<s>[INST] 
Eres un experto en SPARQL. Genera una consulta SPARQL **bien formada** para GraphDB basada en la siguiente pregunta.

📌 **Pregunta:** {pregunta}

🌐 **Ontología disponible:**
- `ling:Language`: Representa una lengua.
- `ling:hasFeature`: Relaciona una lengua con una característica gramatical.
- `ling:hasFeatureValue`: Representa el valor de una característica.
- `rdfs:label`: Contiene el nombre de la lengua.

⚠️ **Reglas que debes seguir:**
1️⃣ La consulta debe comenzar con `SELECT DISTINCT ?lang ?feature ?value WHERE {`
2️⃣ Usar `FILTER(LCASE(STR(?name)) = "xxx")` para comparar nombres de lenguas.
3️⃣ Comparar solo características compartidas entre dos lenguas.
4️⃣ **Cierra correctamente la consulta** con `}`.
5️⃣ **NO uses funciones inválidas** como `LDIFF`, `LANG()`, `||`, `ABS()`, `FILTER(LANG(...))`, etc.
6️⃣ La consulta debe devolver **máximo 20 resultados** con `LIMIT 20`.

📝 **Ejemplo de consulta bien formada**:
SELECT DISTINCT ?feature ?value WHERE { ?lang a ling:Language ; rdfs:label ?name ; ling:hasFeature ?feature ; ling:hasFeatureValue ?value . FILTER(LCASE(STR(?name)) = "quechua") } LIMIT 20

📌 Ahora genera **una consulta válida y completa** para responder a la pregunta:
{pregunta}

**Devuelve SOLO la consulta SPARQL sin explicaciones.** [/INST]"""

    mistral = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.3,
        max_new_tokens=350
    )
    
    consulta = mistral(plantilla.format(pregunta=pregunta)).strip()
    
    # Validación antes de ejecutarla
    if not consulta.startswith("SELECT") or "WHERE {" not in consulta or "}" not in consulta:
        raise ValueError(f"❌ Consulta generada no válida:\n{consulta}")
    
    print("🔎 Consulta SPARQL generada por Mistral:\n", consulta)
    
    return consulta

def ejecutar_consulta(consulta):
    """Ejecuta consultas SPARQL con manejo robusto de errores"""
    try:
        sparql = SPARQLWrapper(f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}")
        sparql.setCredentials(GRAPHDB_USER, GRAPHDB_PASSWORD)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(consulta)
        return sparql.query().convert()
    except SPARQLExceptions.SPARQLWrapperException as e:
        print(f"🔧 Error en consulta SPARQL: {str(e)}")
        return None

def procesar_resultados(resultados):
    """Procesamiento seguro de resultados SPARQL"""
    contextos = []
    if resultados and 'results' in resultados:
        for res in resultados['results']['bindings']:
            try:
                entrada = {
                    'lengua': res.get('lang', {}).get('value', '').split('/')[-1],
                    'caracteristica': res.get('feature', {}).get('value', '').split('#')[-1],
                    'valor': res.get('value', {}).get('value', 'N/D'),
                    'nombre': res.get('name', {}).get('value', 'Sin nombre')
                }
                contextos.append(f"{entrada['nombre']}: {entrada['caracteristica']} = {entrada['valor']}")
            except Exception as e:
                print(f"⚠️ Error procesando resultado: {str(e)}")
    return contextos

def generar_respuesta(contextos, pregunta):
    """Genera respuestas naturales con formato académico"""
    if not contextos:
        return "No se encontraron datos relevantes en la base de conocimiento."
    
    mistral = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.6, "max_new_tokens": 600}
    )
    
    prompt = f"""<s>[INST] Eres un lingüista experto. Analiza estos datos y responde:

Contexto:
{chr(10).join(contextos)}

Pregunta: {pregunta}

Requisitos:
- Respuesta en español con formato académico
- Menciona valores específicos cuando sea relevante
- Sé preciso y conciso [/INST]"""
    
    try:
        respuesta = mistral(prompt)
        return respuesta.split("[/INST]")[-1].strip().replace("  ", " ")
    except Exception as e:
        return f"Error generando respuesta: {str(e)}"

# 4. Sistema RAG Mejorado con Diagnóstico
def consulta_rag(pregunta):
    """Sistema RAG con autodiagnóstico y recuperación mejorada"""
    try:
        print("\n🔍 Iniciando proceso de consulta...")
        
        # Paso 1: Verificar conexión con GraphDB
        if not verificar_conexion_graphdb():
            return "Error: No se puede conectar a GraphDB. Verifique la configuración."
        
        # Paso 2: Detección de entidades mejorada
        lenguas = detectar_entidades(pregunta)
        print(f"🕵️ Entidades detectadas: {lenguas or 'Ninguna'}")
        
        # Paso 3: Búsqueda estructurada inicial
        resultados, consulta_usada = None, None
        if lenguas:
            consulta = construir_consulta_segura(lenguas)
            print(f"⚙️ Consulta inicial:\n{consulta}")
            resultados = ejecutar_consulta(consulta)
            consulta_usada = consulta
        
        # Paso 4: Búsqueda aumentada con IA si es necesario
        if not resultados or not resultados['results']['bindings']:
            print("⚡ Activando modo de búsqueda avanzada...")
            consulta_ia = generar_consulta_ia(pregunta)
            if consulta_ia and validar_consulta_sparql(consulta_ia):
                print(f"🤖 Consulta generada por IA:\n{consulta_ia}")
                resultados = ejecutar_consulta(consulta_ia)
                consulta_usada = consulta_ia
        
        # Paso 5: Procesamiento inteligente de resultados
        contextos = procesar_resultados(resultados) if resultados else []
        
        # Paso 6: Diagnóstico si no hay resultados
        if not contextos:
            return generar_mensaje_error(pregunta, consulta_usada)
        
        # Paso 7: Generación de respuesta
        return generar_respuesta(contextos, pregunta)
    
    except Exception as e:
        return f"Error en el sistema: {str(e)}"

# Funciones de soporte nuevas
def verificar_conexion_graphdb():
    """Verifica que GraphDB esté disponible"""
    try:
        endpoint = f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}/size"
        response = requests.get(endpoint, auth=(GRAPHDB_USER, GRAPHDB_PASSWORD), timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def validar_consulta_sparql(consulta):
    """Valida sintaxis básica de consulta SPARQL"""
    return all(palabra in consulta for palabra in ["SELECT", "WHERE"]) and "{" in consulta

def generar_mensaje_error(pregunta, consulta):
    """Genera mensajes de error detallados"""
    sugerencias = """
Posibles razones:
1. La lengua mencionada no está en nuestra base de datos
2. La característica solicitada no está documentada
3. Hay un error en la formulación de la pregunta

Sugerencias:
- Verifique la ortografía del nombre de la lengua
- Use términos técnicos (ej: 'orden de palabras', 'sistema de casos')
- Reformule su pregunta (ej: '¿Tiene el quechua incorporación nominal?')
"""
    
    diagnostico = f"""
⚠️ No se encontraron resultados para: '{pregunta}'
Consulta utilizada: {consulta or 'Ninguna'}
{sugerencias}
"""
    return diagnostico

# 5. Ejecución principal con verificación inicial
if __name__ == "__main__":
    if not verificar_conexion_graphdb():
        print("❌ Error: No se puede conectar a GraphDB. Verifique:")
        print(f"- URL: {GRAPHDB_SERVER}")
        print(f"- Repositorio: {REPO_NAME}")
        print(f"- Credenciales: Usuario '{GRAPHDB_USER}'")
        exit()
    
    if cargar_graphdb():
        print("\n💬 Sistema RAG listo (escribe 'salir' para terminar)")
        while True:
            try:
                pregunta = input("\nTu pregunta: ").strip()
                if not pregunta:
                    continue
                if pregunta.lower() in ['salir', 'exit']:
                    break
                
                respuesta = consulta_rag(pregunta)
                print("\n🤖 Respuesta:")
                print(respuesta)
                
            except KeyboardInterrupt:
                print("\n🛑 Operación cancelada por el usuario")
                break
            except Exception as e:
                print(f"\n💥 Error crítico: {str(e)}")