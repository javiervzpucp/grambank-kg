# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:44:43 2025

@author: jveraz
"""

import requests
from SPARQLWrapper import SPARQLWrapper, JSON

# Configuración de GraphDB
GRAPHDB_SERVER = "http://localhost:7200"  # Cambia según tu servidor
REPO_NAME = "linguistic_knowledge_graph"  # Nombre del repositorio en GraphDB
GRAPHDB_USER = "grambank"
GRAPHDB_PASSWORD = "efgwa231A"  # Asegúrate de que es correcto

# Ruta del archivo TTL que quieres subir
ttl_file = "linguistic_kg.ttl"

# Endpoint para subir datos
endpoint = f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}/statements"

# Leer el archivo TTL
with open(ttl_file, "rb") as f:
    data = f.read()

# Configurar headers
headers = {"Content-Type": "application/x-turtle"}

# Realizar la solicitud POST con autenticación
response = requests.post(endpoint, data=data, headers=headers, auth=(GRAPHDB_USER, GRAPHDB_PASSWORD))

# Verificar resultado
if response.status_code == 204:
    print("✅ Archivo TTL subido exitosamente a GraphDB")
else:
    print(f"❌ Error: {response.status_code} - {response.text}")

sparql = SPARQLWrapper(f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}")
sparql.setCredentials(GRAPHDB_USER, GRAPHDB_PASSWORD)
sparql.setReturnFormat(JSON)

# Consulta para ver los primeros 10 datos
sparql.setQuery("""
    SELECT * WHERE {
        ?s ?p ?o .
    } LIMIT 10
""")

# Ejecutar consulta
results = sparql.query().convert()

# Mostrar resultados
for res in results["results"]["bindings"]:
    print(f"{res['s']['value']}  {res['p']['value']}  {res['o']['value']}")