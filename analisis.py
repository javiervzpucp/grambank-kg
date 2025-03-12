# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 21:02:59 2025

@author: jveraz
"""

import rdflib

# Cargar el grafo RDF
ttl_file_path = "data/language_knowledge_graph.ttl"
g = rdflib.Graph()
g.parse(ttl_file_path, format="turtle")

# Definir la URI de la lengua Cusco Quechua (cusc1236)
lang_uri = rdflib.URIRef("http://example.org/resource/cusc1236")

# Obtener todas las propiedades y valores asociados a la lengua
properties = list(g.predicate_objects(lang_uri))

# Obtener rasgos lingüísticos asociados a esta lengua
has_feature = rdflib.URIRef("http://example.org/ontology/hasFeature")
features = list(g.objects(lang_uri, has_feature))

# Obtener valores y sus significados para los rasgos de la lengua
has_value = rdflib.URIRef("http://example.org/ontology/hasValue")
meaning = rdflib.URIRef("http://example.org/ontology/meaning")

feature_values = {}
for feature in features:
    values = list(g.objects(feature, has_value))
    for value in values:
        meanings = list(g.objects(value, meaning))
        feature_values[str(feature)] = {
            "value": str(value),
            "meaning": str(meanings[0]) if meanings else "Desconocido"
        }

# Mostrar los resultados
print("📌 Propiedades de Cusco Quechua (cusc1236):")
for prop, obj in properties[:10]:  # Mostrar solo las primeras 10 propiedades
    print(f"{prop} → {obj}")

print("\n📌 Número de rasgos asociados:", len(features))

print("\n📌 Valores y significados de los primeros 10 rasgos:")
for feature, data in list(feature_values.items())[:10]:
    print(f"{feature}: {data['value']} ({data['meaning']})")