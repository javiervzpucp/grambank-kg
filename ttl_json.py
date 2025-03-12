# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:11:28 2025

@author: jveraz
"""

import json
import warnings
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# Suprimir advertencias de depreciación
warnings.simplefilter("ignore", category=FutureWarning)

# Cargar el TTL actualizado
ttl_file_path = "grambank_simple.ttl"
graph = Graph()
graph.parse(ttl_file_path, format="turtle")

# Extraer información del grafo y convertir a JSON
languages = {}
for s in graph.subjects(RDF.type, URIRef("http://purl.org/linguistics#Language")):
    glottocode = str(s).split("/")[-1]
    label = str(graph.value(s, RDFS.label, default=Literal("Desconocido")))
    family = str(graph.value(s, URIRef("http://purl.org/linguistics#languageFamily"), default=Literal("Desconocida")))
    
    present_features = {}
    for feature in graph.objects(s, URIRef("http://purl.org/linguistics#hasFeaturePresent")):
        feature_id = str(feature).split("/")[-1]
        feature_label = str(graph.value(feature, RDFS.label, default=Literal("Sin descripción")))
        present_features[feature_id] = feature_label
    
    absent_features = {}
    for feature in graph.objects(s, URIRef("http://purl.org/linguistics#hasFeatureAbsent")):
        feature_id = str(feature).split("/")[-1]
        feature_label = str(graph.value(feature, RDFS.label, default=Literal("Sin descripción")))
        absent_features[feature_id] = feature_label
    
    languages[label.lower()] = {
        "glottocode": glottocode,
        "family": family,
        "present_features": present_features,
        "absent_features": absent_features
    }

# Guardar el JSON
ttl_json_file = "grambank_simple.json"
with open(ttl_json_file, "w", encoding="utf-8") as json_file:
    json.dump(languages, json_file, indent=4, ensure_ascii=False)

print(f"✅ Archivo JSON guardado en {ttl_json_file}")