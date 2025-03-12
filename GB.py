# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:11:06 2025

@author: jveraz
"""

import rdflib
import pandas as pd
import urllib.parse

# Cargar los datos
feature_grouping_df = pd.read_csv("data/feature_grouping_for_analysis.csv")
families_df = pd.read_csv("data/families.csv")
codes_df = pd.read_csv("data/codes.csv")
languages_df = pd.read_csv("data/languages.csv")
values_df = pd.read_csv("data/values.csv")
parameters_df = pd.read_csv("data/parameters.csv")

# Filtrar lenguas Quechuas, Mapudungun y Aymara
selected_families = ["Quechuan", "Aymaran", "Araucanian"]
languages_df = languages_df[languages_df["Family_name"].isin(selected_families)]

# Leer estructura filogenética de familias lingüísticas desde Newick
def parse_newick(newick_str):
    return newick_str.strip().split(";")[0]  # Obtener solo la estructura principal

families_df["Newick"] = families_df["Newick"].apply(parse_newick)

# Crear un grafo RDF
g = rdflib.Graph()

# Definir prefijos RDF
g.bind("dc", rdflib.Namespace("http://purl.org/dc/elements/1.1/"))
g.bind("dcterms", rdflib.Namespace("http://purl.org/dc/terms/"))
g.bind("owl", rdflib.Namespace("http://www.w3.org/2002/07/owl#"))
g.bind("rdfs", rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#"))
g.bind("ling", rdflib.Namespace("http://example.org/linguistic_features/"))
g.bind("fam", rdflib.Namespace("http://example.org/language_families/"))
g.bind("lang", rdflib.Namespace("http://example.org/languages/"))

# Base URI
base_uri = "http://example.org/resource/"

# Función para sanitizar URIs
def sanitize_uri(text):
    if pd.isna(text) or text.strip() == "":
        return "Unknown"
    text = text.strip().replace(" ", "_")
    return urllib.parse.quote(text)

# Agregar lenguas
for _, row in languages_df.iterrows():
    lang_uri = rdflib.URIRef(base_uri + sanitize_uri(row['Glottocode']))
    g.add((lang_uri, rdflib.RDF.type, rdflib.URIRef("http://example.org/ontology/Language")))
    g.add((lang_uri, rdflib.RDFS.label, rdflib.Literal(row["Name"])))
    g.add((lang_uri, rdflib.URIRef("http://example.org/ontology/latitude"), rdflib.Literal(row["Latitude"])))
    g.add((lang_uri, rdflib.URIRef("http://example.org/ontology/longitude"), rdflib.Literal(row["Longitude"])))
    family_id = sanitize_uri(row["Family_level_ID"]) if row["Family_level_ID"] else "Unknown"
    g.add((lang_uri, rdflib.URIRef("http://example.org/ontology/family"), rdflib.URIRef(base_uri + family_id)))

# Agregar familias lingüísticas
for _, row in families_df.iterrows():
    family_uri = rdflib.URIRef(base_uri + sanitize_uri(row["ID"]))
    g.add((family_uri, rdflib.RDF.type, rdflib.URIRef("http://example.org/ontology/LanguageFamily")))
    g.add((family_uri, rdflib.RDFS.label, rdflib.Literal(row["ID"])))
    g.add((family_uri, rdflib.URIRef("http://example.org/ontology/structure"), rdflib.Literal(row["Newick"])))

# Agregar rasgos lingüísticos y su agrupación
grouping_dict = feature_grouping_df.set_index("Feature_ID")["Finer_grouping"].to_dict()
for _, row in parameters_df.iterrows():
    feature_uri = rdflib.URIRef(base_uri + sanitize_uri(row["ID"]))
    g.add((feature_uri, rdflib.RDF.type, rdflib.URIRef("http://example.org/ontology/Feature")))
    g.add((feature_uri, rdflib.RDFS.label, rdflib.Literal(row["Name"])))
    g.add((feature_uri, rdflib.URIRef("http://example.org/ontology/description"), rdflib.Literal(row["Description"])))
    if row["ID"] in grouping_dict:
        g.add((feature_uri, rdflib.URIRef("http://example.org/ontology/group"), rdflib.Literal(grouping_dict[row["ID"]])))

# Agregar valores de los rasgos con sus significados
for _, row in values_df.iterrows():
    if row["Language_ID"] in languages_df["ID"].values:
        lang_uri = rdflib.URIRef(base_uri + sanitize_uri(row["Language_ID"]))
        feature_uri = rdflib.URIRef(base_uri + sanitize_uri(row["Parameter_ID"]))
        value_literal = rdflib.Literal(row["Value"])
        g.add((lang_uri, rdflib.URIRef("http://example.org/ontology/hasFeature"), feature_uri))
        g.add((feature_uri, rdflib.URIRef("http://example.org/ontology/hasValue"), value_literal))
        
        # Agregar significado del valor si está en codes_df
        code_info = codes_df[(codes_df["Parameter_ID"] == row["Parameter_ID"]) & (codes_df["ID"] == row["Code_ID"])]
        if not code_info.empty:
            meaning_literal = rdflib.Literal(code_info.iloc[0]["Description"])
            g.add((value_literal, rdflib.URIRef("http://example.org/ontology/meaning"), meaning_literal))

# Guardar en archivo Turtle
ttl_file = "data/language_knowledge_graph.ttl"
g.serialize(destination=ttl_file, format="turtle")

print("✅ Archivo TTL generado con éxito.")




