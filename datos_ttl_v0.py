import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, GEO, DC, DCTERMS, SKOS
import requests
import time

# Configurar namespaces
LING = Namespace("http://purl.org/linguistics#")
GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
GRAMBANK = Namespace("https://grambank.clld.org/parameters/")
WIKIDATA = Namespace("https://www.wikidata.org/wiki/")

# Crear grafo RDF
g = Graph()

# Vincular namespaces
namespaces = {
    "ling": LING,
    "glotto": GLOTTO,
    "gb": GRAMBANK,
    "wikidata": WIKIDATA,
    "geo": GEO,
    "dc": DC,
    "dcterms": DCTERMS,
    "skos": SKOS
}

for prefix, uri in namespaces.items():
    g.bind(prefix, uri)

def cargar_datos():
    """Carga y filtra los datasets"""
    print("Cargando datos...")
    
    languages = pd.read_csv("https://raw.githubusercontent.com/grambank/grambank/master/cldf/languages.csv")
    parameters = pd.read_csv("https://raw.githubusercontent.com/grambank/grambank/master/cldf/parameters.csv")
    values = pd.read_csv("https://raw.githubusercontent.com/grambank/grambank/master/cldf/values.csv")
    
    languages = languages[languages['Macroarea'] == "South America"]
    
    values['Value'] = pd.to_numeric(values['Value'], errors='coerce')
    values = values[values['Value'].isin([0, 1])]
    values['Value'] = values['Value'].astype(int)
    
    return languages, parameters, values

def procesar_familias(languages):
    """Procesa familias lingüísticas desde Glottolog"""
    print("Procesando familias lingüísticas...")
    
    familias_glottocodes = languages['Family_level_ID'].dropna().unique()
    
    for glottocode in familias_glottocodes:
        familia_uri = URIRef(GLOTTO[glottocode])
        ttl_url = f"https://glottolog.org/resource/languoid/id/{glottocode}.ttl"
        
        try:
            time.sleep(0.5)
            response = requests.get(ttl_url, timeout=15)
            response.raise_for_status()
            
            temp_graph = Graph()
            temp_graph.parse(data=response.text, format="turtle")
            
            # Añadir metadatos básicos
            g.add((familia_uri, RDF.type, LING.LanguageFamily))
            
            # Obtener nombre de la familia desde el CSV
            nombre_familia = languages[languages['Family_level_ID'] == glottocode]['Family_name'].iloc[0]
            g.add((familia_uri, RDFS.label, Literal(nombre_familia)))
            
            # Añadir triples del TTL
            for triple in temp_graph.triples((familia_uri, None, None)):
                g.add(triple)
                
            print(f"✓ Familia {glottocode} procesada")
            
        except Exception as e:
            print(f"⚠️ Error en familia {glottocode}: {str(e)}")
            continue

def procesar_lenguas(languages):
    """Procesa lenguas con sus metadatos"""
    print("Procesando lenguas...")
    for _, row in languages.iterrows():
        lang_uri = URIRef(GLOTTO[row['Glottocode']])
        
        # Metadatos básicos
        g.add((lang_uri, RDF.type, LING.Language))
        g.add((lang_uri, RDFS.label, Literal(row['Name'])))
        g.add((lang_uri, LING.glottocode, Literal(row['Glottocode'])))
        
        if pd.notna(row['ISO639P3code']):
            g.add((lang_uri, LING.iso639P3code, Literal(row['ISO639P3code'])))
        
        # Enlace a familia lingüística
        if pd.notna(row['Family_level_ID']):
            family_uri = URIRef(GLOTTO[row['Family_level_ID']])
            g.add((lang_uri, LING.languageFamily, family_uri))
        
        # Coordenadas geográficas
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            geo_uri = BNode()
            g.add((geo_uri, RDF.type, GEO.Point))
            g.add((geo_uri, GEO.lat, Literal(float(row['Latitude']), datatype=XSD.float)))
            g.add((geo_uri, GEO.long, Literal(float(row['Longitude']), datatype=XSD.float)))
            g.add((lang_uri, GEO.location, geo_uri))

def procesar_rasgos(parameters):
    """Procesa rasgos con metadatos extendidos"""
    print("Procesando rasgos...")
    for idx, row in parameters.iterrows():
        feature_id = row['ID']
        feature_uri = URIRef(GRAMBANK[feature_id])
        
        g.add((feature_uri, RDF.type, LING.GrammaticalFeature))
        g.add((feature_uri, RDFS.label, Literal(row['Name'])))
        if pd.notna(row['Description']):
            g.add((feature_uri, RDFS.comment, Literal(row['Description'])))
        
        ttl_url = f"https://grambank.clld.org/parameters/{feature_id}.ttl"
        try:
            time.sleep(0.3)
            response = requests.get(ttl_url, timeout=10)
            response.raise_for_status()
            
            temp_graph = Graph()
            temp_graph.parse(data=response.text, format="turtle")
            
            for triple in temp_graph.triples((feature_uri, None, None)):
                g.add(triple)
                
            print(f"✓ Rasgo {feature_id} enriquecido")
            
        except Exception as e:
            print(f"⚠️ Error en rasgo {feature_id}: {str(e)}")
            continue

def procesar_valores(languages, values):
    """Crea relaciones entre lenguas y rasgos"""
    print("Creando relaciones...")
    lenguas_dict = languages.set_index('ID')['Glottocode'].to_dict()
    
    for _, row in values.iterrows():
        try:
            glottocode = lenguas_dict[row['Language_ID']]
            lang_uri = URIRef(GLOTTO[glottocode])
            feature_uri = URIRef(GRAMBANK[row['Parameter_ID']])
            
            if row['Value'] == 1:
                g.add((lang_uri, LING.hasFeaturePresent, feature_uri))
            else:
                g.add((lang_uri, LING.hasFeatureAbsent, feature_uri))
                
        except KeyError as e:
            print(f"Error en fila {row.name}: {str(e)}")
            continue

def main():
    languages, parameters, values = cargar_datos()
    
    procesar_familias(languages)
    procesar_lenguas(languages)
    procesar_rasgos(parameters)
    procesar_valores(languages, values)
    
    print("\nGuardando grafo...")
    g.serialize("grambank_completo.ttl", format="turtle")
    print(f"✅ KG generado! Triples totales: {len(g):,}")

if __name__ == "__main__":
    main()