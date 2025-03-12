import os
import json
from rdflib import Graph, RDF, RDFS, SKOS, Namespace
from rdflib.namespace import GEO
from rdflib.plugins.sparql import prepareQuery

# Configuración
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ttl_file = "grambank_completo.ttl"

# Namespaces
LING = Namespace("http://purl.org/linguistics#")
GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
GRAMBANK = Namespace("https://grambank.clld.org/parameters/")
GEO_NS = Namespace("http://www.opengis.net/ont/geosparql#")

# Cargar grafo RDF
g = Graph()
g.parse(ttl_file, format="turtle")

# Preparar consultas recurrentes
FEATURE_DISTRIBUTION_QUERY = prepareQuery('''
    SELECT ?lang ?langName WHERE {
        ?lang ling:hasFeaturePresent $feature ;
              rdfs:label ?langName .
    }
    LIMIT 5
''', initNs={"ling": LING, "rdfs": RDFS})

def extract_entities():
    """Extrae entidades estructuradas del grafo RDF"""
    entities = {
        'languages': {},
        'families': {},
        'features': {}
    }

    # Extraer lenguas
    for lang in g.subjects(RDF.type, LING.Language):
        lang_uri = str(lang)
        entities['languages'][lang_uri] = {
            'name': g.value(lang, RDFS.label),
            'iso': g.value(lang, LING.iso639P3code),
            'family': g.value(lang, LING.languageFamily),
            'location': get_location(lang),
            'features': {
                'present': list(g.objects(lang, LING.hasFeaturePresent)),
                'absent': list(g.objects(lang, LING.hasFeatureAbsent))
            }
        }

    # Extraer familias
    for family in g.subjects(RDF.type, LING.LanguageFamily):
        family_uri = str(family)
        entities['families'][family_uri] = {
            'name': g.value(family, RDFS.label),
            'parent': g.value(family, SKOS.broader),
            'children': list(g.objects(family, SKOS.narrower)),
            'languages': list(g.subjects(LING.languageFamily, family))
        }

    # Extraer rasgos
    for feature in g.subjects(RDF.type, LING.GrammaticalFeature):
        feature_uri = str(feature)
        entities['features'][feature_uri] = {
            'name': g.value(feature, RDFS.label),
            'description': g.value(feature, RDFS.comment),
            'present_in': list(g.subjects(LING.hasFeaturePresent, feature)),
            'absent_in': list(g.subjects(LING.hasFeatureAbsent, feature))
        }
    
    return entities

def get_location(lang):
    """Extrae datos geográficos estructurados"""
    location = g.value(lang, GEO.location)
    if location:
        return {
            'wkt': g.value(location, GEO_NS.asWKT),
            'lat': g.value(location, GEO.lat),
            'lon': g.value(location, GEO.long)
        }
    return None

def generate_language_questions(lang_uri, data):
    """Genera preguntas para una lengua"""
    questions = []
    lang_name = data['name']
    
    # Preguntas básicas
    questions.append({
        "input": f"¿Qué es {lang_name}?",
        "output": f"{lang_name} es una lengua indígena de Sudamérica documentada en GramBank.",
        "sparql": f"ASK {{ <{lang_uri}> a ling:Language }}"
    })
    
    # Familia lingüística
    if data['family']:
        questions.append({
            "input": f"¿A qué familia pertenece {lang_name}?",
            "output": f"Pertenece a la familia {g.value(data['family'], RDFS.label)}.",
            "sparql": f"SELECT ?familyLabel WHERE {{ <{lang_uri}> ling:languageFamily/rdfs:label ?familyLabel }}"
        })
    
    # Datos geográficos
    if data['location']:
        questions.append({
            "input": f"¿Dónde se habla {lang_name}?",
            "output": f"Coordenadas: Lat {data['location']['lat']}, Long {data['location']['lon']}",
            "sparql": f"SELECT ?lat ?long WHERE {{ <{lang_uri}> geo:location/geo:lat ?lat ; geo:location/geo:long ?long }}"
        })
    
    # Rasgos lingüísticos
    if data['features']['present']:
        questions.append({
            "input": f"¿Qué rasgos gramatales caracterizan a {lang_name}?",
            "output": f"Presenta {len(data['features']['present'])} rasgos como: {', '.join([g.value(f, RDFS.label) for f in data['features']['present'][:3]])}",
            "sparql": f"SELECT ?featureLabel WHERE {{ <{lang_uri}> ling:hasFeaturePresent/rdfs:label ?featureLabel }} LIMIT 3"
        })
    
    return questions

def generate_family_questions(family_uri, data):
    """Genera preguntas para una familia lingüística"""
    questions = []
    family_name = data['name']
    
    # Composición de la familia
    questions.append({
        "input": f"¿Qué lenguas incluye la familia {family_name}?",
        "output": f"Incluye {len(data['languages'])} lenguas como: {', '.join([g.value(l, RDFS.label) for l in data['languages'][:3]])}",
        "sparql": f"SELECT ?langLabel WHERE {{ ?lang ling:languageFamily <{family_uri}> ; rdfs:label ?langLabel }} LIMIT 3"
    })
    
    # Jerarquía familiar
    if data['parent']:
        questions.append({
            "input": f"¿A qué familia mayor pertenece {family_name}?",
            "output": f"Pertenece a la familia {g.value(data['parent'], RDFS.label)}",
            "sparql": f"SELECT ?parentLabel WHERE {{ <{family_uri}> skos:broader/rdfs:label ?parentLabel }}"
        })
    
    # Distribución geográfica
    questions.append({
        "input": f"¿En qué región se hablan las lenguas de la familia {family_name}?",
        "output": "Región sudamericana (coordenadas específicas varían por lengua)",
        "sparql": f"SELECT DISTINCT ?lat ?long WHERE {{ ?lang ling:languageFamily <{family_uri}> ; geo:location/geo:lat ?lat ; geo:location/geo:long ?long }}"
    })
    
    return questions

def generate_feature_questions(feature_uri, data):
    """Genera preguntas para un rasgo gramatical"""
    questions = []
    feature_name = data['name']
    
    # Descripción del rasgo
    questions.append({
        "input": f"¿En qué consiste el rasgo {feature_name}?",
        "output": data['description'] or "Descripción no disponible",
        "sparql": f"SELECT ?desc WHERE {{ <{feature_uri}> rdfs:comment ?desc }}"
    })
    
    # Distribución
    questions.append({
        "input": f"¿Cuántas lenguas tienen el rasgo {feature_name}?",
        "output": f"Presente en {len(data['present_in'])} lenguas, ausente en {len(data['absent_in'])}",
        "sparql": f"SELECT (COUNT(?lang) as ?count) WHERE {{ ?lang ling:hasFeaturePresent <{feature_uri}> }}"
    })
    
    # Ejemplos
    if data['present_in']:
        questions.append({
            "input": f"¿Qué lenguas tienen el rasgo {feature_name}?",
            "output": f"Ejemplos: {', '.join([g.value(l, RDFS.label) for l in data['present_in']][:3])}",
            "sparql": f"SELECT ?langLabel WHERE {{ ?lang ling:hasFeaturePresent <{feature_uri}> ; rdfs:label ?langLabel }} LIMIT 3"
        })
    
    return questions

def main():
    # Extraer datos estructurados
    entities = extract_entities()
    training_data = []
    
    # Generar preguntas para cada entidad
    for lang_uri, lang_data in entities['languages'].items():
        training_data.extend(generate_language_questions(lang_uri, lang_data))
    
    for family_uri, family_data in entities['families'].items():
        training_data.extend(generate_family_questions(family_uri, family_data))
    
    for feature_uri, feature_data in entities['features'].items():
        training_data.extend(generate_feature_questions(feature_uri, feature_data))
    
    # Guardar resultados
    output_file = "rag_training_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset generado con {len(training_data)} preguntas")

if __name__ == "__main__":
    main()