import pandas as pd
import requests
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, GEO, XSD
import spacy
from bs4 import BeautifulSoup
import concurrent.futures
from urllib.parse import urljoin, quote_plus
import re
from tqdm import tqdm

# Configuración NLP
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")

# Namespaces
LING = Namespace("http://purl.org/linguistics#")
GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
WIKIDATA = Namespace("http://www.wikidata.org/entity/")
LEXVO = Namespace("http://lexvo.org/id/iso639-3/")
DCT = Namespace("http://purl.org/dc/terms/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# Configuración
target_families = {"Quechuan", "Aymaran", "Araucanian", "Mapudungun"}
WIKI_BASE = "https://github.com/grambank/Grambank/wiki/"

session = requests.Session()
session.headers.update({"User-Agent": "GrambankKG/4.0"})

def obtener_datos_wikidata(glottocode):
    """Consulta SPARQL mejorada con múltiples propiedades"""
    query = f'''
    SELECT DISTINCT 
        ?item 
        (GROUP_CONCAT(DISTINCT ?familyLabel; separator="|") as ?familias
        ?population
        (GROUP_CONCAT(DISTINCT ?writingSystemLabel; separator="|") as ?escrituras
        (GROUP_CONCAT(DISTINCT ?countryLabel; separator="|") as ?paises
        (GROUP_CONCAT(DISTINCT ?iso; separator="|") as ?isocodes
    WHERE {{
        ?item wdt:P2208 "{glottocode}".
        
        OPTIONAL {{
            ?item wdt:P171* ?family.
            ?family rdfs:label ?familyLabel.
            FILTER(LANG(?familyLabel) = "en"
        }}
        
        OPTIONAL {{ ?item wdt:P1098 ?population. }}
        OPTIONAL {{ ?item wdt:P282 ?writingSystem. }}
        OPTIONAL {{ ?item wdt:P17 ?country. }}
        OPTIONAL {{ ?item wdt:P218 ?iso. }}
        
        SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?writingSystem rdfs:label ?writingSystemLabel.
            ?country rdfs:label ?countryLabel.
        }}
    }}
    GROUP BY ?item ?population
    '''
    try:
        response = session.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            timeout=30
        )
        data = response.json()
        return procesar_resultados_wikidata(data.get("results", {}).get("bindings", []))
    except Exception as e:
        print(f"⚠️ Wikidata {glottocode}: {str(e)}")
        return None

def procesar_resultados_wikidata(bindings):
    """Procesa resultados de Wikidata de forma segura"""
    if not bindings:
        return None
    
    result = bindings[0]
    return {
        "wikidata_id": extraer_id(result.get("item", {})),
        "familias": separar_valores(result.get("familias", {}).get("value", "")),
        "population": safe_int(result.get("population", {}).get("value")),
        "escrituras": separar_valores(result.get("escrituras", {}).get("value", "")),
        "paises": separar_valores(result.get("paises", {}).get("value", "")),
        "isocodes": separar_valores(result.get("isocodes", {}).get("value", ""))
    }

def extraer_id(item):
    return item.get("value", "").split("/")[-1] if item else None

def safe_int(value):
    try: return int(float(value)) if value else None
    except: return None

def separar_valores(cadena):
    return [v.strip() for v in cadena.split("|") if v.strip()] if cadena else []

def extraer_wiki_feature(url):
    """Extrae características de la wiki con manejo de errores mejorado"""
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", {"class": "markdown-body"})
        
        return {
            "url": url,
            "sections": procesar_secciones(content),
            "tables": procesar_tablas(content),
            "raw_text": " ".join(p.get_text() for p in content.find_all("p"))
        }
    except Exception as e:
        print(f"⚠️ Error en {url}: {str(e)}")
        return None

def procesar_secciones(content):
    sections = {}
    current_header = "Introducción"
    for element in content.find_all(["h1", "h2", "h3", "p", "ul", "ol"]):
        if element.name in ["h1", "h2", "h3"]:
            current_header = element.get_text().strip()
            sections[current_header] = []
        else:
            sections[current_header].append(element.get_text().strip())
    return sections

def procesar_tablas(content):
    tables = {}
    for table in content.find_all("table"):
        headers = [th.get_text().strip() for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text().strip() for td in tr.find_all("td")]
            if cells and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
        if rows:
            tables[f"Tabla_{len(tables)+1}"] = rows
    return tables

def obtener_features_wiki():
    """Extracción paralelizada mejorada con validación estricta de IDs"""
    try:
        response = session.get(urljoin(WIKI_BASE, "List-of-all-features"))
        soup = BeautifulSoup(response.text, "html.parser")
        
        feature_links = []
        for link in soup.find_all("a", href=re.compile(r'GB\d{3}')):
            if (match := re.search(r'(GB\d{3})', link["href"], re.IGNORECASE)):
                feature_links.append({
                    "id": match.group(1).upper(),
                    "url": urljoin(WIKI_BASE, link["href"])
                })
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            features = []
            urls = [link["url"] for link in feature_links]
            for future in tqdm(concurrent.futures.as_completed(
                [executor.submit(extraer_wiki_feature, url) for url in urls]),
                total=len(urls),
                desc="Extrayendo features"
            ):
                if (result := future.result()):
                    features.append(result)
        
        return {link["id"]: feat for link, feat in zip(feature_links, features) if feat}
    except Exception as e:
        print(f"⚠️ Error crítico: {str(e)}")
        return {}

def procesar_entidades(texto):
    doc = nlp(texto)
    entidades = []
    for ent in doc.ents:
        tipo = ent.label_
        categoria = LING.Location if tipo in ["GPE", "LOC"] else \
                   LING.Language if tipo == "LANGUAGE" else LING.Concept
        
        uri_segura = quote_plus(ent.text.strip().lower().replace(" ", "_"))
        entidad_uri = URIRef(LEXVO[uri_segura]) if tipo == "LANGUAGE" else LING[uri_segura]
        
        entidades.append({
            "uri": entidad_uri,
            "label": ent.text.strip(),
            "type": categoria,
            "context": ent.sent.text
        })
    return entidades

# Inicialización del grafo RDF
g = Graph()
for prefix, ns in [("ling", LING), ("glotto", GLOTTO), ("wd", WIKIDATA),
                  ("lexvo", LEXVO), ("prov", PROV), ("dct", DCT), ("geo", GEO)]:
    g.bind(prefix, ns)

# Metadatos de procedencia
provenance_uri = URIRef("http://linguistic.lkg/provenance/grambank")
g.add((provenance_uri, RDF.type, PROV.Activity))
g.add((provenance_uri, DCT.source, URIRef("https://github.com/grambank/grambank")))
g.add((provenance_uri, DCT.created, Literal(pd.Timestamp.now().isoformat(), datatype=XSD.dateTime)))

# Carga y preprocesamiento de datos
print("⏳ Cargando datos CLDF...")
languages = pd.read_csv("https://raw.githubusercontent.com/grambank/grambank/master/cldf/languages.csv")
parameters = pd.read_csv("https://raw.githubusercontent.com/grambank/grambank/master/cldf/parameters.csv")
values = pd.read_csv("https://raw.githubusercontent.com/grambank/grambank/master/cldf/values.csv")

# Filtrado y validación
values = values[values['Value'].astype(str).str.strip().isin(['0','1'])].copy()
values['Value'] = values['Value'].astype(int)
values = values[values['Language_ID'].isin(languages['ID'])].copy()

languages = languages[
    languages['Family_name'].str.strip().str.title().isin(target_families) & 
    languages['Glottocode'].notna()
].reset_index(drop=True)

# Procesamiento de lenguas
print("🔍 Procesando lenguas...")
for _, lang in tqdm(languages.iterrows(), total=len(languages)):
    try:
        glottocode = lang['Glottocode'].strip()
        lang_uri = URIRef(GLOTTO[glottocode])
        
        g.add((lang_uri, RDF.type, LING.Language))
        g.add((lang_uri, RDFS.label, Literal(lang['Name'].strip())))
        g.add((lang_uri, LING.glottocode, Literal(glottocode)))
        
        # Coordenadas
        if pd.notna(lang['Latitude']) and pd.notna(lang['Longitude']):
            try:
                lat = round(float(lang['Latitude']), 4)
                lon = round(float(lang['Longitude']), 4)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    geo_uri = URIRef(f"{lang_uri}/location")
                    g.add((geo_uri, RDF.type, GEO.Point))
                    g.add((geo_uri, GEO.lat, Literal(lat, datatype=XSD.float)))
                    g.add((geo_uri, GEO.long, Literal(lon, datatype=XSD.float)))
                    g.add((lang_uri, GEO.location, geo_uri))
            except (ValueError, TypeError):
                pass
        
        # Datos de Wikidata
        if (wikidata_info := obtener_datos_wikidata(glottocode)):
            wikidata_uri = URIRef(WIKIDATA[wikidata_info['wikidata_id']])
            g.add((lang_uri, OWL.sameAs, wikidata_uri))
            
            for familia in wikidata_info.get('familias', []):
                family_uri = LING[f"family/{quote_plus(familia)}"]
                g.add((family_uri, RDF.type, LING.LanguageFamily))
                g.add((family_uri, RDFS.label, Literal(familia)))
                g.add((lang_uri, LING.hasFamily, family_uri))
            
            for iso in wikidata_info.get('isocodes', []):
                if iso: g.add((lang_uri, LING.iso639Code, Literal(iso.strip())))
            
            if (pop := wikidata_info.get('population')):
                g.add((lang_uri, LING.speakerCount, Literal(pop, datatype=XSD.integer)))
                
    except Exception as e:
        print(f"⚠️ Error en {lang['Name']}: {str(e)}")

# Procesamiento de características
print("📚 Extrayendo características...")
features_wiki = obtener_features_wiki()

for feature_id, feat in tqdm(features_wiki.items(), desc="Procesando features"):
    try:
        feature_uri = URIRef(LING[f"feature/{feature_id}"])
        g.add((feature_uri, RDF.type, LING.GrammaticalFeature))
        g.add((feature_uri, RDFS.label, Literal(feature_id)))
        g.add((feature_uri, RDFS.seeAlso, URIRef(feat['url'])))
        g.add((feature_uri, PROV.wasGeneratedBy, provenance_uri))
        
        # Secciones y entidades
        for section, content in feat['sections'].items():
            section_uri = URIRef(f"{feature_uri}/section/{quote_plus(section)}")
            g.add((section_uri, RDF.type, LING.WikiSection))
            g.add((section_uri, RDFS.label, Literal(section)))
            g.add((section_uri, DCT.description, Literal("\n".join(content))))
            g.add((feature_uri, LING.hasSection, section_uri))
            
            for ent in procesar_entidades("\n".join(content)):
                g.add((ent['uri'], RDF.type, ent['type']))
                g.add((ent['uri'], RDFS.label, Literal(ent['label'])))
                g.add((section_uri, LING.mentions, ent['uri']))
        
        # Tablas
        for table_name, rows in feat['tables'].items():
            table_uri = URIRef(f"{feature_uri}/table/{quote_plus(table_name)}")
            g.add((table_uri, RDF.type, LING.DataTable))
            g.add((table_uri, RDFS.label, Literal(table_name)))
            
            for i, row in enumerate(rows, 1):
                row_uri = URIRef(f"{table_uri}/row/{i}")
                g.add((row_uri, RDF.type, LING.TableRow))
                for key, value in row.items():
                    g.add((row_uri, LING[quote_plus(key)], Literal(value)))
                g.add((table_uri, LING.containsRow, row_uri))
            
            g.add((feature_uri, LING.hasDataTable, table_uri))
            
    except Exception as e:
        print(f"⚠️ Error en {feature_id}: {str(e)}")

# Procesamiento de observaciones
print("🔢 Procesando valores...")
for _, row in tqdm(values.iterrows(), total=len(values)):
    try:
        if not (match := re.match(r'GB(\d+)', row['Parameter_ID'])):
            continue
        param_number = match.group(1).zfill(3)
        feature_id = f"GB{param_number}"
        
        lang_data = languages[languages['ID'] == row['Language_ID']]
        if lang_data.empty:
            continue
            
        glottocode = lang_data['Glottocode'].values[0].strip()
        lang_uri = URIRef(GLOTTO[glottocode])
        feature_uri = URIRef(LING[f"feature/{feature_id}"])
        
        observation_uri = URIRef(f"{lang_uri}/obs/{row['ID']}")
        g.add((observation_uri, RDF.type, LING.FeatureObservation))
        g.add((observation_uri, LING.forLanguage, lang_uri))
        g.add((observation_uri, LING.aboutFeature, feature_uri))
        g.add((observation_uri, LING.hasValue, Literal(row['Value'], datatype=XSD.integer)))
        
        if pd.notna(row['Source']):
            g.add((observation_uri, PROV.wasDerivedFrom, URIRef(row['Source'])))
            
    except Exception as e:
        print(f"⚠️ Error en fila {row.name}: {str(e)}")

# Guardado final
print("💾 Guardando grafo...")
g.serialize("grambank_kg.ttl", format="turtle", encoding="utf-8")

print(f"""
✅ KG generado exitosamente!
- Lenguas: {len(languages)}
- Features: {len(features_wiki)}
- Observaciones: {len(values)}
- Tripletas: {len(g):,}
""")