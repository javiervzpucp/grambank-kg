import os
import json
from copy import deepcopy
from rdflib import Graph, URIRef, Literal, RDF, RDFS, XSD, Namespace
from rdflib.namespace import SKOS, DC, DCTERMS, GEO
from rdflib.plugins.sparql import prepareQuery
from tqdm import tqdm

# Configuración
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TTL_FILE = "grambank_completo.ttl"
OUTPUT_FILE = "rag_training_dataset_final.json"

# Namespaces personalizados
LING = Namespace("http://purl.org/linguistics#")
GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
GRAMBANK = Namespace("https://grambank.clld.org/parameters/")
GEO_NS = Namespace("http://www.opengis.net/ont/geosparql#")

# Mapeo completo de rasgos GB a español
GB_FEATURES = {
    'GB020': 'Presencia de artículos definidos/específicos',
    'GB021': 'Uso de artículos indefinidos en nombres no específicos',
    'GB022': 'Artículos prenominales',
    'GB023': 'Artículos posnominales',
    'GB024': 'Orden entre numeral y sustantivo',
    'GB025': 'Orden entre demostrativo y sustantivo',
    'GB026': 'Modificadores adnominales discontinuos',
    'GB027': 'Diferencia entre conjunción nominal y comitativo',
    'GB028': 'Distinción inclusivo/exclusivo',
    'GB030': 'Género en pronombres de tercera persona',
    'GB031': 'Formas duales/aumentadas en pronombres',
    'GB035': 'Demostrativos con 3+ distancias',
    'GB036': 'Demostrativos con distinción de elevación',
    'GB037': 'Demostrativos visible/no visible',
    'GB038': 'Clasificadores demostrativos',
    'GB039': 'Alomorfia no fonológica en marcadores de número',
    'GB041': 'Sustantivos supletivos para número',
    'GB042': 'Marcador morfológico de singular',
    'GB043': 'Marcador morfológico de dual',
    'GB044': 'Marcador morfológico de plural',
    'GB046': 'Marcador de plural asociativo',
    'GB047': 'Derivación de nombres de acción/estado',
    'GB048': 'Derivación de nombres de agente',
    'GB049': 'Derivación de nombres de objeto',
    'GB051': 'Sistema de género con masculino/femenino',
    'GB052': 'Clasificación por forma',
    'GB053': 'Clasificación por animacidad',
    'GB054': 'Clasificación por tipo de planta',
    'GB057': 'Clasificadores numerales',
    'GB058': 'Clasificadores posesivos',
    'GB059': 'Diferencia posesión alienable/inalienable',
    'GB065': 'Orden poseedor-poseído',
    'GB068': 'Propiedades léxicas como verbos',
    'GB069': 'Concordancia de modificadores adnominales',
    'GB070': 'Casos morfológicos para argumentos nucleares',
    'GB071': 'Casos para pronombres nucleares',
    'GB072': 'Casos para oblicuos no pronominales',
    'GB073': 'Casos para pronombres oblicuos',
    'GB074': 'Preposiciones',
    'GB075': 'Posposiciones',
    'GB079': 'Prefijos/proclíticos no argumentales',
    'GB080': 'Sufijos/enclíticos no argumentales',
    'GB081': 'Infixación verbal',
    'GB082': 'Marcación de tiempo presente',
    'GB083': 'Marcación de tiempo pasado',
    'GB084': 'Marcación de tiempo futuro',
    'GB086': 'Distinción perfectivo/imperfectivo',
    'GB089': 'Indexación de S con sufijo',
    'GB090': 'Indexación de S con prefijo',
    'GB091': 'Indexación de A con sufijo',
    'GB092': 'Indexación de A con prefijo',
    'GB093': 'Indexación de P con sufijo',
    'GB094': 'Indexación de P con prefijo',
    'GB095': 'Variación en marcación por TAM',
    'GB096': 'Variación en marcación por clase verbal',
    'GB098': 'Variación en marcación por persona',
    'GB099': 'Supletivismo verbal por persona',
    'GB103': 'Marcación de benefactivo',
    'GB104': 'Marcación de instrumental',
    'GB105': 'Receptor marcado como paciente',
    'GB107': 'Negación con afijo/clítico',
    'GB108': 'Marcación direccional/locativa',
    'GB109': 'Supletivismo verbal por número',
    'GB110': 'Supletivismo verbal por tiempo/aspecto',
    'GB111': 'Clases de conjugación',
    'GB113': 'Transitivización morfológica',
    'GB114': 'Marcador reflexivo ligado',
    'GB115': 'Marcador recíproco ligado',
    'GB116': 'Clasificación de argumentos en verbos',
    'GB117': 'Copula para predicados nominales',
    'GB118': 'Construcciones seriales verbales',
    'GB119': 'Modo marcado con auxiliar',
    'GB120': 'Aspecto marcado con auxiliar',
    'GB121': 'Tiempo marcado con auxiliar',
    'GB122': 'Composición verbal',
    'GB123': 'Construcciones verbo-adjunto',
    'GB124': 'Incorporación nominal intransitivizante',
    'GB126': 'Verbo existencial',
    'GB127': 'Verbos posturales obligatorios',
    'GB129': 'Inventario verbal reducido',
    'GB130': 'Orden S-V en intransitivas',
    'GB131': 'Orden verbo-inicial en transitivas',
    'GB132': 'Orden verbo-medial en transitivas',
    'GB133': 'Orden verbo-final en transitivas',
    'GB134': 'Mismo orden en principales/subordinadas',
    'GB135': 'Objetos clausales como nominales',
    'GB136': 'Orden fijo de argumentos',
    'GB137': 'Negación al final de cláusula',
    'GB138': 'Negación al inicio de cláusula',
    'GB139': 'Diferencia negación imperativa/declarativa',
    'GB140': 'Mismo negador para diferentes predicados',
    'GB146': 'Distinción eventos controlados/no controlados',
    'GB147': 'Pasiva morfológica',
    'GB148': 'Antipasiva morfológica',
    'GB149': 'Marca de inverso verbal',
    'GB150': 'Encadenamiento clausal',
    'GB151': 'Referencia cruzada (switch reference)',
    'GB152': 'Distinción simultáneo/secuencial',
    'GB155': 'Causativos con afijos/clíticos',
    'GB156': 'Causativo gramaticalizado de "decir"',
    'GB158': 'Reduplicación verbal',
    'GB159': 'Reduplicación nominal',
    'GB160': 'Reduplicación no verbal/nominal',
    'GB165': 'Marcador morfológico de trial',
    'GB166': 'Marcador morfológico de paucal',
    'GB167': 'Pronombre logofórico',
    'GB170': 'Concordancia de género en modificadores',
    'GB171': 'Concordancia de género en demostrativos',
    'GB172': 'Concordancia de género en artículos',
    'GB177': 'Marca de animacidad en verbos',
    'GB184': 'Concordancia de número en modificadores',
    'GB185': 'Concordancia de número en demostrativos',
    'GB186': 'Concordancia de número en artículos',
    'GB187': 'Marcación diminutiva en sustantivos',
    'GB188': 'Marcación aumentativa en sustantivos',
    'GB192': 'Clasificación por propiedades fonológicas',
    'GB193': 'Orden modificador-sustantivo',
    'GB196': 'Género en segunda persona',
    'GB197': 'Género en primera persona',
    'GB198': 'Concordancia numeral en género/clase',
    'GB203': 'Orden cuantificador universal-sustantivo',
    'GB204': 'Diferencia colectivo/distributivo',
    'GB250': 'Posesión con verbo "tener"',
    'GB252': 'Posesión con locativo',
    'GB253': 'Posesión con dativo',
    'GB254': 'Posesión con marcación adnominal',
    'GB256': 'Posesión comitativa',
    'GB257': 'Interrogativas polares por entonación',
    'GB260': 'Interrogativas polares por orden',
    'GB262': 'Partícula interrogativa inicial',
    'GB263': 'Partícula interrogativa final',
    'GB264': 'Partícula interrogativa media',
    'GB265': 'Comparativo con "superar"',
    'GB266': 'Comparativo con marcador locativo',
    'GB270': 'Comparativo con cláusulas coordinadas',
    'GB273': 'Comparativo con marcador no locativo',
    'GB275': 'Marcador ligado de comparativo',
    'GB276': 'Marcador libre de comparativo',
    'GB285': 'Interrogativa polar mixta',
    'GB286': 'Interrogativa polar solo verbal',
    'GB291': 'Interrogativa polar por tono',
    'GB297': 'Interrogativa polar V-no-V',
    'GB298': 'Negación con auxiliar flexivo',
    'GB299': 'Negación con partícula',
    'GB300': 'Supletivismo en verbo "dar"',
    'GB301': 'Construcción inclusiva',
    'GB302': 'Pasiva con partícula',
    'GB303': 'Antipasiva con partícula',
    'GB304': 'Agente explícito en pasivas',
    'GB305': 'Marcador reflexivo independiente',
    'GB306': 'Marcador recíproco independiente',
    'GB309': 'Múltiples tiempos pasados/futuros',
    'GB312': 'Marcación morfológica de modo',
    'GB313': 'Pronombres posesivos especiales',
    'GB314': 'Aumentativo por cambio de género',
    'GB315': 'Diminutivo por cambio de género',
    'GB316': 'Singular marcado con elemento libre',
    'GB317': 'Dual marcado con elemento libre',
    'GB318': 'Plural marcado con elemento libre',
    'GB319': 'Trial marcado con elemento libre',
    'GB320': 'Paucal marcado con elemento libre',
    'GB321': 'Clases nominales impredecibles',
    'GB322': 'Evidencialidad directa',
    'GB323': 'Evidencialidad indirecta',
    'GB324': 'Verbo interrogativo',
    'GB325': 'Distinción contable/masivo en interrogativos',
    'GB326': 'Interrogativos in situ',
    'GB327': 'Relativas posnominales',
    'GB328': 'Relativas prenominales',
    'GB329': 'Relativas de núcleo interno',
    'GB330': 'Relativas correlativas',
    'GB331': 'Relativas no adyacentes',
    'GB333': 'Sistema decimal',
    'GB334': 'Elementos quinarios',
    'GB335': 'Elementos vigesimales',
    'GB336': 'Sistema de conteo corporal',
    'GB400': 'Neutralización de personas',
    'GB401': 'Verbos paciente-lábiles',
    'GB402': 'Supletivismo en "ver"',
    'GB403': 'Supletivismo en "venir"',
    'GB408': 'Alineamiento acusativo',
    'GB409': 'Alineamiento ergativo',
    'GB410': 'Alineamiento neutro',
    'GB415': 'Distinción de cortesía en segunda persona',
    'GB421': 'Complementizador prepuesto',
    'GB422': 'Complementizador pospuesto',
    'GB430': 'Posesión con prefijo en poseedor',
    'GB431': 'Posesión con prefijo en poseído',
    'GB432': 'Posesión con sufijo en poseedor',
    'GB433': 'Posesión con sufijo en poseído',
    'GB519': 'Modo con partícula',
    'GB520': 'Aspecto con partícula',
    'GB521': 'Tiempo con partícula',
    'GB522': 'Omisión de S/A (pro-drop)'
}

def mejorar_fluidez_respuesta(original_output):
    """Mejora la fluidez y formato de las respuestas"""
    mejoras = {
        "Rasgos gramatales presentes: ": "**Características destacadas:**\n",
        "Rasgos ausentes: ": "\n**Rasgos ausentes:**\n",
        "Información sobre ": "",
        "Lat ": "Latitud: ",
        "Lon ": "Longitud: ",
        "Código ISO 639-3: ": "**Código ISO:** ",
        "Familia: ": "**Familia lingüística:** ",
        "Ubicación: ": "**Ubicación geográfica:** ",
        "?, ": "\n"
    }
    
    resultado = original_output
    for key, value in mejoras.items():
        resultado = resultado.replace(key, value)
    
    # Formatear listas
    for seccion in ["**Características destacadas:**", "**Rasgos ausentes:**"]:
        if seccion in resultado:
            partes = resultado.split(seccion)
            items = [f"- {item.strip()}" for item in partes[1].split("\n") if item.strip()]
            resultado = partes[0] + seccion + "\n" + "\n".join(items)
    
    return resultado

def crear_pregunta_corta(original_input):
    """Crea versiones concisas de las preguntas"""
    shortcuts = {
        "¿Qué información tienes sobre la lengua ": "¿Datos de ",
        "¿Qué lenguas pertenecen a la familia ": "¿Miembros de ",
        "¿Dónde se hablan las lenguas de la familia ": "¿Ubicación de ",
        " lengua": "",
        " la familia": "",
        "?": ""
    }
    
    corta = original_input
    for key, value in shortcuts.items():
        corta = corta.replace(key, value)
    
    return corta.strip() + "?"

class KGProcessor:
    def __init__(self):
        self.g = Graph()
        self.g.parse(TTL_FILE, format="turtle")
        self.entities = {
            'languages': {},
            'families': {},
            'features': {}
        }

    def extract_entities(self):
        """Extrae todas las entidades del grafo RDF con validación"""
        print("\nExtrayendo entidades del grafo RDF...")
        
        # Procesar lenguas
        for lang in tqdm(self.g.subjects(RDF.type, LING.Language), desc="Lenguas"):
            lang_uri = str(lang)
            try:
                self.entities['languages'][lang_uri] = {
                    'name': self._get_label(lang),
                    'iso': self._get_optional(lang, LING.iso639P3code),
                    'family': self._get_family(lang),
                    'location': self._get_location(lang),
                    'features': {
                        'present': list(self.g.objects(lang, LING.hasFeaturePresent)),
                        'absent': list(self.g.objects(lang, LING.hasFeatureAbsent))
                    }
                }
            except Exception as e:
                print(f"\nError procesando lengua {lang_uri}: {str(e)}")
                continue

        # Procesar familias
        for family in tqdm(self.g.subjects(RDF.type, LING.LanguageFamily), desc="Familias"):
            family_uri = str(family)
            try:
                self.entities['families'][family_uri] = {
                    'name': self._get_label(family),
                    'members': [
                        {
                            'uri': str(member),
                            'name': self._get_label(member)
                        } for member in self.g.subjects(LING.languageFamily, family)
                    ]
                }
            except Exception as e:
                print(f"\nError procesando familia {family_uri}: {str(e)}")
                continue

        # Procesar rasgos con datos extendidos
        for feature in tqdm(self.g.subjects(RDF.type, LING.GrammaticalFeature), desc="Rasgos"):
            feature_uri = str(feature)
            try:
                present_in = []
                for lang in self.g.subjects(LING.hasFeaturePresent, feature):
                    lang_data = self.entities['languages'].get(str(lang), {})
                    present_in.append({
                        'uri': str(lang),
                        'name': lang_data.get('name', str(lang).split('/')[-1])
                    })
                
                self.entities['features'][feature_uri] = {
                    'name': self._get_label(feature),
                    'description': self._get_optional(feature, RDFS.comment),
                    'present_in': present_in,
                    'present_count': len(present_in),
                    'example_languages': present_in[:3]
                }
            except Exception as e:
                print(f"\nError procesando rasgo {feature_uri}: {str(e)}")
                continue

    def _get_label(self, uri):
        """Obtiene etiqueta principal con validación"""
        label = self.g.value(uri, RDFS.label)
        return str(label) if label else uri.split("/")[-1]

    def _get_optional(self, uri, predicate):
        """Obtiene valores opcionales con manejo de nulos"""
        value = self.g.value(uri, predicate)
        return str(value) if value else None

    def _get_family(self, lang_uri):
        """Obtiene familia lingüística con validación de URI"""
        family = self.g.value(lang_uri, LING.languageFamily)
        return str(family) if family else None

    def _get_location(self, lang_uri):
        """Extrae datos geográficos con validación de tipos"""
        loc_node = self.g.value(lang_uri, GEO.location)
        if not loc_node:
            return None
            
        try:
            return {
                'lat': float(self.g.value(loc_node, GEO.lat)),
                'lon': float(self.g.value(loc_node, GEO.long)),
                'wkt': str(self.g.value(loc_node, GEO_NS.asWKT))
            }
        except TypeError as e:
            print(f"\nError en coordenadas de {lang_uri}: {str(e)}")
            return None

    def build_training_data(self):
        """Construye el dataset final con todos los tipos de preguntas"""
        training_data = []
        
        # Preguntas sobre lenguas
        for lang_uri, data in tqdm(self.entities['languages'].items(), desc="Generando preguntas lenguas"):
            try:
                entry = {
                    "input": f"¿Qué información tienes sobre la lengua {data['name']}?",
                    "output": self._build_language_output(lang_uri, data),
                    "sparql": self._generate_language_sparql(lang_uri)
                }
                training_data.append(entry)
            except KeyError as e:
                print(f"\nError en lengua {lang_uri}: Campo faltante {str(e)}")
                continue

        # Preguntas sobre familias
        for family_uri, data in tqdm(self.entities['families'].items(), desc="Generando preguntas familias"):
            try:
                # Pregunta sobre miembros
                entry_members = {
                    "input": f"¿Qué lenguas pertenecen a la familia {data['name']}?",
                    "output": self._build_family_members_output(data),
                    "sparql": self._generate_family_members_sparql(family_uri)
                }
                training_data.append(entry_members)
                
                # Pregunta sobre distribución geográfica
                entry_geo = {
                    "input": f"¿Dónde se hablan las lenguas de la familia {data['name']}?",
                    "output": self._build_family_geo_output(data),
                    "sparql": self._generate_family_geo_sparql(family_uri)
                }
                training_data.append(entry_geo)
            except KeyError as e:
                print(f"\nError en familia {family_uri}: Campo faltante {str(e)}")
                continue

        # Preguntas sobre rasgos
        for feature_uri, data in tqdm(self.entities['features'].items(), desc="Generando preguntas rasgos"):
            try:
                # Pregunta sobre lenguas con el rasgo
                entry_present = {
                    "input": f"¿Qué lenguas tienen el rasgo {data['name']}?",
                    "output": self._build_feature_present_output(data),
                    "sparql": self._generate_feature_present_sparql(feature_uri)
                }
                training_data.append(entry_present)
                
                # Pregunta sobre descripción del rasgo
                entry_desc = {
                    "input": f"¿En qué consiste el rasgo {data['name']}?",
                    "output": self._build_feature_description_output(data),
                    "sparql": self._generate_feature_description_sparql(feature_uri)
                }
                training_data.append(entry_desc)
                
            except KeyError as e:
                print(f"\nError en rasgo {feature_uri}: Campo faltante {str(e)}")
                continue

        return training_data

    def _build_language_output(self, lang_uri, data):
        """Construye la descripción de una lengua con formato mejorado"""
        output = []
        output.append(f"**{data['name']}**")
        
        if data.get('iso'):
            output.append(f"**Código ISO:** {data['iso']}")
        
        if data.get('family'):
            familia = self._get_family_name(data['family']) or "Desconocida"
            output.append(f"**Familia lingüística:** {familia}")
        
        if data.get('location'):
            lat = round(float(data['location']['lat']), 2)
            lon = round(float(data['location']['lon']), 2)
            output.append(f"**Ubicación geográfica:** {lat}°N, {lon}°E")
        
        # Limitar a 5 rasgos presentes y 3 ausentes
        present = [self._get_feature_name(f) for f in data['features']['present'][:5]]
        absent = [self._get_feature_name(f) for f in data['features']['absent'][:3]]
        
        if present:
            output.append("**Características destacadas:**")
            output.extend([f"- {feat}" for feat in present])
        if absent:
            output.append("**Rasgos no presentes:**")
            output.extend([f"- {feat}" for feat in absent])
        
        return "\n".join(output)

    def _get_family_name(self, family_uri):
        """Obtiene el nombre de una familia desde su URI"""
        return self.entities['families'].get(family_uri, {}).get('name', family_uri.split('/')[-1])

    def _get_feature_name(self, feature_uri):
        """Obtiene el nombre traducido del rasgo"""
        gb_code = feature_uri.split('/')[-1]
        return GB_FEATURES.get(gb_code, self.entities['features'].get(feature_uri, {}).get('name', gb_code))

    def _generate_language_sparql(self, lang_uri):
        """Genera SPARQL para obtener datos de una lengua"""
        return f"""
        PREFIX ling: <http://purl.org/linguistics#>
        SELECT ?name ?iso ?family ?lat ?long
        WHERE {{
            <{lang_uri}> rdfs:label ?name ;
                         ling:languageFamily ?family ;
                         geo:location ?loc .
            OPTIONAL {{ <{lang_uri}> ling:iso639P3code ?iso }}
            ?loc geo:lat ?lat ;
                 geo:long ?long .
        }}
        """

    def _build_family_members_output(self, data):
        """Construye la lista de miembros de una familia"""
        members = [f"{m['name']} ({m['uri'].split('/')[-1]})" for m in data['members']]
        if len(members) > 5:
            return f"La familia {data['name']} incluye:\n- " + "\n- ".join(members[:5]) + f"\n(y {len(members)-5} más)"
        return f"La familia {data['name']} incluye:\n- " + "\n- ".join(members)

    def _generate_family_members_sparql(self, family_uri):
        """Genera SPARQL para obtener miembros de una familia"""
        return f"""
        PREFIX ling: <http://purl.org/linguistics#>
        SELECT ?member ?name
        WHERE {{
            ?member ling:languageFamily <{family_uri}> ;
                    rdfs:label ?name .
        }}
        """

    def _build_family_geo_output(self, data):
        """Construye la salida de distribución geográfica"""
        locations = []
        for member in data['members'][:5]:  # Limitar a 5 ejemplos
            lang_data = self.entities['languages'].get(member['uri'])
            if lang_data and lang_data.get('location'):
                loc = lang_data['location']
                locations.append(f"- {member['name']}: {loc['lat']}, {loc['lon']}")
        
        if not locations:
            return "No hay datos de ubicación disponibles para esta familia."
        
        return f"Distribución de {data['name']}:\n" + "\n".join(locations)

    def _generate_family_geo_sparql(self, family_uri):
        """Genera SPARQL para ubicación de familia"""
        return f"""
        PREFIX ling: <http://purl.org/linguistics#>
        SELECT ?member ?name ?lat ?long
        WHERE {{
            ?member ling:languageFamily <{family_uri}> ;
                    rdfs:label ?name ;
                    geo:location ?loc .
            ?loc geo:lat ?lat ;
                 geo:long ?long .
        }}
        LIMIT 5
        """

    def _build_feature_present_output(self, data):
        """Construye la lista de lenguas con un rasgo"""
        examples = [f"- {lang['name']} ({self._get_family_name(lang['uri'])})" 
                   for lang in data['example_languages']]
        return (
            f"El rasgo **{data['name']}** está presente en {data['present_count']} lenguas.\n"
            f"Ejemplos destacados:\n" + "\n".join(examples)
        )

    def _generate_feature_present_sparql(self, feature_uri):
        """Genera SPARQL para lenguas con un rasgo"""
        return f"""
        PREFIX ling: <http://purl.org/linguistics#>
        SELECT ?lang ?name
        WHERE {{
            ?lang ling:hasFeaturePresent <{feature_uri}> ;
                   rdfs:label ?name .
        }}
        LIMIT 5
        """

    def _build_feature_description_output(self, data):
        """Construye la descripción de un rasgo"""
        if data['description']:
            return f"**{data['name']}**\n{data['description'].split('.')[0].strip()}."
        return f"**{data['name']}**\nDescripción técnica no disponible."

    def _generate_feature_description_sparql(self, feature_uri):
        """Genera SPARQL para descripción de rasgo"""
        return f"""
        PREFIX ling: <http://purl.org/linguistics#>
        SELECT ?description
        WHERE {{
            <{feature_uri}> rdfs:comment ?description .
        }}
        """

def main():
    processor = KGProcessor()
    processor.extract_entities()
    
    print("\nGenerando dataset de entrenamiento base...")
    training_data = processor.build_training_data()
    
    print("\nMejorando fluidez y creando variantes...")
    nuevos_datos = []
    
    for entrada in tqdm(training_data, desc="Procesando entradas"):
        try:
            # Crear versión mejorada
            entrada_mejorada = deepcopy(entrada)
            entrada_mejorada['output'] = mejorar_fluidez_respuesta(entrada['output'])
            nuevos_datos.append(entrada_mejorada)
            
            # Crear versión con pregunta corta
            entrada_corta = deepcopy(entrada_mejorada)
            entrada_corta['input'] = crear_pregunta_corta(entrada['input'])
            nuevos_datos.append(entrada_corta)
            
        except Exception as e:
            print(f"\nError procesando entrada: {str(e)}")
            continue
    
    # Guardar el dataset final
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(nuevos_datos, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset final generado exitosamente: {len(nuevos_datos)} pares pregunta-respuesta")
    print(f"Archivo guardado en: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()