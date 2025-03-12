from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, GEO, SKOS
from collections import defaultdict
import textwrap

def analizar_ttl(archivo_ttl, archivo_reporte):
    """Genera un reporte detallado del Knowledge Graph sin metadatos"""
    print(f"Analizando {archivo_ttl}...")
    
    # Configurar namespaces
    LING = Namespace("http://purl.org/linguistics#")
    GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
    GRAMBANK = Namespace("https://grambank.clld.org/parameters/")
    
    kg = Graph()
    kg.bind("ling", LING)
    kg.bind("glotto", GLOTTO)
    kg.bind("gb", GRAMBANK)
    kg.parse(archivo_ttl, format="turtle")
    
    # ===== Métricas básicas =====
    stats = {
        'total_triples': len(kg),
        'lenguas': set(kg.subjects(RDF.type, LING.Language)),
        'familias': set(kg.subjects(RDF.type, LING.LanguageFamily)),
        'rasgos': set(kg.subjects(RDF.type, LING.GrammaticalFeature)),
        'presentes': len(list(kg.subject_objects(LING.hasFeaturePresent))),
        'ausentes': len(list(kg.subject_objects(LING.hasFeatureAbsent))),
        'coordenadas': len(list(kg.objects(None, GEO.location))),
    }

    # ===== Análisis de entidades =====
    def get_short_id(uri):
        return uri.split('/')[-1] if isinstance(uri, URIRef) else str(uri)
    
    # Distribución geográfica aproximada
    regiones = defaultdict(int)
    for coord in kg.objects(None, GEO.location):
        if (coord, GEO.lat, None) in kg and (coord, GEO.long, None) in kg:
            lat = float(kg.value(coord, GEO.lat))
            region = "América" if lat < 15 else "Otros"  # Simplificación para ejemplo
            regiones[region] += 1
    
    # ===== Análisis de relaciones =====
    relaciones = defaultdict(int)
    for s, p, o in kg:
        if isinstance(p, URIRef):
            relaciones[str(p)] += 1

    # ===== Análisis jerárquico =====
    familias_detalle = defaultdict(list)
    for s, p, o in kg.triples((None, LING.languageFamily, None)):
        if isinstance(o, URIRef):
            familias_detalle[o].append(s)
    
    # ===== Análisis de rasgos =====
    rasgos_detalle = defaultdict(lambda: {'presentes': 0, 'ausentes': 0})
    for s, p, o in kg:
        if p == LING.hasFeaturePresent:
            rasgos_detalle[o]['presentes'] += 1
        elif p == LING.hasFeatureAbsent:
            rasgos_detalle[o]['ausentes'] += 1

    # ===== Construcción del reporte =====
    reporte = textwrap.dedent(f"""
    === Reporte Lingüístico del Knowledge Graph ===
    
    [Estadísticas Fundamentales]
    - Triples totales: {stats['total_triples']:,}
    - Lenguas documentadas: {len(stats['lenguas']):,}
    - Familias lingüísticas: {len(stats['familias']):,}
    - Rasgos gramaticales: {len(stats['rasgos']):,}
    
    [Distribución Geográfica]
    - Lenguas con coordenadas: {stats['coordenadas']} ({stats['coordenadas']/len(stats['lenguas']):.1%})
    - Regiones aproximadas: {dict(regiones)}
    
    [Topología del Grafo]
    - Relaciones únicas: {len(relaciones)}
    - Relaciones más frecuentes:
    """)
    
    for rel, count in sorted(relaciones.items(), key=lambda x: -x[1])[:3]:
        reporte += f"    - {rel.split('#')[-1]}: {count}\n"

    reporte += textwrap.dedent(f"""
    [Jerarquía Lingüística]
    - Familias con más miembros:
    """)
    
    top_familias = sorted(familias_detalle.items(), key=lambda x: -len(x[1]))[:5]
    for familia, miembros in top_familias:
        nombre = kg.value(familia, RDFS.label) or get_short_id(familia)
        reporte += f"    - {nombre}: {len(miembros)} lenguas\n"

    reporte += "\n[Rasgos más Significativos]\n"
    for rasgo, counts in sorted(rasgos_detalle.items(), key=lambda x: -(x[1]['presentes'] + x[1]['ausentes']))[:5]:
        nombre = kg.value(rasgo, RDFS.label) or get_short_id(rasgo)
        total = counts['presentes'] + counts['ausentes']
        reporte += textwrap.dedent(f"""
        - {nombre} ({get_short_id(rasgo)}):
          * Frecuencia total: {total}
          * Presentes: {counts['presentes']} ({counts['presentes']/total:.1%})
          * Ausentes: {counts['ausentes']} ({counts['ausentes']/total:.1%})
        """)

    # Guardar reporte
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write(reporte.strip())

    print(f"✅ Reporte generado: {archivo_reporte}")

if __name__ == "__main__":
    analizar_ttl("grambank_completo.ttl", "grambank_report.txt")