# -*- coding: utf-8 -*-
"""
Visualización t-SNE para Quechua y Arawak - Versión Corregida
"""

import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Configurar namespaces
LING = Namespace("http://purl.org/linguistics#")

def extract_families_from_ttl(ttl_path):
    """Extrae relaciones lengua-familia del TTL"""
    g = Graph()
    g.parse(ttl_path, format="turtle")
    
    families = {}
    lang_family = {}
    
    for lang in g.subjects(RDF.type, LING.Language):
        glottocode = str(lang).split("/")[-1]
        family_uris = list(g.objects(lang, LING.languageFamily))
        
        if family_uris:
            family_uri = str(family_uris[0])
            family_code = family_uri.split("/")[-1]
            lang_family[glottocode] = family_code
            
            if family_code not in families:
                family_name = list(g.objects(family_uris[0], RDFS.label))[0]
                families[family_code] = str(family_name)
    
    return lang_family, families

def load_and_preprocess_embeddings(embeddings_path):
    """Carga y normaliza los embeddings"""
    df = pd.read_csv(embeddings_path, index_col=0)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

def plot_tsne_quechua_arawak(embeddings, families, family_names):
    """Visualización t-SNE específica para Quechua y Arawak"""
    plt.figure(figsize=(12, 8))
    
    # Reducción dimensional con t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Filtrar familias objetivo
    target_families = {'quec1387': 'Quechua', 'araw1246': 'Arawak'}
    mask = np.isin(families, list(target_families.keys()))
    
    # Crear subconjuntos
    filtered_embeddings = embeddings_2d[mask]
    filtered_families = np.array(families)[mask]
    
    # Mapear a nombres legibles
    family_labels = [target_families[f] for f in filtered_families]
    
    # Configuración visual
    colors = {'Quechua': '#FF5733', 'Arawak': '#3380FF'}
    markers = {'Quechua': 'o', 'Arawak': 's'}
    
    # Plotear cada familia
    for family in target_families.values():
        idx = np.where(np.array(family_labels) == family)
        plt.scatter(
            filtered_embeddings[idx, 0],
            filtered_embeddings[idx, 1],
            label=f"{family} ({len(idx[0])} lenguas)",
            c=colors[family],
            marker=markers[family],
            alpha=0.7,
            edgecolor='w',
            s=80
        )
    
    # Ajustes finales
    plt.title('Distribución de lenguas Quechua y Arawak en el espacio de embeddings', fontsize=14)
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
    plt.legend(title="Familia Lingüística", loc='best')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carga de datos
    lang_family, family_names = extract_families_from_ttl("grambank_completo.ttl")
    embeddings_df = load_and_preprocess_embeddings("embeddings_from_ttl.csv")
    
    # Preparar datos
    common_keys = embeddings_df.index.intersection(lang_family.keys())
    filtered_embeddings = embeddings_df.loc[common_keys].values
    families = [lang_family[idx] for idx in common_keys]
    
    # Generar gráfico
    plot_tsne_quechua_arawak(filtered_embeddings, families, family_names)