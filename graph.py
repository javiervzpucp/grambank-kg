# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 21:25:12 2025

@author: jveraz
"""

import rdflib
import networkx as nx
import matplotlib.pyplot as plt

# Cargar el grafo RDF
ttl_file_path = "data/language_knowledge_graph.ttl"
g = rdflib.Graph()
g.parse(ttl_file_path, format="turtle")

# Crear un grafo de NetworkX
G = nx.DiGraph()

# Definir colores según el tipo de entidad
node_colors = {}
color_map = {
    "Language": "blue",
    "Feature": "green",
    "Value": "red",
    "Family": "orange",
    "Default": "gray"
}

# Agregar nodos y relaciones al grafo de NetworkX
for s, p, o in g:
    G.add_edge(s, o, label=p)
    
    if s not in node_colors:
        if "ontology/Language" in str(g.value(s, rdflib.RDF.type, default="")):
            node_colors[s] = color_map["Language"]
        elif "ontology/Feature" in str(g.value(s, rdflib.RDF.type, default="")):
            node_colors[s] = color_map["Feature"]
        elif "ontology/Value" in str(g.value(s, rdflib.RDF.type, default="")):
            node_colors[s] = color_map["Value"]
        elif "ontology/LanguageFamily" in str(g.value(s, rdflib.RDF.type, default="")):
            node_colors[s] = color_map["Family"]
        else:
            node_colors[s] = color_map["Default"]

# Generar lista de colores en orden de los nodos
node_color_list = [node_colors.get(node, "gray") for node in G.nodes()]

# Visualizar el grafo con un layout mejorado
plt.figure(figsize=(14, 10))
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, with_labels=False, node_size=80, edge_color="gray", node_color=node_color_list, alpha=0.8)

# Dibujar algunas etiquetas para entender mejor la estructura
selected_nodes = list(G.nodes())[:15]  # Etiquetar solo algunos nodos
node_labels = {node: node.split('/')[-1] for node in selected_nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")

plt.title("Visualización del Grafo RDF de Lenguas")
plt.show()
