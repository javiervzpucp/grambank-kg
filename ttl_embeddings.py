# -*- coding: utf-8 -*-
"""
Generador de Embeddings Lingüísticos Multimodales - Versión Final Integrada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from rdflib import Graph, Namespace, RDF, RDFS
from rdflib.namespace import GEO, DCTERMS
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import logging
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de namespaces
LING = Namespace("http://purl.org/linguistics#")
GLOTTO = Namespace("https://glottolog.org/resource/languoid/id/")
GRAMBANK = Namespace("https://grambank.clld.org/parameters/")

class EnhancedTTLProcessor:
    def __init__(self, ttl_path):
        self.g = Graph()
        try:
            self.g.parse(ttl_path, format="turtle")
            logging.info("TTL cargado exitosamente")
        except Exception as e:
            logging.error(f"Error cargando TTL: {str(e)}")
            raise
        self.features = {}
        self.graph_data = {}
        
    def extract_all_features(self):
        """Coordina la extracción de todas las características con manejo de errores"""
        try:
            self._extract_grammatical_features()
            self._extract_geographic_data()
            self._extract_textual_data()
            self._build_family_graph()
            return self
        except Exception as e:
            logging.error(f"Error en extracción de características: {str(e)}")
            raise
    
    def _extract_grammatical_features(self):
        """Procesa características gramaticales con validación robusta"""
        try:
            features = sorted(str(f).split("/")[-1] for f in self.g.subjects(RDF.type, LING.GrammaticalFeature))
            languages = [str(lang).split("/")[-1] for lang in self.g.subjects(RDF.type, LING.Language)]
            
            matrix = -np.ones((len(languages), len(features)), dtype=np.float32)
            lang_index = {lang: idx for idx, lang in enumerate(languages)}
            feat_index = {feat: idx for idx, feat in enumerate(features)}
            
            for lang in self.g.subjects(RDF.type, LING.Language):
                lang_id = str(lang).split("/")[-1]
                if lang_id not in lang_index:
                    continue
                
                idx = lang_index[lang_id]
                present = set()
                absent = set()
                
                try:
                    present = {str(f).split("/")[-1] for f in self.g.objects(lang, LING.hasFeaturePresent)}
                    absent = {str(f).split("/")[-1] for f in self.g.objects(lang, LING.hasFeatureAbsent)}
                except Exception as e:
                    logging.warning(f"Error procesando características para {lang_id}: {str(e)}")
                
                for feat, i in feat_index.items():
                    matrix[idx, i] = 1.0 if feat in present else 0.0 if feat in absent else -1.0
            
            self.features['grammatical'] = pd.DataFrame(matrix, index=languages, columns=features)
            logging.info("Características gramaticales procesadas: %d lenguas, %d rasgos", len(languages), len(features))
        except Exception as e:
            logging.error("Error en características gramaticales: %s", str(e))
            raise
    
    def _extract_geographic_data(self):
        """Procesa datos geográficos con imputación robusta"""
        try:
            geo_data = {}
            for lang in self.g.subjects(RDF.type, LING.Language):
                lang_id = str(lang).split("/")[-1]
                lat, long = np.nan, np.nan
                
                try:
                    point = next(self.g.objects(lang, GEO.location), None)
                    if point:
                        lat_obj = next(self.g.objects(point, GEO.lat), None)
                        long_obj = next(self.g.objects(point, GEO.long), None)
                        lat = float(lat_obj) if lat_obj else np.nan
                        long = float(long_obj) if long_obj else np.nan
                except Exception as e:
                    logging.warning(f"Error geográfico para {lang_id}: {str(e)}")
                
                geo_data[lang_id] = [lat, long]
            
            df = pd.DataFrame.from_dict(geo_data, orient='index', columns=['lat', 'long'])
            imp = SimpleImputer(strategy='median')
            self.features['geo'] = pd.DataFrame(imp.fit_transform(df), index=df.index, columns=df.columns)
            logging.info("Datos geográficos procesados: %d registros", len(df))
        except Exception as e:
            logging.error("Error en datos geográficos: %s", str(e))
            raise
    
    def _extract_textual_data(self):
        """Genera embeddings textuales con manejo de errores robusto"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            model = AutoModel.from_pretrained("bert-base-multilingual-cased")
            text_data = {}
            
            for lang in self.g.subjects(RDF.type, LING.Language):
                lang_id = str(lang).split("/")[-1]
                name, desc = "", ""
                
                try:
                    name = str(next(self.g.objects(lang, DCTERMS.title))) if self.g.objects(lang, DCTERMS.title) else ""
                except Exception:
                    pass
                
                try:
                    desc = str(next(self.g.objects(lang, DCTERMS.description))) if self.g.objects(lang, DCTERMS.description) else ""
                except Exception:
                    pass
                
                text = f"{name}: {desc}".strip()[:512]  # Limitar longitud
                if not text or text == ":":
                    text = "Lengua sin descripción"
                
                try:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                except Exception as e:
                    logging.warning(f"Error texto para {lang_id}: {str(e)}")
                    embedding = np.zeros(768)
                
                text_data[lang_id] = embedding
            
            self.features['text'] = pd.DataFrame.from_dict(text_data, orient='index')
            logging.info("Embeddings textuales generados: %d lenguas", len(text_data))
        except Exception as e:
            logging.error("Error en datos textuales: %s", str(e))
            raise
    
    def _build_family_graph(self):
        """Construye grafo de relaciones familiares con validación"""
        try:
            edges = []
            family_map = {}
            if 'grammatical' not in self.features:
                raise ValueError("Procesar características gramaticales primero")
            
            lang_index = {lang: idx for idx, lang in enumerate(self.features['grammatical'].index)}
            
            for lang in self.g.subjects(RDF.type, LING.Language):
                lang_id = str(lang).split("/")[-1]
                if lang_id not in lang_index:
                    continue
                
                current_idx = lang_index[lang_id]
                relations = []
                
                try:
                    relations += list(self.g.objects(lang, LING.languageFamily))
                    relations += list(self.g.objects(lang, LING.dialectOf))
                except Exception as e:
                    logging.warning(f"Error relaciones para {lang_id}: {str(e)}")
                    continue
                
                for rel in relations:
                    try:
                        family_uri = str(rel)
                        if family_uri not in family_map:
                            family_map[family_uri] = len(family_map)
                        edges.append([current_idx, family_map[family_uri]])
                    except Exception as e:
                        logging.warning(f"Error procesando relación {rel}: {str(e)}")
                        continue
            
            self.graph_data['edge_index'] = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.graph_data['num_nodes'] = len(family_map)
            logging.info("Grafo familiar construido: %d nodos, %d aristas", len(family_map), len(edges))
        except Exception as e:
            logging.error("Error construyendo grafo: %s", str(e))
            raise

class MultiModalEmbedder(nn.Module):
    def __init__(self, input_dims, emb_dim=256):
        super().__init__()
        
        # Configuración dimensional
        self.gram_out = 256
        self.geo_out = 32
        self.text_out = 128
        self.graph_out = 256 * 2  # 2 heads
        
        # Encoders
        self.gram_encoder = nn.Sequential(
            nn.LayerNorm(input_dims['gram']),
            nn.Linear(input_dims['gram'], 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, self.gram_out)
        )
        
        self.geo_encoder = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.geo_out)
        )
        
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, self.text_out)
        )
        
        # Graph Transformer
        self.graph_conv = TransformerConv(
            in_channels=self.gram_out,
            out_channels=self.graph_out // 2,
            heads=2,
            dropout=0.3
        )
        
        # Proyección final
        self.projection = nn.Sequential(
            nn.Linear(self.gram_out + self.geo_out + self.text_out + self.graph_out, 1024),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, emb_dim)
        )
        
        # Decoders
        self.decoder = nn.ModuleDict({
            'gram': nn.Linear(emb_dim, input_dims['gram']),
            'geo': nn.Linear(emb_dim, 2),
            'text': nn.Linear(emb_dim, 768)
        })
    
    def forward(self, gram, geo, text, edge_index):
        # Codificación
        g_enc = self.gram_encoder(gram)
        geo_enc = self.geo_encoder(geo)
        txt_enc = self.text_encoder(text)
        
        # Procesamiento gráfico
        graph_enc = F.leaky_relu(self.graph_conv(g_enc, edge_index))
        
        # Fusión multimodal
        combined = torch.cat([g_enc, geo_enc, txt_enc, graph_enc], dim=-1)
        embeddings = self.projection(combined)
        
        # Reconstrucción
        return {
            'gram': self.decoder['gram'](embeddings),
            'geo': self.decoder['geo'](embeddings),
            'text': self.decoder['text'](embeddings)
        }, embeddings

class EnhancedEmbeddingGenerator:
    def __init__(self, ttl_path):
        self.processor = EnhancedTTLProcessor(ttl_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Dispositivo de cómputo: {self.device}")
    
    def generate(self, epochs=300):
        """Generación de embeddings con entrenamiento supervisado y early stopping"""
        try:
            # Extracción de características
            self.processor.extract_all_features()
            
            # Preparación de datos
            gram = torch.tensor(self.processor.features['grammatical'].values, dtype=torch.float32).to(self.device)
            geo = torch.tensor(self.processor.features['geo'].values, dtype=torch.float32).to(self.device)
            text = torch.tensor(self.processor.features['text'].values, dtype=torch.float32).to(self.device)
            edge_index = self.processor.graph_data['edge_index'].to(self.device)
            
            # Inicialización del modelo
            model = MultiModalEmbedder({
                'gram': gram.shape[1],
                'geo': geo.shape[1],
                'text': text.shape[1]
            }).to(self.device)
            
            # Configuración de optimización
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
            
            # Entrenamiento con early stopping
            best_loss = float('inf')
            patience = 20
            no_improve = 0
            loss_history = []
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                reconstructions, _ = model(gram, geo, text, edge_index)
                
                # Cálculo de pérdidas
                loss_gram = F.mse_loss(reconstructions['gram'], gram)
                loss_geo = F.mse_loss(reconstructions['geo'], geo)
                loss_text = F.cosine_embedding_loss(
                    reconstructions['text'], 
                    text, 
                    torch.ones(text.size(0), device=self.device)
                )
                total_loss = loss_gram + 0.5 * loss_geo + 0.2 * loss_text
                
                # Backpropagation
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(total_loss)
                
                # Registro de pérdidas
                loss_history.append(total_loss.item())
                
                # Early stopping
                if total_loss < best_loss:
                    best_loss = total_loss
                    no_improve = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    logging.info(f"Early stopping en época {epoch+1}")
                    break
                
                # Logging cada 10 épocas
                if (epoch + 1) % 10 == 0:
                    logging.info(
                        f"Época {epoch+1}/{epochs} | "
                        f"Loss: {total_loss:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )
            
            # Visualización de la curva de aprendizaje
            plt.plot(loss_history)
            plt.title("Curva de Aprendizaje")
            plt.xlabel("Época")
            plt.ylabel("Loss")
            plt.savefig("learning_curve.png")
            plt.close()
            
            # Carga del mejor modelo
            model.load_state_dict(torch.load('best_model.pth'))
            
            # Generación de embeddings finales
            with torch.no_grad():
                model.eval()
                _, final_embeddings = model(gram, geo, text, edge_index)
                embeddings_df = pd.DataFrame(
                    final_embeddings.cpu().numpy(),
                    index=self.processor.features['grammatical'].index
                )
            
            return embeddings_df
        
        except Exception as e:
            logging.error(f"Error crítico en generación: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        logging.info("Iniciando proceso de generación de embeddings...")
        generator = EnhancedEmbeddingGenerator("grambank_completo.ttl")
        embeddings_df = generator.generate(epochs=500)
        embeddings_df.to_csv("embeddings_from_ttl.csv", index_label="glottocode")
        logging.info("Proceso completado exitosamente!")
        logging.info(f"Embeddings generados: {embeddings_df.shape}")
    except Exception as e:
        logging.error(f"Error en ejecución principal: {str(e)}")