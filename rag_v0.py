import json
import warnings
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Dict, Tuple, List

# Configuración inicial
warnings.simplefilter("ignore")
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_API_TOKEN")

# Cargar la interpretación de los rasgos desde codes.csv
codes_df = pd.read_csv("data/codes.csv")
codes_dict = dict(zip(codes_df["ID"], codes_df["Description"]))

class LanguageAnalyzer:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.language_data = self._load_data()
        self.vectorstore = self._setup_vectorstore()
        self.qa_chain = self._setup_qa_chain()
        self.chat_history: List[Tuple[str, str]] = []

    def _load_data(self) -> Dict:
        """Carga y procesa los datos lingüísticos"""
        with open("grambank_simple.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        return {
            label.lower(): {
                "metadata": {
                    "label": label,
                    "glottocode": data.get("glottocode", "Desconocido"),
                    "family": data.get("family", "Desconocida")
                },
                "features": {
                    "present": {feat: codes_dict.get(feat, feat) for feat in data.get("present_features", {})},
                    "absent": {feat: codes_dict.get(feat, feat) for feat in data.get("absent_features", {})}
                }
            } for label, data in raw_data.items()
        }

    def _setup_vectorstore(self):
        """Configura el almacén vectorial con optimización de memoria"""
        documents = [
            f"Lengua: {data['metadata']['label']}\n"
            f"Familia: {data['metadata']['family']}\n"
            f"Glottocode: {data['metadata']['glottocode']}"
            for data in self.language_data.values()
        ]
        
        return FAISS.from_texts(
            documents, 
            self.embedding_model,
            metadatas=[data["metadata"] for data in self.language_data.values()]
        )

    def _setup_qa_chain(self):
        """Configura la cadena de QA con plantilla personalizada"""
        prompt_template = """
        Eres un experto en lingüística. Responde en español de manera clara y fluida.
        Contexto: {context}
        Historial: {chat_history}
        Pregunta: {question}
        Respuesta:
        """
        
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            temperature=0.4,
            max_new_tokens=450,
            repetition_penalty=1.1,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            condense_question_prompt=PromptTemplate.from_template(prompt_template),
            verbose=False
        )

    def _generate_linguistic_analysis(self, lang_data: Dict) -> str:
        """Genera un análisis lingüístico basado en los rasgos presentes y ausentes."""
        present = lang_data["features"]["present"]
        absent = lang_data["features"]["absent"]
        
        analysis = (
            f"La lengua {lang_data['metadata']['label']} pertenece a la familia {lang_data['metadata']['family']} y se caracteriza por los siguientes rasgos lingüísticos:\n\n"
        )
        
        if present:
            analysis += "Rasgos presentes:\n" + "\n".join(
                [f"- {desc}: {codes_dict.get(feat, 'Descripción no disponible')}" for feat, desc in list(present.items())[:10]]
            ) + "\n\n"
            
        if absent:
            analysis += "Rasgos ausentes:\n" + "\n".join(
                [f"- {desc}: {codes_dict.get(feat, 'Descripción no disponible')}" for feat, desc in list(absent.items())[:10]]
            ) + "\n\n"
        
        return analysis

    def ask_question(self, question: str) -> str:
        """Procesa una pregunta y genera respuesta estructurada en español"""
        result = self.qa_chain.invoke({
            "question": f"Responde en español: {question}",
            "chat_history": self.chat_history
        })
        
        self.chat_history = [(result["question"], result["answer"])] + self.chat_history[:3]
        
        best_match = self.vectorstore.similarity_search(question, k=1)
        if best_match:
            lang_info = self.language_data[best_match[0].metadata["label"].lower()]
            analysis = self._generate_linguistic_analysis(lang_info)
            return f"{result['answer']}\n\nAnálisis Lingüístico:\n{analysis}"
        
        return result["answer"]

# Ejecución principal
if __name__ == "__main__":
    analyzer = LanguageAnalyzer()
    
    print("Sistema de Análisis Lingüístico - Grambank\n")
    while True:
        try:
            query = input("Tu pregunta (escribe 'salir' para terminar): ").strip()
            if query.lower() in ["salir", "exit"]:
                break
                
            response = analyzer.ask_question(query)
            print(f"\n{response}\n{'='*50}\n")
            
        except KeyboardInterrupt:
            print("\nSesión finalizada")
            break
