import json
import warnings
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Suprimir advertencias de depreciación
warnings.simplefilter("ignore", category=FutureWarning)

# Cargar variables de entorno
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_API_TOKEN")

# Cargar el JSON generado desde el TTL
json_file_path = "grambank_simple.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    languages = json.load(f)

# Extraer información textual de las lenguas para embeddings
documents = []
language_data = {}
for label, data in languages.items():
    glottocode = data.get("glottocode", "Desconocido")
    family = data.get("family", "Desconocida")
    present_features = data.get("present_features", {})
    absent_features = data.get("absent_features", {})
    
    description = f"Lengua: {label.capitalize()}\nGlottocode: {glottocode}\nFamilia: {family}\nFuente: Grambank"
    documents.append(description)
    language_data[label.lower()] = (description, present_features, absent_features, family)

# Inicializar el modelo de embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Comprobar si FAISS ya está guardado
faiss_db_path = "quechua_rag.index"
if os.path.exists(faiss_db_path + ".pkl"):
    with open(faiss_db_path + ".pkl", "rb") as f:
        vectorstore = pickle.load(f)
else:
    vectorstore = FAISS.from_texts(documents, embedding_model)
    vectorstore.save_local(faiss_db_path)
    with open(faiss_db_path + ".pkl", "wb") as f:
        pickle.dump(vectorstore, f)

# Configurar el endpoint de Hugging Face para Mixtral 8x7B
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3,
    max_new_tokens=350
)

# Configurar el sistema RAG con ConversationalRetrievalChain
retriever = vectorstore.as_retriever()
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Mantener historial de conversación
chat_history = []

def generate_analysis_paragraph(label, family, present_features, absent_features, question):
    """
    Genera un análisis basado en los rasgos presentes y ausentes de la lengua.
    """
    present_list = list(present_features.items())[:10]
    absent_list = list(absent_features.items())[:10]
    
    feature_summary = "\n".join([f"- {desc}: Sí" for _, desc in present_list]) + "\n" + "\n".join([f"- {desc}: No" for _, desc in absent_list])
    
    explanation_prompt = (
        f"La lengua {label.capitalize()}, perteneciente a la familia {family}, presenta las siguientes características lingüísticas:\n"
        f"{feature_summary}\n"
        f"Basado en esta información, proporciona un análisis detallado de cómo estos rasgos influyen en la gramática, morfología y sintaxis de la lengua. "
        f"Incluye ejemplos concretos de su impacto en la estructura lingüística."
    )
    
    return llm.invoke(explanation_prompt)

def ask_question(question):
    global chat_history
    response = qa_chain.invoke({"question": f"Responde en español: {question}", "chat_history": chat_history})
    chat_history.append((question, response["answer"]))  # Agregar pregunta-respuesta al historial
    
    # Buscar información relevante en el JSON
    analysis_paragraph = ""
    for key, (value, present_features, absent_features, family) in language_data.items():
        if key in question.lower():
            analysis_paragraph = generate_analysis_paragraph(key.capitalize(), family, present_features, absent_features, question)
            break
    
    # Construcción estructurada de la respuesta
    respuesta_ordenada = "\n".join([
        "--- RESPUESTA ---",
        response["answer"],
        analysis_paragraph if analysis_paragraph else "No se encontraron características detalladas en la base de datos."
    ])
    
    return respuesta_ordenada

# Ejemplo de conversación
while True:
    query = input("Haz una pregunta (o escribe 'salir' para terminar): ")
    if query.lower() == "salir":
        break
    response = ask_question(query)
    print(response)
