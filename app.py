import os
import streamlit as st
import pandas as pd
import joblib

from langchain_community.chat_models import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

# Configuração da chave Groq via variável de ambiente
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 Variável GROQ_API_KEY não encontrada. Configure no Streamlit Secrets!")
    st.stop()

# Inicializa o modelo de chat com Groq
chat_model = ChatGroq(
    model_name="mixtral-8x7b-32768",  # ou "llama3-8b-8192"
    groq_api_key=GROQ_API_KEY
)

# Título do App
st.title("🧠 Otimizador de Preços com IA + Chat Inteligente (via Groq)")

# Upload de CSV
uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    # Carrega o CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dados carregados")
    st.dataframe(df.head())

    # Cria o agente LangChain com IA para entender os dados
    agent = create_pandas_dataframe_agent(chat_model, df, verbose=False)

    st.info("⌛ Interpretan
