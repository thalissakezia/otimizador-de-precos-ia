import os
import streamlit as st
import pandas as pd
import joblib

from langchain_community.chat_models import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

# ----------------------
# Funções utilitárias
# ----------------------

@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load("modelo_otimizador_preco.pkl")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_resource
def inicializar_chat_model():
    groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not groq_key:
        st.error("🚨 Variável GROQ_API_KEY não encontrada.")
        st.stop()
    return ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_key)

# ----------------------
# Interface do App
# ----------------------

st.set_page_config(page_title="Otimizador de Preços IA", layout="wide")
st.title("🧠 Otimizador de Preços com IA + Chat Inteligente (via Groq)")

# Upload do CSV
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        st.stop()

    st.subheader("📊 Dados carregados")
    st.dataframe(df.head())

    modelo = carregar_modelo()
    if modelo is None:
        st.stop()

    colunas = df.columns.tolist()
    col_preco_custo = next((c for c in colunas if "custo" in c.lower()), None)
    col_vendas = next((c for c in colunas if "vendas" in c.lower()), None)
    col_concorrencia = next((c for c in colunas if "concorr" in c.lower()), None)
    col_categoria = next((c for c in colunas if "categoria" in c.lower() or "tipo" in c.lower()), None)
    col_data = next((c for c in colunas if "data" in c.lower()), None)

    if not all([col_preco_custo, col_vendas, col_concorrencia, col_categoria, col_data]):
        st.warning("⚠️ Use um CSV com colunas: preco_custo, vendas_30d, concorrente_preco_medio, categoria, data.")
    else:
        df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
        df["dia_do_ano"] = df[col_data].dt.dayofyear
        df[col_categoria] = df[col_categoria].astype('category').cat.codes

        X = df[[col_preco_custo, col_vendas, col_concorrencia, col_categoria, "dia_do_ano"]]
        df["preco_sugerido"] = modelo.predict(X)

        st.success("✅ Preços sugeridos calculados com sucesso!")
        st.dataframe(df[[col_preco_custo, "preco_sugerido"]].head())

        csv_resultado = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar CSV com preços sugeridos",
            data=csv_resultado,
            file_name="precos_sugeridos.csv",
            mime="text/csv"
        )

        st.divider()

        # Chat IA
        st.subheader("💬 Pergunte sobre seus dados")
        pergunta = st.text_input("Digite sua pergunta:")

        if pergunta:
            with st.spinner("💬 Consultando IA..."):
                try:
                    chat_model = inicializar_chat_model()
                    agent = create_pandas_dataframe_agent(chat_model, df, verbose=False)
                    resposta = agent.run(pergunta)
                    st.markdown(f"**Resposta:** {resposta}")
                except Exception as e:
                    st.error(f"Erro ao responder: {e}")
else:
    st.info("📥 Envie um arquivo CSV para começar.")
