import os
import streamlit as st
import pandas as pd
import joblib

from langchain_community.chat_models import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

# 🔐 Verificação da chave da API do Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 Variável GROQ_API_KEY não encontrada. Configure-a no ambiente ou em `.streamlit/secrets.toml`.")
    st.stop()

# Inicialização do modelo de chat
try:
    chat_model = ChatGroq(
        model_name="mixtral-8x7b-32768",  # ou "llama3-8b-8192"
        groq_api_key=GROQ_API_KEY
    )
except Exception as e:
    st.error(f"❌ Erro ao inicializar modelo do Groq: {e}")
    st.stop()

st.title("🧠 Otimizador de Preços com IA + Chat Inteligente")

# Upload do CSV
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        st.stop()

    st.subheader("📊 Dados carregados")
    st.dataframe(df.head())

    try:
        agent = create_pandas_dataframe_agent(chat_model, df, verbose=False)
    except Exception as e:
        st.error(f"Erro ao criar agente LangChain: {e}")
        st.stop()

    st.info("⌛ Interpretando dados e calculando preços...")

    try:
        modelo = joblib.load("modelo_otimizador_preco.pkl")
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        st.stop()

    # Identificação de colunas
    colunas = df.columns.tolist()
    col_preco_custo = next((c for c in colunas if "custo" in c.lower()), None)
    col_vendas = next((c for c in colunas if "vendas" in c.lower()), None)
    col_concorrencia = next((c for c in colunas if "concorr" in c.lower()), None)
    col_categoria = next((c for c in colunas if "categoria" in c.lower() or "tipo" in c.lower()), None)
    col_data = next((c for c in colunas if "data" in c.lower()), None)

    if not all([col_preco_custo, col_vendas, col_concorrencia, col_categoria, col_data]):
        st.warning("⚠️ CSV deve conter colunas como: preco_custo, vendas_30d, concorrente_preco_medio, categoria e data.")
    else:
        try:
            df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
            df['dia_do_ano'] = df[col_data].dt.dayofyear
            df[col_categoria] = df[col_categoria].astype('category').cat.codes

            X = df[[col_preco_custo, col_vendas, col_concorrencia, col_categoria, 'dia_do_ano']]
            df['preco_sugerido'] = modelo.predict(X)

            st.success("✅ Preços sugeridos calculados com sucesso!")
            st.dataframe(df[[col_preco_custo, 'preco_sugerido']].head())

            csv_resultado = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Baixar CSV com preços sugeridos",
                data=csv_resultado,
                file_name="precos_sugeridos.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao processar os dados: {e}")

    # Chat com IA
    st.subheader("💬 Faça perguntas sobre seus dados")
    pergunta = st.text_input("Digite sua pergunta:")

    if pergunta:
        try:
            resposta = agent.run(pergunta)
            st.markdown(f"**Resposta:** {resposta}")
        except Exception as e:
            st.error(f"Erro ao responder: {e}")
else:
    st.info("📥 Envie um arquivo CSV para começar.")
