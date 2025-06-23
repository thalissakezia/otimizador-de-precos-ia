import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from langchain_community.chat_models import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

# Aviso sobre versão do Python
if not sys.version.startswith("3.11"):
    st.warning("Recomenda-se Python 3.11 para evitar problemas de compatibilidade.")

# Aviso sobre versões de pacotes
if not (np.__version__.startswith("1.") and pd.__version__.startswith("2.")):
    st.warning(f"Versões detectadas: numpy {np.__version__}, pandas {pd.__version__}. Se encontrar erros, ajuste o ambiente.")

@st.cache_resource
def carregar_modelo():
    try:
        modelo = joblib.load("modelo_otimizador_preco.pkl")
        st.toast("Modelo carregado com sucesso!", icon="✅")
        return modelo
    except FileNotFoundError:
        st.error("Arquivo do modelo não encontrado. Verifique se 'modelo_otimizador_preco.pkl' existe.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

@st.cache_resource
def inicializar_chat_model():
    try:
        groq_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
        if not groq_key:
            st.error("Chave da API GROQ não encontrada. Configure a variável de ambiente 'GROQ_API_KEY' ou adicione em st.secrets.")
            st.stop()
        return ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_key)
    except Exception as e:
        st.error(f"Erro ao inicializar o chat: {str(e)}")
        st.stop()

def processar_dataframe(df):
    col_mapping = {
        'preco_custo': ['custo', 'preço_custo', 'cost'],
        'vendas': ['vendas', 'sales', 'quantidade'],
        'concorrencia': ['concorr', 'competitor', 'concorrente'],
        'categoria': ['categoria', 'tipo', 'category', 'type'],
        'data': ['data', 'date', 'datetime']
    }
    return {
        target: next((c for c in df.columns if any(opt in c.lower() for opt in options)), None)
        for target, options in col_mapping.items()
    }

def main():
    st.title("🧠 Otimizador de Preços com IA")
    st.caption("Análise de preços + Chat Inteligente (via Groq)")

    uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.toast("CSV carregado com sucesso!", icon="✅")
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {str(e)}")
            st.stop()

        with st.expander("📊 Visualizar dados", expanded=True):
            st.dataframe(df.head(100))

        cols = processar_dataframe(df)
        missing_cols = [k for k, v in cols.items() if v is None]

        if missing_cols:
            st.error(f"Colunas não encontradas no CSV: {', '.join(missing_cols)}. Renomeie ou ajuste seu arquivo.")
            st.stop()

        modelo = carregar_modelo()
        if modelo is None:
            st.stop()

        try:
            df_proc = df.dropna()
        except Exception as e:
            st.error(f"Erro no pré-processamento: {str(e)}")
            st.stop()

        try:
            X = df_proc[[cols['preco_custo'], cols['concorrencia'], cols['categoria']]]
            pred = modelo.predict(X)
            st.success("Previsão realizada com sucesso!")
            st.write("Primeiras previsões:", pred[:10])
        except Exception as e:
            st.error(f"Erro ao rodar o modelo: {str(e)}")

        if st.checkbox("Ativar Chat IA"):
            agent = create_pandas_dataframe_agent(
                inicializar_chat_model(),
                df,
                verbose=True
            )
            pergunta = st.text_input("Pergunte algo sobre seus dados:")
            if pergunta:
                resposta = agent.run(pergunta)
                st.info(resposta)
    else:
        st.info("📤 Envie um arquivo CSV para começar.")

if __name__ == "__main__":
    main()
