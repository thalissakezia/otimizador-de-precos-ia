import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import sys
import subprocess
import streamlit as st

def verify_environment():
    try:
        import numpy as np
        import pandas as pd
        if np.__version__ != "1.26.4" or pd.__version__ != "2.2.2":
            st.warning("Versões incorretas detectadas. Reinstalando dependências...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            st.rerun()
    except ImportError:
        st.error("Dependências faltando. Instalando automaticamente...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        st.rerun()

verify_environment()

from langchain_community.chat_models import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

# ----------------------
# Configurações Iniciais
# ----------------------
st.set_page_config(
    page_title="Otimizador de Preços IA",
    layout="wide",
    page_icon="🧠"
)

# ----------------------
# Funções Utilitárias
# ----------------------

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo de otimização de preços"""
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
    """Inicializa o modelo de chat Groq"""
    try:
        groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not groq_key:
            st.error("🚨 Chave API GROQ não encontrada. Configure GROQ_API_KEY.")
            st.stop()
        return ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_key)
    except Exception as e:
        st.error(f"Erro ao inicializar o chat: {str(e)}")
        st.stop()

def processar_dataframe(df):
    """Processa o dataframe e identifica colunas relevantes"""
    col_mapping = {
        'preco_custo': ['custo', 'preço_custo', 'cost'],
        'vendas': ['vendas', 'sales', 'quantidade'],
        'concorrencia': ['concorr', 'competitor', 'concorrente'],
        'categoria': ['categoria', 'tipo', 'category', 'type'],
        'data': ['data', 'date', 'datetime']
    }
    
    return {
        target: next((c for c in df.columns if any(opt in c.lower() for opt in options), None)
        for target, options in col_mapping.items()
    }

# ----------------------
# Interface do App
# ----------------------

def main():
    st.title("🧠 Otimizador de Preços com IA")
    st.caption("Análise de preços + Chat Inteligente (via Groq)")

    # Upload do CSV
    uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.toast("CSV carregado com sucesso!", icon="✅")
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {str(e)}")
            st.stop()

        # Mostrar dados
        with st.expander("📊 Visualizar dados", expanded=True):
            st.dataframe(df.head(3))

        # Processar colunas
        cols = processar_dataframe(df)
        missing_cols = [k for k, v in cols.items() if v is None]
        
        if missing_cols:
            st.warning(
                f"⚠️ Colunas não encontradas: {', '.join(missing_cols)}\n"
                "Certifique-se de que seu CSV contém colunas com nomes similares."
            )
            st.stop()

        # Carregar modelo
        modelo = carregar_modelo()
        if modelo is None:
            st.stop()

        # Pré-processamento
        try:
            df[cols['data']] = pd.to_datetime(df[cols['data']], errors="coerce")
            df["dia_do_ano"] = df[cols['data']].dt.dayofyear
            df[cols['categoria']] = df[cols['categoria']].astype('category').cat.codes

            # Previsões
            X = df[[cols['preco_custo'], cols['vendas'], cols['concorrencia'], cols['categoria'], "dia_do_ano"]]
            df["preco_sugerido"] = modelo.predict(X)

            # Resultados
            st.success("✅ Preços sugeridos calculados!")
            st.dataframe(df[[cols['preco_custo'], cols['concorrencia'], "preco_sugerido"]].head())

            # Download
            csv_resultado = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Baixar CSV com preços sugeridos",
                data=csv_resultado,
                file_name=f"precos_sugeridos_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Chat IA
            st.divider()
            st.subheader("💬 Pergunte sobre seus dados")
            
            if "agent" not in st.session_state:
                with st.spinner("Inicializando IA..."):
                    try:
                        chat_model = inicializar_chat_model()
                        st.session_state.agent = create_pandas_dataframe_agent(
                            chat_model, df, verbose=False
                        )
                    except Exception as e:
                        st.error(f"Falha ao iniciar o agente: {str(e)}")
                        st.stop()

            if prompt := st.chat_input("Ex: Qual produto tem o maior markup?"):
                with st.chat_message("user"):
                    st.write(prompt)
                
                with st.spinner("Processando..."):
                    try:
                        resposta = st.session_state.agent.run(prompt)
                        with st.chat_message("assistant"):
                            st.write(resposta)
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")

        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
    else:
        st.info("📤 Envie um arquivo CSV para começar.")

if __name__ == "__main__":
    main()
