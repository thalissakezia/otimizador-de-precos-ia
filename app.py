import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

from langchain_community.chat_models import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

# ----------------------
# Funções utilitárias
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
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o modelo: {str(e)}")
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
    colunas = df.columns.tolist()
    
    # Mapeamento de colunas com fallback
    col_mapping = {
        'preco_custo': ['custo', 'preço_custo', 'cost'],
        'vendas': ['vendas', 'sales', 'quantidade'],
        'concorrencia': ['concorr', 'competitor', 'concorrente'],
        'categoria': ['categoria', 'tipo', 'category', 'type'],
        'data': ['data', 'date', 'datetime']
    }
    
    found_cols = {}
    for target, options in col_mapping.items():
        found_cols[target] = next((c for c in colunas if any(opt in c.lower() for opt in options)), None)
    
    return found_cols

# ----------------------
# Interface do App
# ----------------------

def main():
    st.set_page_config(
        page_title="Otimizador de Preços IA", 
        layout="wide",
        page_icon="🧠"
    )
    
    st.title("🧠 Otimizador de Preços com IA")
    st.caption("Análise de preços + Chat Inteligente (via Groq)")
    
    # Upload do CSV
    with st.expander("📂 Envie seu arquivo CSV", expanded=True):
        uploaded_file = st.file_uploader(
            "Selecione seu arquivo de dados", 
            type=["csv"],
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.toast("Arquivo CSV carregado com sucesso!", icon="✅")
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {str(e)}")
            st.stop()
        
        # Mostrar dados
        with st.expander("📊 Visualizar dados carregados"):
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Total de registros: {len(df)}")
        
        # Carregar modelo
        modelo = carregar_modelo()
        if modelo is None:
            st.stop()
        
        # Processar colunas
        cols = processar_dataframe(df)
        
        # Verificar colunas obrigatórias
        required_cols = ['preco_custo', 'vendas', 'concorrencia', 'categoria', 'data']
        missing_cols = [col for col in required_cols if not cols[col]]
        
        if missing_cols:
            st.warning(
                f"⚠️ Não foram encontradas colunas para: {', '.join(missing_cols)}. "
                "Certifique-se de que seu CSV contém colunas com esses nomes ou similares."
            )
            st.stop()
        
        # Pré-processamento
        try:
            df[cols['data']] = pd.to_datetime(df[cols['data']], errors="coerce")
            df["dia_do_ano"] = df[cols['data']].dt.dayofyear
            df[cols['categoria']] = df[cols['categoria']].astype('category').cat.codes
            
            # Preparar features
            X = df[[
                cols['preco_custo'], 
                cols['vendas'], 
                cols['concorrencia'], 
                cols['categoria'], 
                "dia_do_ano"
            ]]
            
            # Fazer previsões
            df["preco_sugerido"] = modelo.predict(X)
            
            # Mostrar resultados
            st.success("✅ Preços sugeridos calculados com sucesso!")
            
            with st.expander("🔍 Comparação de preços"):
                st.dataframe(df[[
                    cols['preco_custo'], 
                    cols['concorrencia'], 
                    "preco_sugerido"
                ]].head(), use_container_width=True)
            
            # Botão de download
            csv_resultado = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Baixar CSV com preços sugeridos",
                data=csv_resultado,
                file_name=f"precos_sugeridos_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )
            
            # Chat IA
            st.divider()
            st.subheader("💬 Pergunte sobre seus dados")
            
            # Inicializar chat apenas quando necessário
            if "chat_initialized" not in st.session_state:
                with st.spinner("Inicializando agente de chat..."):
                    try:
                        st.session_state.chat_model = inicializar_chat_model()
                        st.session_state.agent = create_pandas_dataframe_agent(
                            st.session_state.chat_model, 
                            df, 
                            verbose=False
                        )
                        st.session_state.chat_initialized = True
                    except Exception as e:
                        st.error(f"Falha ao inicializar o agente: {str(e)}")
                        st.stop()
            
            # Interface de chat
            pergunta = st.chat_input("Digite sua pergunta sobre os dados...")
            
            if pergunta:
                with st.spinner("💭 Processando sua pergunta..."):
                    try:
                        resposta = st.session_state.agent.run(pergunta)
                        with st.chat_message("assistant"):
                            st.markdown(resposta)
                    except Exception as e:
                        st.error(f"Erro ao processar pergunta: {str(e)}")
        
        except Exception as e:
            st.error(f"Erro durante o processamento: {str(e)}")
            st.stop()
    else:
        st.info("ℹ️ Por favor, envie um arquivo CSV para começar a análise.")

if __name__ == "__main__":
    main()
