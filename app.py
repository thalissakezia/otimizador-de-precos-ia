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

    st.info("⌛ Interpretando e sugerindo preços, aguarde...")

    # Tenta carregar o modelo de precificação
    try:
        modelo = joblib.load("modelo_otimizador_preco.pkl")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

    # Detecta colunas relevantes automaticamente
    colunas = df.columns.tolist()
    st.write(f"Colunas detectadas no CSV: {colunas}")

    col_preco_custo = next((c for c in colunas if "custo" in c.lower()), None)
    col_vendas = next((c for c in colunas if "vendas" in c.lower()), None)
    col_concorrencia = next((c for c in colunas if "concorr" in c.lower()), None)
    col_categoria = next((c for c in colunas if "categoria" in c.lower() or "tipo" in c.lower()), None)
    col_data = next((c for c in colunas if "data" in c.lower()), None)

    if not all([col_preco_custo, col_vendas, col_concorrencia, col_categoria, col_data]):
        st.warning("⚠️ Não foram encontradas todas as colunas necessárias. Use um CSV com colunas semelhantes a: preco_custo, vendas_30d, concorrente_preco_medio, categoria, data.")
    else:
        # Pré-processamento para o modelo
        df[col_data] = pd.to_datetime(df[col_data])
        df['dia_do_ano'] = df[col_data].dt.dayofyear
        df[col_categoria] = df[col_categoria].astype('category').cat.codes

        X = df[[col_preco_custo, col_vendas, col_concorrencia, col_categoria, 'dia_do_ano']]
        precos_sugeridos = modelo.predict(X)
        df['preco_sugerido'] = precos_sugeridos

        st.success("✅ Preços sugeridos calculados com sucesso!")
        st.dataframe(df[[col_preco_custo, 'preco_sugerido']].head())

        # Botão para baixar CSV com os preços sugeridos
        csv_resultado = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar CSV com preços sugeridos",
            data=csv_resultado,
            file_name='precos_sugeridos.csv',
            mime='text/csv'
        )

    # Interface do Chat IA
    st.subheader("💬 Pergunte sobre seus dados e os preços sugeridos")
    pergunta = st.text_input("Digite sua pergunta:")

    if pergunta:
        try:
            resposta = agent.run(pergunta)
            st.markdown(f"**Resposta:** {resposta}")
        except Exception as e:
            st.error(f"Erro ao processar pergunta: {e}")
else:
    st.info("Envie um arquivo CSV para começar.")
