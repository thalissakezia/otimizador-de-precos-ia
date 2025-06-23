import os
import streamlit as st
import pandas as pd
import joblib
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

# Configuração da chave OpenAI oculta via variável de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🚨 Variável OPENAI_API_KEY não encontrada. Configure no Streamlit Secrets!")
    st.stop()

# Inicializa o modelo Chat da OpenAI via LangChain
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

st.title("🧠 Otimizador de Preços com IA + Chat Inteligente")

uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    # Carrega CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dados carregados")
    st.dataframe(df.head())

    # Cria agente LangChain para perguntas sobre dados
    agent = create_pandas_dataframe_agent(chat_model, df, verbose=False)

    # Aqui você pode colocar sua lógica de interpretação / pré-processamento dos dados
    st.info("⌛ Interpretando e sugerindo preços, aguarde...")

    # Exemplo simples: tenta carregar modelo e prever preços
    try:
        modelo = joblib.load("modelo_otimizador_preco.pkl")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

    # Para simplificar, tentaremos usar as colunas mais parecidas possíveis
    # Vamos pedir para a IA identificar colunas-chave (exemplo simples)
    colunas = df.columns.tolist()
    st.write(f"Colunas detectadas no CSV: {colunas}")

    # Você pode expandir aqui para usar a IA para mapear as colunas, mas para já
    # tentaremos usar nomes padrão, se existirem
    col_preco_custo = next((c for c in colunas if "custo" in c.lower()), None)
    col_vendas = next((c for c in colunas if "vendas" in c.lower()), None)
    col_concorrencia = next((c for c in colunas if "concorr" in c.lower()), None)
    col_categoria = next((c for c in colunas if "categoria" in c.lower() or "tipo" in c.lower()), None)
    col_data = next((c for c in colunas if "data" in c.lower()), None)

    if not all([col_preco_custo, col_vendas, col_concorrencia, col_categoria, col_data]):
        st.warning("⚠️ Não foram encontradas todas as colunas necessárias para previsão automática. Use um CSV com colunas semelhantes a: preco_custo, vendas_30d, concorrente_preco_medio, categoria, data.")
    else:
        # Prepara os dados para o modelo
        df[col_data] = pd.to_datetime(df[col_data])
        df['dia_do_ano'] = df[col_data].dt.dayofyear
        df[col_categoria] = df[col_categoria].astype('category').cat.codes

        X = df[[col_preco_custo, col_vendas, col_concorrencia, col_categoria, 'dia_do_ano']]
        precos_sugeridos = modelo.predict(X)
        df['preco_sugerido'] = precos_sugeridos

        st.success("✅ Preços sugeridos calculados com sucesso!")
        st.dataframe(df[[col_preco_custo, 'preco_sugerido']].head())

        csv_resultado = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar CSV com preços sugeridos",
            data=csv_resultado,
            file_name='precos_sugeridos.csv',
            mime='text/csv'
        )

    # Chat para dúvidas sobre o CSV e preços
    st.subheader("💬 Pergunte sobre seus dados e preços")

    pergunta = st.text_input("Digite sua pergunta aqui:")

    if pergunta:
        resposta = agent.run(pergunta)
        st.markdown(f"**Resposta:** {resposta}")

else:
    st.info("Por favor, faça upload de um arquivo CSV para começar.")
