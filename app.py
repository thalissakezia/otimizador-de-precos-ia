
import streamlit as st
import pandas as pd
import joblib

st.title("🧠 Otimizador de Preços com IA")
st.write("Envie seu arquivo CSV com os dados do produto e veja a sugestão de preço ideal.")

uploaded_file = st.file_uploader("Escolha seu arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Dados carregados:")
    st.dataframe(df.head())

    colunas_obrigatorias = [
        'produto', 'categoria', 'preco_custo', 'vendas_30d', 'concorrente_preco_medio', 'data'
    ]

    if all(col in df.columns for col in colunas_obrigatorias):
        df['data'] = pd.to_datetime(df['data'])
        df['dia_do_ano'] = df['data'].dt.dayofyear
        df['categoria'] = df['categoria'].astype('category').cat.codes

        X = df[['preco_custo', 'vendas_30d', 'concorrente_preco_medio', 'categoria', 'dia_do_ano']]

        modelo = joblib.load('modelo_otimizador_preco.pkl')
        precos_sugeridos = modelo.predict(X)

        df['preco_sugerido'] = precos_sugeridos

        st.success("✅ Preços sugeridos calculados com sucesso!")
        st.dataframe(df[['produto', 'preco_custo', 'preco_sugerido']].head())

        csv_resultado = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar arquivo com preços sugeridos",
            data=csv_resultado,
            file_name='precos_sugeridos.csv',
            mime='text/csv'
        )
    else:
        st.error("❌ Seu arquivo não tem todas as colunas obrigatórias.")
        st.write("Colunas obrigatórias:", colunas_obrigatorias)
