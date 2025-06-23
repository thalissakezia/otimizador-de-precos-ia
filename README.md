# otimizador-de-precos-ia
# 🧠 Otimizador de Preços com IA para Pequenos Negócios

Este é um app online que usa **Inteligência Artificial** para sugerir o **preço ideal de venda** de produtos, com base em dados reais como custo, concorrência e volume de vendas.

📈 Ideal para pequenos negócios que querem aumentar lucros com decisões mais inteligentes!

---

## 🚀 Acesse o App

👉 [Clique aqui para testar o Otimizador de Preços](https://otimizador-de-precos-ia.streamlit.app)

---

## 📂 Como usar

1. Prepare seu arquivo CSV com os seguintes campos:

  produto, categoria, preco_custo, vendas_30d, concorrente_preco_medio, data

  
2. Faça upload do seu arquivo no app.
3. Veja os preços sugeridos com base nos seus dados.
4. Baixe o novo CSV com os preços otimizados!

---

## 🛠️ Tecnologias usadas

- Python
- Streamlit
- XGBoost
- Pandas
- Scikit-learn

---

## 🤖 Como funciona?

O app usa um modelo de Machine Learning (XGBoost) treinado para prever o **preço ideal de venda** com base em:

- Preço de custo
- Quantidade vendida nos últimos 30 dias
- Preço médio dos concorrentes
- Categoria do produto
- Época do ano (extraída da data)

---

## 👩‍💻 Rodar localmente (opcional)

```bash
git clone https://github.com/seu-usuario/otimizador-de-precos-ia.git
cd otimizador-de-precos-ia
pip install -r requirements.txt
streamlit run app.py

Feito com ❤️ para ajudar pequenos negócios a crescer com tecnologia.

