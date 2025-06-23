import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("dados_exemplo.csv")
df['categoria'] = df['categoria'].astype('category').cat.codes
X = df[['preco_custo', 'concorrente_preco_medio', 'categoria']]
y = df['preco_atual']

modelo = RandomForestRegressor()
modelo.fit(X, y)
joblib.dump(modelo, "modelo_otimizador_preco.pkl")
print("Modelo salvo como modelo_otimizador_preco.pkl")