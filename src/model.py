import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Carregar os dados
tabela = pd.read_csv(R'data/advertising.csv')

# Visualização da correlação entre os investimentos
plt.figure(figsize=(8, 6))
sns.heatmap(tabela.corr(), cmap='Wistia', annot=True)
plt.title("Correlação entre Investimentos e Vendas")
plt.show()

# Separação das variáveis independentes (X) e dependente (Y)
x = tabela[['TV', 'Radio', 'Jornal']]
y = tabela['Vendas']

# Divisão dos dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=42)

# Instanciar os modelos
modelo_rl = LinearRegression()
modelo_ad = RandomForestRegressor()

# Treinamento dos modelos
modelo_rl.fit(x_treino, y_treino)
modelo_ad.fit(x_treino, y_treino)

# Fazer previsões
previsao_rl = modelo_rl.predict(x_teste)
previsao_ad = modelo_ad.predict(x_teste)

# Avaliação dos modelos
r2_rl = metrics.r2_score(y_teste, previsao_rl)
r2_ad = metrics.r2_score(y_teste, previsao_ad)

print(f'A porcentagem de acerto da Regressão Linear é {r2_rl:.4f}!')
print(f'A porcentagem de acerto da Árvore de Decisão é {r2_ad:.4f}!')

# Escolher o melhor modelo
melhor_ia = modelo_rl if r2_rl > r2_ad else modelo_ad

# Criar gráfico comparativo
tabela_prova = pd.DataFrame({
    'y_teste': y_teste.values,
    'Previsão RL': previsao_rl,
    'Previsão AD': previsao_ad
})

plt.figure(figsize=(15, 6))
sns.lineplot(data=tabela_prova)
plt.title("Comparação entre valores reais e previsões")
plt.show()

# Fazer previsões futuras
tabela_nova = pd.read_csv(R'data/novos.csv')
previsao_futura = melhor_ia.predict(tabela_nova)

# Exibir os dados formatados
tabela_nova['Previsão'] = previsao_futura  # Adiciona a coluna da previsão
print("\n📊 Previsão para novos dados:\n")
print(tabela_nova.to_string(index=False))  # Exibir tabela formatada sem índices
