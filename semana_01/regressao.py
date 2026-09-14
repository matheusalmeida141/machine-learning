# %%

import pandas as pd
from sklearn import linear_model
from sklearn import tree
df = pd.read_excel("data/dados_cerveja_nota.xlsx")

df.head()
# %%
X = df[["cerveja"]]
y = df["nota"]

reg = linear_model.LinearRegression()
reg.fit(X, y)
reg.coef_, reg.intercept_
predict_reg = reg.predict(X.drop_duplicates())


arvore_full = tree.DecisionTreeRegressor(random_state = 42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d1 = tree.DecisionTreeRegressor(random_state = 42, max_depth = 1)
arvore_d1.fit(X, y)
predict_arvore_d1 = arvore_d1.predict(X.drop_duplicates())


arvore_d2 = tree.DecisionTreeRegressor(random_state = 42, max_depth = 2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())



# %%

import matplotlib.pyplot as plt

plt.plot(X["cerveja"], y, 'o')
plt.title("Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates(), predict_reg)

plt.plot(X.drop_duplicates(), predict_arvore_full)

plt.plot(X.drop_duplicates(), predict_arvore_d2)

plt.plot(X.drop_duplicates(), predict_arvore_d1)

plt.legend([
            "observado", 
            f"y = {reg.coef_[0]:.3f}*x + {reg.intercept_:.3f}", 
            "arvore full", 
            "arvore d1",
            "arvore d2"])
# %%
