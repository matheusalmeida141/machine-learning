# %%

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("data/dados_cerveja_nota.xlsx")


df["aprovado"] = (df["nota"] >= 5).astype(int)


df.head()

# %%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

import numpy as np
X = df[["cerveja"]]
y = df["aprovado"]


reg_logi = linear_model.LogisticRegression(penalty=None,
                                           fit_intercept=True)
reg_logi.fit(X, y)
logi_predict = reg_logi.predict(X.drop_duplicates())
logi_prob = reg_logi.predict_proba(X.drop_duplicates())[:,1]

reg_tree_full = tree.DecisionTreeClassifier(random_state = 42)
reg_tree_full.fit(X, y)
tree_predict_full = reg_tree_full.predict(X.drop_duplicates())
tree_proba_full = reg_tree_full.predict_proba(X.drop_duplicates())[:,1]

reg_tree_d2 = tree.DecisionTreeClassifier(random_state = 42, max_depth = 2)
reg_tree_d2.fit(X, y)
tree_predict_d2 = reg_tree_d2.predict(X.drop_duplicates())
tree_proba_d2 =  reg_tree_d2.predict_proba(X.drop_duplicates())[:,1]

nb = naive_bayes.GaussianNB()
nb.fit(X, y)
nb_predict = nb.predict(X.drop_duplicates())
nb_proba = nb.predict_proba(X.drop_duplicates())[:,1]



# %%
plt.figure(dpi=500)
plt.plot(df["cerveja"], df["aprovado"], 'o', color="royalblue", label="observado")
plt.grid()
plt.title("Cerveja vs Aprovado")
plt.xlabel("nº cerveja")
plt.ylabel("aprovado")

plt.plot(X.drop_duplicates(), logi_predict, color="tomato", label="regressão logitistica")
plt.plot(X.drop_duplicates(), logi_prob, color="green", linestyle="dotted", label="probabilidade regressão logistica")
# plt.plot(X.drop_duplicates(), tree_predict_full, color="green", label="arvore de decisão")
# plt.plot(X.drop_duplicates(), tree_proba_full, color="skyblue", label="probabilidade regressão logistica")
plt.plot(X.drop_duplicates(), tree_predict_d2, color="blue", label="arvore de decisão")
plt.plot(X.drop_duplicates(), tree_proba_d2, linestyle="dotted", color="darkblue", label="probabilidade arvore de decisao")

plt.plot(X.drop_duplicates(), nb_predict, color="pink", label="naive bayes")
plt.plot(X.drop_duplicates(), nb_proba, linestyle="dotted", color="pink", label="probabilidade naive bayes")


plt.legend()
plt.xlim(0,np.max(X))
plt.hlines(0.5, xmin=0, xmax=9, linestyles='--', colors='black')

# %%
