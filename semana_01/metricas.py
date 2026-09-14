# %%

import pandas as pd

df = pd.read_csv("data/comunidade.csv")

df = df.replace({"Sim" : 1, "Não" : 0})

dummy_vars = [
"Como conheceu o Téo Me Why?",
"Quantos cursos acompanhou do Téo Me Why?",
"Estado que mora atualmente",
"Área de Formação",
"Tempo que atua na área de dados",
"Posição da cadeira (senioridade)"
]

num_vars = [
    "Curte games?",
    "Curte futebol?",
    "Curte livros?",
    "Curte jogos de tabuleiro?",
    "Curte jogos de fórmula 1?",	
    "Curte jogos de MMA?",
    "Idade"
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise["happy"]= df["Você se considera uma pessoa feliz?"].copy().astype(int)
df_analise.head()

from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

arvore= tree.DecisionTreeClassifier(random_state=42,
                                    min_samples_leaf=5,

                                    )

nb = naive_bayes.GaussianNB()

logi = linear_model.LogisticRegression(fit_intercept= True)

features = df_analise.columns[:-1].tolist()
X = df_analise[features]
y = df_analise["happy"]

arvore.fit(X, y)
nb.fit(X, y)
logi.fit(X, y)

arvore_pre = arvore.predict(X)
nb_predict = nb.predict(X)
logi_predict = logi.predict(X)


df_predict = df_analise[['happy']]
df_predict['arvore_predict'] = arvore_pre
df_predict["arvore_proba"] = arvore.predict_proba(X)[:,1]
df_predict["nb_predict"] = nb_predict
df_predict["nb_proba"] = nb.predict_proba(X)[:,1]
df_predict["logi_predict"] = logi_predict
df_predict["logi_proba"] = logi.predict_proba(X)[:,1]

(df_predict["happy"] == df_predict["arvore_predict"]).mean()


pd.crosstab(df_predict['happy'], df_predict['arvore_predict'])

df_predict.query("arvore_predict == 1").sum()/160

from sklearn import metrics

metrics.accuracy_score(df_predict["happy"], df_predict["arvore_predict"])
metrics.precision_score(df_predict["happy"], df_predict["arvore_predict"])
metrics.recall_score(df_predict["happy"], df_predict["arvore_predict"])
roc_arvore = metrics.roc_curve(df_predict["happy"], df_predict["arvore_proba"])
auc_arvore = metrics.roc_auc_score(df_predict["happy"], df_predict["arvore_proba"])

roc_nb = metrics.roc_curve(df_predict["happy"], df_predict["nb_proba"])
auc_naive = metrics.roc_auc_score(df_predict["happy"], df_predict["nb_proba"])


roc_logi = metrics.roc_curve(df_predict["happy"], df_predict["logi_proba"])
auc_logi = metrics.roc_auc_score(df_predict["happy"], df_predict["logi_proba"])

import matplotlib.pyplot as plt

plt.plot(roc_arvore[0], roc_arvore[1], 'o-')
plt.plot(roc_nb[0], roc_nb[1], '-o')
plt.plot(roc_logi[0], roc_logi[1], '-o')
plt.legend([f"arvore: {auc_arvore:.2f}", 
            f"naive bayes: {auc_naive:.2f}", 
            f"logi: {auc_logi:.2f}"])
# %%

pd.Series({"model": logi, "features":features}).to_pickle("model_feliz.pkl")

# %%
