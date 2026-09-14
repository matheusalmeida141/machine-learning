# %%
import pandas as pd

df = pd.read_csv("../data/abt_churn.csv")
df.head()
# %%


#safra
oot = df[df["dtRef"] == df["dtRef"].max()].copy()
oot

# %%


#conjunto de dados para treino, não incluido safra
df_train = df[ df["dtRef"] < df["dtRef"].max()].copy()
df_train["dtRef"]


# %%


#definindo as features (variáveis) e target
features = df_train.columns[2:-1]
target = "flagChurn"

X, y = df_train[features], df_train[target]

# %%

from sklearn import model_selection

#definindo treino e teste

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y
                                                                    )
# %%

#comparar se as duas target estão parecidas
print(y_train.mean(), y_test.mean())
# %%

#ver se tem missing
X_train.isna().sum().sort_values(ascending=False)


# %%

#análise bivariada
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(["mean", "median"]).T

sumario["diff_abs"] = sumario[0] - sumario[1]
sumario["diff_relativa"] = (sumario[0])/sumario[1]
sumario.sort_values(by=["diff_relativa"], ascending=False)


# %%
from sklearn import tree
#import matplotlib.pyplot as plt

#rodando arvore para ver qual features são mais importantes
#plt.figure(dpi= 800)
arvore = tree.DecisionTreeClassifier(random_state=42 )#, max_depth=5)
arvore.fit(X_train, y_train)
# tree.plot_tree(arvore, feature_names=X_train.columns,
#                 filled=True,
#                 class_names=[str(i) for i in arvore.classes_])

features_importance = pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()

features_importance["freqAcumulada"] = features_importance[0].cumsum()
features_importance[features_importance["freqAcumulada"] < 0.96]
# %%
# Modify

#melhores features
best_features =  features_importance[features_importance["freqAcumulada"] < 0.96]["index"].to_list()
best_features

# %%
#discretizar a variavel com tree
from feature_engine.discretisation import DecisionTreeDiscretiser

disc = DecisionTreeDiscretiser(regression=False,
                            bin_output="bin_number",
                            cv=3,
                            variables = best_features
                            )
#fit
disc.fit(X_train[best_features], y_train)
#transform
X_train_t = disc.transform(X_train[best_features])
X_train_t
# %%

#Model

#passar o modelo utilizado, usou o regression
from sklearn import linear_model
reg = linear_model.LogisticRegression(
    #penalty=None,
    random_state=42,
    max_iter=10000
)

#fit
reg.fit(X_train_t, y_train)

# %%
#Metricas
from sklearn import metrics

y_train_predict = reg.predict(X_train_t)
y_train_proba   = reg.predict_proba(X_train_t)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print(f"Treino: Acurácia: {acc_train} \nAUC: {auc_train}")

# %%
# Conjunto de teste
X_test_t = disc.transform(X_test[best_features])

y_test_predict = reg.predict(X_test_t)
y_test_proba = reg.predict_proba(X_test_t)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print(f"Teste: Acurácia: {acc_test} \nAUC: {auc_test}")

# %%

# conjunto oot
X_oot_t = disc.transform(oot[best_features])

y_oot_predict = reg.predict(X_oot_t)
y_oot_proba = reg.predict_proba(X_oot_t)[:,1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
print(f"Teste: Acurácia: {acc_oot} \nAUC: {auc_oot}")

# %%
