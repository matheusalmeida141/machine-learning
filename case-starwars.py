# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree

# %%
arvore = tree.DecisionTreeClassifier()

df = pd.read_parquet("data/dados_clones.parquet")

df.head()

# %%

df["Status "].replace({
    "Defeituoso" : 0,
    "Apto": 1
}, inplace = True)


# %%

X = df[["Massa(em kilos)"]]
y = df["Status "]


# %%
arvore.fit(X, y)


# %%

plt.figure(dpi=1000)
tree.plot_tree(arvore,
               max_depth= 3, 
               feature_names=["kg"],
               class_names=arvore.classes_,
               filled=True)
plt.savefig("starwars.png")
    # %%
