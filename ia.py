#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

datos = pd.read_csv("hearth.csv")
datos.head()
datos.describe()
# sb.countplot(x="Survived", data=datos)
# sb.countplot(x="Survived", data=datos, hue="Sex")
# datos.isna().sum()
# sb.displot(x="Age", data=datos)
# datos["Age"]
# datos["Age"].mean()
# datos["Age"].fillna(datos["Age"].mean())
# datos["Age"] = datos["Age"].fillna(datos["Age"].mean())
# datos["Age"]
# datos.isna().sum()
# datos = datos.drop(["Cabin"], axis=1)
# datos["Embarked"].value_counts()
# datos = datos.dropna()
# datos.head()