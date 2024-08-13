#%%
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import colorama
from colorama import Back
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

Main_Dataset=pd.read_csv("heart_2020_cleaned.csv")
Main_Dataset = pd.DataFrame(Main_Dataset)
print(Main_Dataset.shape)
Main_Dataset.head(10)
def describe(Main_Dataset):
    
    
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    min_ = []
    max_ = []
    
    
    for item in Main_Dataset.columns:
        variables.append(item)
        dtypes.append(Main_Dataset[item].dtype)
        count.append(len(Main_Dataset[item]))
        unique.append(len(Main_Dataset[item].unique()))
        missing.append(Main_Dataset[item].isna().sum())
        
        if Main_Dataset[item].dtypes == 'float64' or Main_Dataset[item].dtypes == 'int64':
            min_.append(Main_Dataset[item].min())
            max_.append(Main_Dataset[item].max())
        else:
            min_.append('Str')
            max_.append('Str')
        

    output = pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing,
        'Min': min_,
        'Max': max_
    })    
        
    return output

desc_df = describe(Main_Dataset)
print (desc_df)
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
