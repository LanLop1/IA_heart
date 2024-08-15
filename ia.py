#%%
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# Cargar el dataset
Main_Dataset = pd.read_csv("heart_2020_cleaned.csv")
print(Main_Dataset.shape)
# Eliminar filas con valores NaN
Main_Dataset = Main_Dataset.dropna(axis=0, how='any')  # Eliminar filas con al menos un NaN
# Columnas que se van a convertir de "Yes"/"No" a 1/0
print(Main_Dataset['Sex'].unique())
print(Main_Dataset['Sex'].dtype)
Main_Dataset['Sex'] = Main_Dataset['Sex'].astype(str)
Main_Dataset['Sex'] = Main_Dataset['Sex'].map({'Male': 1, 'Female': 0})



columns_to_convert = [
    'HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 
    'DiffWalking', 'Diabetic', 
    'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer'
]

# Convertir las columnas "Yes"/"No" a 1/0
for column in columns_to_convert:
    if column in Main_Dataset.columns:
        Main_Dataset[column] = Main_Dataset[column].map({'Yes': 1, 'No': 0})

# Para la columna "Sex", mapeamos "Male" a 1 y "Female" a 0


# Mapeo de categorías de edad a valores numéricos
age_category_mapping = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80 or older': 12
}
Main_Dataset['AgeCategory'] = Main_Dataset['AgeCategory'].map(age_category_mapping)

# Mapeo de categorías raciales a valores numéricos
race_mapping = {
    'White': 0,
    'Black': 1,
    'Asian': 2,
    'American Indian/Alaskan Native': 3,
    'Other': 4,
    'Hispanic': 5
}
Main_Dataset['Race'] = Main_Dataset['Race'].map(race_mapping)

# Mapeo de categorías de salud a valores numéricos
gen_health_mapping = {
    'Excellent': 0,
    'Very good': 1,
    'Good': 2,
    'Fair': 3,
    'Poor': 4
}
Main_Dataset['GenHealth'] = Main_Dataset['GenHealth'].map(gen_health_mapping)



# Verificamos las primeras filas para asegurarnos de que la conversión se realizó correctamente
print(Main_Dataset.head(10))

# Calcular las correlaciones
correlations = Main_Dataset.corr()['Stroke'].drop('Stroke')
correlations_df = correlations.to_frame().sort_values(by='Stroke', ascending=False)

# Crear el heatmap
plt.figure(figsize=(8, 12))
sns.heatmap(correlations_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlación de Variables con Stroke')
plt.show()

X = Main_Dataset[['DiffWalking', 'GenHealth', 'HeartDisease', 'AgeCategory','Diabetic','KidneyDisease','PhysicalActivity']]  # Selecciona las características
y = Main_Dataset['Stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=["Pred: No", "Pred: Si"], index=["Real: No", "Real: Si"]))
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns)

plt.show()

'''
frecuencias_muestra_1 = Target_0_data['PhysicalActivity'].value_counts(normalize=True) * 100
frecuencias_muestra_2 = Target_1_data['PhysicalActivity'].value_counts(normalize=True) * 100
comparacion = pd.DataFrame({
    'No sufren infarto': frecuencias_muestra_1,
    'Han sufrido infarto': frecuencias_muestra_2
}).fillna(0)
comparacion['Diferencia (%)'] = comparacion['No sufren infarto'] - comparacion['Han sufrido infarto']
print(comparacion)
column_name = 'Race'  # Cambia 'Sex' por el nombre de la columna que deseas inspeccionar


fig, ax = plt.subplots(figsize=(10, 6))

# Gráfica de barras para los porcentajes
comparacion[['No sufren infarto', 'Han sufrido infarto']].plot(kind='bar', ax=ax)
ax.set_title('Comparación de Porcentajes')
ax.set_ylabel('Porcentaje')
ax.set_xlabel('Condición')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()



'''

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
#%%