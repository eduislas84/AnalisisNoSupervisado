import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Importar el Dataset con los datos de entrenamiento

df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")
print(df_datos_clientes.info())
print(df_datos_clientes.head())

# 2. Convertir el Dataframe a un Arrat de Numpy
X = df_datos_clientes.values
#print(X)

# 3. Entrenar el modelo
modelo = KMeans(n_clusters=2, random_state=1234,n_init=10)
modelo.fit(X)

# 4. Analisis del modelo
df_datos_clientes['cluster'] = modelo.labels_
analisis = df_datos_clientes.groupby('cluster').mean()
print(analisis)