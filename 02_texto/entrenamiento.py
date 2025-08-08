import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import csv


nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Leer los títulos de las noticias desde Noticias.csv
titulos_noticias = []
with open("Noticias.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row:  # Evitar filas vacías
            titulos_noticias.append(row[0])

spanish_stop_words = stopwords.words("spanish")
vectorizador = TfidfVectorizer(stop_words=spanish_stop_words)
X = vectorizador.fit_transform(titulos_noticias)
modelo = KMeans(n_clusters=10, random_state=1234, n_init=10)
modelo.fit(X)
joblib.dump(modelo, "modelo_texto.pkl")
print(f"Cluster {modelo.labels_}")
for i, texto in enumerate(titulos_noticias):
    print(f"{texto}: Cluster {modelo.labels_[i]}")