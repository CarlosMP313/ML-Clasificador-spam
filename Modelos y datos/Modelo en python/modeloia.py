# Importamos las librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re

# Descargar stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Cargar datos (ejemplo de estructura)
# Supongamos que tenemos un DataFrame llamado "df" con columnas ['email_text', 'label']
# donde 'email_text' es el texto del correo y 'label' indica 1 (malicioso) o 0 (seguro).
df = pd.read_csv('emails.csv')  # Reemplaza con el path de tu dataset
print(df.head())  # Muestra las primeras filas del DataFrame

# Limpieza de texto
def clean_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar URLs, caracteres especiales y números
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    # Eliminar palabras de parada
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Aplicar limpieza de texto
df['cleaned_text'] = df['email_text'].apply(clean_text)

# Dividir los datos en entrenamiento y prueba
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorización con TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Limitar a 1000 características para eficiencia
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predicción
y_pred = model.predict(X_test_tfidf)

# Evaluación del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:", classification_report(y_test, y_pred))
