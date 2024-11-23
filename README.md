# Spam Classifier

Este proyecto implementa un clasificador de correos electrónicos (spam vs. ham) utilizando técnicas de procesamiento de texto y aprendizaje automático. El clasificador fue entrenado con el conjunto de datos de SpamAssassin, un repositorio público de correos electrónicos etiquetados como spam o no spam.

## Características Principales
- **Extracción de texto**: El contenido del correo se extrae de las partes HTML y texto plano.
- **Procesamiento de texto**: Uso de técnicas como stemming, eliminación de puntuación y manejo de URLs.
- **Modelo de clasificación**: Implementación de un clasificador de Naive Bayes para predecir si un correo es spam o no.
- **Evaluación**: Se emplean métricas como precisión, recall, F1-score y matrices de confusión para evaluar el rendimiento del modelo.
- **Persistencia del modelo**: El modelo entrenado y el pipeline de preprocesamiento se guardan utilizando `joblib` para facilitar su carga y uso en el futuro.

## Instalación

Para ejecutar este proyecto, necesitas tener instaladas las siguientes librerías:

```bash
pip install -r requirements.txt
```

## Requisitos

Las siguientes bibliotecas son necesarias para ejecutar el proyecto:

- `scikit-learn`
- `nltk`
- `joblib`
- `matplotlib`
- `urlextract`
- `scipy`
- `numpy`
- `html`
- `email`
- `tarfile`

Puedes instalar estas dependencias ejecutando:

```bash
pip install scikit-learn nltk joblib matplotlib urlextract scipy numpy
```
## Uso

Para ejecutar el proyecto, simplemente corre el script principal. Este script realizará las siguientes tareas:

1. Descargar los datos de SpamAssassin.
2. Preprocesar los correos electrónicos (extracto de texto, limpieza de HTML, etc.).
3. Entrenar un modelo de Naive Bayes sobre el conjunto de entrenamiento.
4. Evaluar el modelo utilizando las métricas estándar de clasificación.
5. Guardar el modelo entrenado y el pipeline de preprocesamiento para su uso futuro.

```bash
python spam_classifier.py
```

## Guardado y Carga del Modelo

Una vez entrenado el modelo, este se guarda en dos archivos:

- `Spam_Classifier.pkl`: El modelo de clasificación entrenado.
- `preprocess_pipeline.pkl`: El pipeline de preprocesamiento que se utiliza para transformar los correos antes de hacer predicciones.

Puedes cargar los modelos guardados y hacer predicciones en nuevos datos como se muestra a continuación:

```python
modelo_cargado = joblib.load("Spam_Classifier.pkl")
pipeline_cargado = joblib.load("preprocess_pipeline.pkl")
y_pred_cargado = modelo_cargado.predict(pipeline_cargado.transform(X_prueba))
```

## Métricas del Modelo

Al final del script, se imprime un reporte de clasificación con métricas como:

- **F1-Score**
- **Precisión**
- **Recall**

También se visualiza una matriz de confusión para evaluar el rendimiento del modelo.

## Estructura del Proyecto

```bash
.
├── datasets/ 
|  └── spam/ 
├── Spam_Classifier.pkl 
├── preprocess_pipeline.pkl 
├── api_spam.py 
├── probar_api.py 
└── requirements.txt

```
## Archivos del Proyecto

### `api_spam.py`
Este script es responsable de correr el modelo de clasificación de spam. Implementa una API (probablemente utilizando Flask o FastAPI) que permite interactuar con el modelo entrenado. A través de esta API, puedes enviar correos electrónicos para que sean clasificados como "spam" o "ham" (no spam). El script carga el modelo y el pipeline de preprocesamiento previamente guardado y realiza las predicciones sobre los correos enviados.

### `probar_api.py`
Este script está diseñado para probar la API definida en `api_spam.py`. Envía correos electrónicos de ejemplo a la API para verificar que el modelo esté funcionando correctamente y que la clasificación de los correos como "spam" o "ham" sea precisa. Es útil para asegurar que la implementación de la API y el modelo esté operando de manera adecuada.

## Uso

1. Para correr la API que clasifica correos, ejecuta el script `api_spam.py`:

    ```bash
    python api_spam.py
    ```

2. Para probar la API y verificar la clasificación de los correos electrónicos, ejecuta:

    ```bash
    python probar_api.py
    ```


## Contribuciones

Si deseas contribuir a este proyecto, por favor haz un fork del repositorio, realiza los cambios necesarios y envía un pull request.



