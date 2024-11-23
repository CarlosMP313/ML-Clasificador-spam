from flask import Flask, request, jsonify
import joblib
import email
from sklearn.pipeline import Pipeline
import numpy as np

# Cargar el modelo y el pipeline de preprocesamiento
pipeline_preprocesamiento = joblib.load("pipeline_preprocesamiento.pkl")
modelo_clasificador_spam = joblib.load("Clasificador_Spam.pkl")

# Crear la aplicación Flask
app = Flask(__name__)

# Definir una ruta para predecir si un correo es spam o ham
@app.route("/predecir", methods=["POST"])
def predecir_spam():
    # Recibir el contenido del correo electrónico como un JSON
    data = request.get_json()
    if not data or "email_content" not in data:
        return jsonify({"error": "Por favor proporciona el contenido del email"}), 400

    # Extraer el contenido del correo
    email_content = data["email_content"]

    # Crear un objeto de correo electrónico
    correo = email.message_from_string(email_content)

    # Preprocesar y transformar el correo
    correo_transformado = pipeline_preprocesamiento.transform([correo])

    # Realizar la predicción
    prediccion = modelo_clasificador_spam.predict(correo_transformado)[0]

    # Devolver el resultado de la predicción
    resultado = "spam" if prediccion == 1 else "ham"
    return jsonify({"prediccion": resultado})

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)



'''

curl -X POST http://127.0.0.1:5000/predecir -H "Content-Type: application/json" -d "{\"email_content\": \"¡Felicidades! Has sido seleccionado para ganar un increíble premio en efectivo. Gana dinero rápido y sin esfuerzo desde la comodidad de tu hogar. Solo haz clic aquí para obtener tu premio: http://example.com. No pierdas esta oportunidad única. ¡Oferta limitada!\"}"


curl -X POST http://127.0.0.1:5000/predecir -H "Content-Type: application/json" -d "{\"email_content\": \"Hola Juan, espero que estés bien. Te escribo para recordarte sobre nuestra reunión de equipo el próximo lunes a las 10 a.m. en la oficina.\"}"

'''