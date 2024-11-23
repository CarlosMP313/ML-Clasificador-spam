import requests

# URL del servidor Flask
url = "http://127.0.0.1:5000/predict"

# Contenido del correo electrónico que quieres clasificar
email_content = """
Subject: ¡Gana un iPhone gratis!
Hola, hemos seleccionado tu correo para ofrecerte un iPhone gratis. ¡Haz clic en el enlace para reclamarlo ahora!
"""

# Realizar la solicitud POST
response = requests.post(url, json={"content": email_content})

# Verificar el resultado
if response.status_code == 200:
    print("Predicción:", response.json()["prediction"])
else:
    print("Error:", response.json().get("error", "Ocurrió un error desconocido"))
