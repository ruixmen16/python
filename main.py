from flask import Flask, jsonify, request
from ROSTRO.CompararRostro import comparar
from TEXTO.ocr import detectar_texto
import cv2
import numpy as np
import sys 
app = Flask(__name__)


def procesar_imagen_bytes(bytes_imagen):
    np_bytes = np.frombuffer(bytes_imagen, dtype=np.uint8)
    imagen = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    return imagen

@app.route("/comparar", methods=['POST'])
def comparar_caras():
    # Se espera recibir dos imágenes codificadas en bytes como form-data 'foto1' y 'foto2'
    imagen1 = request.files['foto1'].read()
    imagen2 = request.files['foto2'].read()

    imagen1 = procesar_imagen_bytes(imagen1)
    imagen2 = procesar_imagen_bytes(imagen2)

    resultado = comparar(imagen1, imagen2)

    return jsonify(resultado)

@app.route("/ocr", methods=['POST'])
def detectar_texto_imagen():
    imagen = request.files['imagen'].read()

    resultado = detectar_texto(imagen)

    return jsonify(resultado)

def run_flask_app():
    # Verificar si el script se ejecuta como un servicio de Windows
    if 'pywin32_system32' in sys.executable:
        # Si se ejecuta como un servicio de Windows, no se inicia el servidor Flask
        return
    # Si no se está ejecutando como un servicio de Windows, iniciar el servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    run_flask_app()
