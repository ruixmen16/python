from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os
import io
import json

def detectar_texto(bytearray_imagen):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

    imagen = Image.open(io.BytesIO(bytearray_imagen))
    imagen = imagen.convert('L')

    enhancer = ImageEnhance.Contrast(imagen)
    imagen = enhancer.enhance(2)

    imagen = imagen.point(lambda x: 0 if x < 128 else 255, '1')
    imagen = imagen.filter(ImageFilter.SHARPEN)

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    texto = pytesseract.image_to_string(imagen, config=custom_config)

    lineas = texto.split('\n')
    respuesta = {"estado": False, "respuesta": ""}
    for linea in lineas:
        if len(linea) == 10:
            respuesta["estado"] = True
            respuesta["respuesta"] = linea
            break

    return respuesta
