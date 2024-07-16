import os
import cv2
import dlib
import numpy as np
def comparar(rostroUno, rostroDos):
   
    predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
    face_path =  os.path.join(os.path.dirname(__file__), "dlib_face.dat")
    facerec = dlib.face_recognition_model_v1(face_path)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    def compare_faces(face_encodings, face_to_compare_encoding, tolerance=0.):
        face_encodings = np.array(face_encodings)
        face_to_compare_encoding = np.array(face_to_compare_encoding)
        distances = np.linalg.norm(face_encodings - face_to_compare_encoding, axis=1)
        return any(distance <= tolerance for distance in distances)
    
    gray1 = cv2.cvtColor(rostroUno, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rostroDos, cv2.COLOR_BGR2GRAY)

    faces1 = detector(gray1)
    faces2 = detector(gray2)
    
    resultado = {"estado": False, "respuesta": "No es la misma cara"}
    
    if len(faces1) == 1 and len(faces2) == 1:
        shape1 = predictor(gray1, faces1[0])
        shape2 = predictor(gray2, faces2[0])

        face_encoding1 = facerec.compute_face_descriptor(rostroUno, shape1)
        face_encoding2 = facerec.compute_face_descriptor(rostroDos, shape2)

        is_same_person = compare_faces([face_encoding1], face_encoding2)
        if is_same_person:
            resultado["estado"] = True
            resultado["respuesta"] = "Si es la misma cara"
    
    return resultado
