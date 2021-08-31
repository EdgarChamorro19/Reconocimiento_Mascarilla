import cv2
import os
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_Mascarilla", "Sin_Mascarilla"]
# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("Facemodel.xml")
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
     min_detection_confidence=0.5) as face_detection:
     while True:
          ret, Imagen = cap.read()
          if ret == False: break
          Imagen = cv2.flip(Imagen, 1)
          height, width, _ = Imagen.shape
          Imagen_rgb = cv2.cvtColor(Imagen, cv2.COLOR_BGR2RGB)
          resultado = face_detection.process(Imagen_rgb)
          if resultado.detections is not None:
               for detection in resultado.detections:
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    if xmin < 0 and ymin < 0:
                         continue
                    imagenCara = Imagen[ymin : ymin + h, xmin : xmin + w]
                    imagenCara = cv2.cvtColor(imagenCara, cv2.COLOR_BGR2GRAY)
                    imagenCara = cv2.resize(imagenCara, (72, 72), interpolation=cv2.INTER_CUBIC)
                    
                    result = face_mask.predict(imagenCara)
                    #cv2.putText(Imagen, "{}".format(result), (xmin, ymin - 5), 1, 1.3, (210, 124, 176), 1, cv2.LINE_AA)
                    if result[1] < 150:
                         color = (0, 255, 0) if LABELS[result[0]] == "Con_Mascarilla" else (0, 0, 255)
                         cv2.putText(Imagen, "{}".format(LABELS[result[0]]), (xmin, ymin - 15), 2, 1, color, 2, cv2.LINE_AA)
                         cv2.rectangle(Imagen, (xmin, ymin), (xmin + w, ymin + h), color, 4)
          cv2.imshow("Imagen", Imagen)
          k = cv2.waitKey(30)
          if k == 27:
               break
cap.release()
cv2.destroyAllWindows()