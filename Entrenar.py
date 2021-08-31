from posix import listdir
import cv2
import os
import numpy as np

RutaDatos='Dataset_faces'
listardir=os.listdir(RutaDatos)

labels=[]
DatosCaras=[]
label=0

for name_dir in listdir:
    DireccionDatos=RutaDatos+"/"+name_dir

    for nombreArchivo in os.listdir(DireccionDatos):
        Rutaimagen=DireccionDatos+"/"+nombreArchivo
        imagenes=cv2.imread(Rutaimagen,0)
        DatosCaras.append(imagenes)
        labels.append(label)
    label +=1

faceMask=cv2.face.LBPHFaceRecognizer_create()
print("Entrenando...... ")
faceMask.train(DatosCaras, np.array(labels))
faceMask.write("Facemodel.xml")

