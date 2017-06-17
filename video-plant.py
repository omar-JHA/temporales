# coding=utf-8
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

fondo=None

def nada(x):
    pass
cv2.namedWindow('umbral')
cv2.createTrackbar('val','umbral',150,255,nada)
cv2.createTrackbar('Are','umbral',500,1500,nada)

while(True):
    ret,frame=cap.read()
    image = cv2.resize(frame,None,None,0.5,0.5,cv2.INTER_LINEAR)
    gris=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blu=cv2.GaussianBlur(gris,(5,5),0)
    if fondo is None:
        fondo = gris
    # Calculo de la diferencia entre el fondo y el frame actual
    resta = cv2.absdiff(fondo,blu)
    # Aplicamos un umbral
    valTr= cv2.getTrackbarPos('val','umbral')
    ret,umbral = cv2.threshold(resta, 25, valTr, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Dilatamos el umbral para tapar agujeros
    kernel = np.ones((3,3),np.uint8)  
    dila = cv2.dilate(umbral, kernel, iterations=2)
    grad = cv2.morphologyEx(dila,cv2.MORPH_GRADIENT,kernel)
    # Copiamos el umbral para detectar los contornos
    contornosimg = grad.copy() 
    # Buscamos contorno en la imagen
    contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contornos,-1,(0,0,255),5)
    # Recorremos todos los contornos encontrados
    for c in contornos:
        # Eliminamos los contornos más pequeños
        Areacon=cv2.contourArea(c)
        valor= cv2.getTrackbarPos('Are','umbral')
        #print("area del objeto detectado: ",Areacon)
        if Areacon<valor:
            continue
        # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
        (x, y, width, heigth) = cv2.boundingRect(c)
        # Dibujamos el rectángulo del bounds
        cv2.rectangle(image, (x, y), (x + width, y + heigth), (0, 255, 0), 2)
        #print"anchura = ",width
        #print"altura = ",heigth
        #print "area = " +str(width*heigth)+" u2"
        cv2.putText(image,"imagen impressa",(200,160),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255))
        cv2.imwrite('/home/omar23/imagen-super.jpg',frame)


    cv2.imshow("dilate",dila)
    cv2.imshow("gradiente",grad)
    cv2.imshow("umbral",umbral)
    cv2.imshow("resta",resta)
    cv2.imshow("gris",gris)
    cv2.imshow("blur",blu)

    cv2.imshow("real",image)
    key=cv2.waitKey(1) & 0xFF

    if key==ord("q"):
        break   
cap.release()
cv2.destroyAllWindows()
