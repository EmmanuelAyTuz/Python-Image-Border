### Libreria ###
import cv2 as cv

image1 = "src/data/logo_itsva.png"
image2 = "src/data/logo_tecnm.png"


def welcome():
    print("################################################################")
    print("################################################################")
    print("######### Procesar Imágenes En Python (Usando OpenCV) ##########")
    print("################################################################")
    print("################################################################")


def original(p_image):
    rimg = cv.imread(p_image)
    cv.imshow('Imagen original', rimg)
    cv.waitKey(0)


def bcanny(p_image):
    # cv2.Canny(image, minVal, maxVal)
    img = cv.imread(p_image)
    edge_img = cv.Canny(img, 100, 200)
    cv.imshow("Deteccion de bordes", edge_img)
    cv.waitKey(0)


def fcontours(p_image, x, y, z):
    img = cv.imread(p_image)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    retval, thresh = cv.threshold(gray_img, x, y, z)
    img_contours, _ = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, img_contours, -1, (0, 0, 255))
    cv.imshow('Contorno de la imagen', img)
    cv.waitKey(0)


# welcome()
# original(image1)
# bcanny(image1)
# original(image2)
# bcanny(image2)
# original(image1)
fcontours(image1, 127, 255, 0)
fcontours(image2, 175, 255, 100)
