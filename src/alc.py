import numpy as np

# TODO: traer todos los metodos en la seccion de abajo cuando estén corregidos
from moduloALC import multiplicar, matricesIguales, esSimetrica, traspuesta


# Funciones del Labo

def multiplicar_en_cadena(*lista):
    res = np.eye(lista[0].shape)

    for matriz in lista:
        res = multiplicar(res, matriz)

    return res

# Funciones TP

def cargarDataset(carpeta):
    matrizGatosTrain = np.load(carpeta + "/train/cats/efficientnet_b3_embeddings.npy")
    matrizPerrosTrain = np.load(carpeta + "/train/dogs/efficientnet_b3_embeddings.npy")

    Xt = np.concatenate((matrizGatosTrain, matrizPerrosTrain), axis=1)

    cantGatos = matrizGatosTrain.shape[1]
    cantPerros = matrizPerrosTrain.shape[1]

    Yt = []

    for i in range(cantGatos):
        Yt.append(np.array([1, 0]))

    for i in range(cantPerros):
        Yt.append(np.array([0, 1]))

    Yt = traspuesta(np.array(Yt))

    #print(f"TAM Yt: {Yt.shape}")

    matrizGatosValidacion = np.load(carpeta + "/val/cats/efficientnet_b3_embeddings.npy")
    matrizPerrosValidacion = np.load(carpeta + "/val/dogs/efficientnet_b3_embeddings.npy")

    Xv = np.concatenate((matrizGatosValidacion, matrizPerrosValidacion), axis=1)

    cantGatos = matrizGatosValidacion.shape[1]
    cantPerros = matrizPerrosValidacion.shape[1]

    Yv = []

    for i in range(cantGatos):
        Yv.append(np.array([1, 0]))

    for i in range(cantPerros):
        Yv.append(np.array([0, 1]))

    Yv = traspuesta(np.array(Yv))

    return Xt, Yt, Xv, Yv

def esPseudoInversa(X, pX, tol=1e-8):
    return matricesIguales(multiplicar_en_cadena(X, pX, X), X, tol) \
        and matricesIguales(multiplicar_en_cadena(pX, X, pX), pX, tol) \
        and esSimetrica(multiplicar(X, pX)) \
        and esSimetrica(multiplicar(pX, X))
