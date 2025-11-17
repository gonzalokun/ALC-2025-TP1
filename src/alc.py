import numpy as np

# TODO: traer todos los metodos en la seccion de abajo cuando estén corregidos
from moduloALC import *

# Funciones del Labo

def multiplicar_en_cadena(*lista):
    res = np.eye(lista[0].shape[0])

    for matriz in lista:
        res = multiplicar(res, matriz)

    return res

def aplicar_raiz_a_diagonal(A):
    res = A
    for i in range(res.shape[0]):
        res[i][i] = np.sqrt(res[i][i])
    return res

def calcular_rango(A):
    X = A if A.shape[0] <= A.shape[1] else traspuesta(A)
    _, res_gauss, _ = calculaLU(X)

    rank = 0
    for i in range(res_gauss.shape[0]):
        todosCeros = True
        for j in range(res_gauss.shape[1]):
            if res_gauss[i][j] != 0:
                todosCeros = False

        if not todosCeros:
            rank += 1

    return rank

def calculaCholesky(A, atol=1e-10):
    # if not esSDP(A, atol):
    #     return None

    L, D, _, _ = calculaLDV(A)

    R = multiplicar(L, aplicar_raiz_a_diagonal(D))

    return R, traspuesta(R)

def resolver_sistema_con_cholesky(A, B, atol=1e-10):
    R, Rt = calculaCholesky(A, atol)

    matriz_solucion = []
    for columna in range(B.shape[1]):
        yi = res_tri(R, B[:, columna])
        xi = res_tri(Rt, yi, False)
        matriz_solucion.append(xi)

    return np.array(matriz_solucion)

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

# 1

def despejar_cholesky(L, X):
    Lt = traspuesta(L)

    matriz_solucion = []
    for columna in range(X.shape[1]):
        xi = res_tri(L, X[:, columna])
        ui = res_tri(Lt, xi, False)
        matriz_solucion.append(ui)

    return traspuesta(np.array(matriz_solucion))

def pinvEcuacionesLineales(X, L, Y):

    rango = calcular_rango(X)

    n, p = X.shape
    if rango == p and n > p:
        U = despejar_cholesky(L, traspuesta(X))
        return multiplicar(Y, U)

    # Caso 2
    if rango == n and n < p:
        Vt = despejar_cholesky(L, X)
        return multiplicar(Y, traspuesta(Vt))

    # Caso 3
    if rango == n and n == p:
        invX = inversa(X)
        return multiplicar(Y, invX)

    print("ERROR HORRIBLE")
    return ":("

def pinvEcuacionesLinealesConRango(X, L, Y, r):

    rango = r

    n, p = X.shape
    if rango == p and n > p:
        U = despejar_cholesky(L, traspuesta(X))
        return multiplicar(Y, U)

    # Caso 2
    if rango == n and n < p:
        Vt = despejar_cholesky(L, X)
        return multiplicar(Y, traspuesta(Vt))

    # Caso 3
    if rango == n and n == p:
        invX = inversa(X)
        return multiplicar(Y, invX)

    print("ERROR HORRIBLE")
    return ":("

def algoritmo1(X, Y):
    # W = Y*L+
    rango = calcular_rango(X)
    Xt = traspuesta(X)

    # Caso 1
    n, p = X.shape
    if rango == p and n > p:
        XtX = multiplicar(Xt, X)
        L, _ = calculaCholesky(XtX)
        return pinvEcuacionesLinealesConRango(X, L, Y, rango)

    # Caso 2
    if rango == n and n < p:
        XXt = multiplicar(X, traspuesta(X))
        L, _ = calculaCholesky(traspuesta(XXt))
        return pinvEcuacionesLinealesConRango(X, L, Y, rango)

    # Caso 3
    if rango == n and n == p:
        invX = inversa(X)
        return multiplicar(Y, invX)

    print("ERROR HORRIBLE")
    return None

# 2

def algoritmo2(X, Y, svdRango="max"):
    #Pre Condicion: X tiene dimension nxp y n<p, rango(X)=n (rango completo) y Y dimension m×p (matrices en los reales)
    print("svd_reducida es llamada desde algoritmo2")
    U, S, V = svd_reducida(X, tol=1e-15, rango = svdRango)
    print("pinvSVD      es llamada desde algoritmo2")
    W = pinvSVD(U, diagonal(S), V, Y)
    return W


def pinvSVD(U, S, V, Y):
    #calcula W  minW : ||Y-WX||_2

    S1_inv = np.zeros((S.shape[0], S.shape[1]))

    for i in range(S.shape[0]):
        if S[i, i] > 1e-15:
            S1_inv[i, i] = 1 / S[i, i]

    U1 = U[:, 0:S.shape[1]]  #sospecho que como usamos el algoritmo svd_reducida, U ya tiene las columnas correctas pero x las dudas las cortamos
    V1 = V[:, 0:S.shape[0]]
    U1_t = traspuesta(U1)

    pinvX = multiplicar_en_cadena(V1, S1_inv, U1_t)
    W = multiplicar_en_cadena(Y, pinvX)
    return W

# 3

def pinvHouseHolder(Q, R, Y):
    print("pinvHouseHolder")
    Qt = traspuesta(Q)
    matriz_solucion = []
    for columna in range(Qt.shape[1]):
        print(f"\r\tcalculando vector: {columna}", end="")
        ui = res_tri(R, Qt[:, columna], False)
        matriz_solucion.append(ui)

    print()
    Vt = np.array(matriz_solucion)
    print("calculando W")
    return multiplicar(Y, Vt)

def pinvGramSchmidt(Q, R, Y):
    print("pinvGramSchmidt")
    Qt = traspuesta(Q)
    matriz_solucion = []
    for columna in range(Qt.shape[1]):
        print(f"\r\tcalculando vector: {columna}", end="")
        ui = res_tri(R, Qt[:, columna], False)
        matriz_solucion.append(ui)

    print()
    Vt = np.array(matriz_solucion)
    print("calculando W")
    return multiplicar(Y, Vt)

def algoritmo3(X, Y, metodo="RH"):

    if metodo not in ["RH", "GS"]:
        return None

    Q, R = calculaQR_exp(traspuesta(X), metodo)

    print()
    if metodo == "RH":
        return pinvHouseHolder(Q, R, Y)

    if metodo == "GS":
        return pinvGramSchmidt(Q, R, Y)

# 4

def esPseudoInversa(X, pX, tol=1e-8):
    return matricesIguales(multiplicar_en_cadena(X, pX, X), X, tol) \
        and matricesIguales(multiplicar_en_cadena(pX, X, pX), pX, tol) \
        and esSimetrica(multiplicar(X, pX)) \
        and esSimetrica(multiplicar(pX, X))

# 5

def matrizConfusion(YPrediction, YReal):

    res = matrizDeCeros(2, 2)

    for j in range(YPrediction.shape[1]):
        valorReal = (YReal[0, j], YReal[1, j])
        valorPred = (YPrediction[0, j], YPrediction[1, j])

        if valorPred[0] >= valorPred[1]:
            # Prediccion Gato
            if np.abs(valorReal[0] - 1) < 1e-15:
                # Era gato
                res[0, 0] += 1
            else:
                res[1, 0] += 1
        else:
            # Prediccion Perro
            if np.abs(valorReal[1] - 1) < 1e-15:
                # Era perro
                res[1, 1] += 1
            else:
                res[0, 1] += 1

    return res
