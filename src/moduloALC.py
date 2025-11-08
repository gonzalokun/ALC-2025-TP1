from fcntl import FASYNC

import numpy as np
import math

from puddlestuff.audioinfo.id3 import v1_option
from soupsieve.util import lower


# labo 0

def esCuadrada(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def matrizDeCeros(filas, columnas):
    return np.array([[0.0 for _ in range(columnas)] for _ in range(filas)])

def triangSup(a):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = matrizDeCeros(filas, columnas)
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas):
            if fila >= columna:
                result[fila][columna] = 0
            else :
                result[fila][columna] = a[fila][columna]
    return result

def triangInf(a):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = matrizDeCeros(filas, columnas)
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas):
            if fila <= columna:
                result[fila][columna] = 0
            else:
                result[fila][columna] = a[fila][columna]
    return result

def diagonal(a):
    a = np.array(a)

    filas = len(a)
    result = matrizDeCeros(filas, filas)
    for fila in range(filas):
        for columna in range(filas):
            if fila != columna:
                result[fila][columna] = 0
            else:
                result[fila][columna] = a[fila]
    return result

def traza(a):
    if not esCuadrada(a):
        return False

    result = 0
    filas, columnas = a.shape
    for fila in range(filas):
        for columna in range(columnas):
            if fila == columna:
                result = result + a[fila][columna]
    return result

def traspuesta(a):

    if a.shape[0] == 0:
        return a

    # Transponer vector
    if len(a.shape) == 1:
        filas, = a.shape
        result = matrizDeCeros(1, filas)
        result = np.array(result)
        for fila in range(filas):
            result[0][fila] = a[fila]
        return result

    # Transponer matriz
    else:
        filas, columnas = a.shape
        result = matrizDeCeros(columnas, filas)
        result = np.array(result)
        for fila in range(filas):
            for columna in range(columnas):
                result[columna][fila] = a[fila][columna]
        return result

def vectorAMatriz(a):
    return traspuesta(traspuesta(a))

def esSimetrica(a):

    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    tras = traspuesta(a)

    dif = restar(a, tras)

    for i in range(filas):
        for j in range(columnas):
            if np.abs(dif[i][j]) >= 1e-10:
                return False

    return True

def restar(a, b):
    # ponele que checkeamos que sean == las dim
    if a.shape != b.shape:
        raise Exception("No se puede :(")

    filas, columnas = a.shape
    res = matrizDeCeros(filas, columnas)
    for i in range(filas):
        for j in range(columnas):
            res[i][j] = a[i][j] - b[i][j]

    return res

def calcularAx(matriz, vector_x):
    tamVector = len(vector_x)
    if matriz.shape[1] != tamVector:
        raise Exception("No se puede :(")

    result = [0 for _ in range(matriz.shape[0])]

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            result[i] += matriz[i][j] * vector_x[j]

    return np.array(result)

def intercambiarFila(matriz, fila1, fila2):
    for j in range(matriz.shape[1]):
        tmp = matriz[fila1][j]
        matriz[fila1][j] = matriz[fila2][j]
        matriz[fila2][j] = tmp

def sumarFilaMultiplo(matriz, fila1, fila2, num):
    for j in range(matriz.shape[1]):
        matriz[fila1][j] += num*matriz[fila2][j]

def esDiagonalmenteDominante(matriz):
    if not esCuadrada(matriz):
        return False

    for i in range(matriz.shape[0]):
        elem_diag = abs(matriz[i][i])
        sum = 0
        for j in range(matriz.shape[1]):
            if i != j:
                sum += abs(matriz[i][j])

        if elem_diag <= sum:
            return False

    return True

def circulante(vector):

    result = matrizDeCeros(vector.shape[0], vector.shape[0])

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[(j-i)%vector.shape[0]]

    return result

def matrizVandermonde(vector):

    result = matrizDeCeros(vector.shape[0], vector.shape[0])

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[j]**i

    return result

def numeroAureo(n):
    a = 0
    b = 1

    for i in range(n+1):
        tmp = a
        a = b
        b += tmp

    if a == 0:
        return 0

    return b/a

def multiplicar(matrizA, matrizB):
    if matrizA.shape[1] != matrizB.shape[0]:
        raise Exception("No se puede :(")

    res = matrizDeCeros(matrizA.shape[0], matrizB.shape[1])

    for i in range(matrizA.shape[0]):
        for j in range(matrizB.shape[1]):
            for k in range(matrizA.shape[1]):
                res[i][j] += matrizA[i][k]*matrizB[k][j]

    return np.array(res)

def multiplacionMatricialDeVectores(vectorA, vectorB):
    res = np.zeros((vectorA.shape[0], vectorB.shape[0]))

    for i in range(vectorA.shape[0]):
        for j in range(vectorB.shape[0]):
            res[i][j] = vectorA[i]*vectorB[j]
    return res

def productoEscalar(vectorA, vectorB):
    if vectorA.shape[0] != vectorB.shape[0]:
        return None
    else:
        res = 0.0
        for i in range(vectorA.shape[0]):
            res += vectorA[i]*vectorB[i]
        return res

def vectorPorEscalar(x, s):
    res = []
    for i in range(len(x)):
        res.append(x[i]*s)
    return np.array(res)

# labo 1

def error(x, y):
    return abs(x - y)

def error_relativo(x, y):
    return abs(x - y) / abs(x)

def matricesIguales(A, B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if A.shape != B.shape:
        return False

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i][j] - B[i][j]) >= 1e-08:
                return False

    return True

# labo 2

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    res =np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    return res

def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i]
    """
    cantElems = len(s)
    res = matrizDeCeros(cantElems, cantElems)
    for i in range(cantElems):
        res[i][i] = s[i]

    return np.array(res)

def rota_y_escala(theta,s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    res = multiplicar(rota(theta), escala(s))
    return res

def afin(theta,s,b):
    """
    Recibe un angulo theta , una tira de numeros s (en R2) , y un vector
    b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor
    fijo b
    """
    matriz2x2 = rota_y_escala(theta,s)
    matriz3x3 = np.array(matrizDeCeros(3, 3))
    matriz3x3[:2, :2] = matriz2x2
    matriz3x3[0][2] = b[0]
    matriz3x3[1][2] = b[1]
    matriz3x3[2][2] = 1
    return np.array(matriz3x3)

def trans_afin(v,theta,s,b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a
    v
    """
    transf = afin(theta,s,b)
    vectorCon1 =  np.append(v, 1)
    vectorColumna = vectorCon1.T
    vectorColumnaRes = calcularAx(transf, vectorColumna)
    res = vectorColumnaRes.T[:2]
    return res

# labo 3

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        vectorAbs = [0 for _ in range(len(x))]
        for i in range(len(x)):
            vectorAbs[i] = abs(x[i])
        return np.max(vectorAbs)

    sum = 0
    for i in range(len(x)):
        sum += abs(x[i])**p

    return sum**(1/p)

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacios, y un escalar p. Devuelve
    una lista donde cada elemento corresponde a normalizar los
    elementos de X con la norma p.
    """
    vectoresNormalizados = []
    for i in range(len(X)):
        vectorActual = X[i]
        vectoresNormalizados.append(vectorPorEscalar(vectorActual, (1/norma(vectorActual, p))))
    return vectoresNormalizados

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma A-{q,p} y el vector x en el cual se alcanza
    el maximo.
    """
    vectoresAlAzar = np.random.rand(Np, A.shape[1])
    vectoresNormalizados = normaliza(vectoresAlAzar, p)

    vectorConNorma = [0 for _ in range(len(vectoresNormalizados))]
    for i in range(len(vectoresNormalizados)):
        vectorConNorma[i] = [norma(calcularAx(A, vectoresNormalizados[i]), q), vectoresNormalizados[i]]

    max = [0, [0, 0]]
    for i in range(len(vectorConNorma)):
        if vectorConNorma[i][0] > max[0]:
            max = vectorConNorma[i]

    # return max(vectorConNorma, key=lambda p: p[0])
    return max

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    if not p in [1,'inf']:
        return None
    if p == 1:
        vectorSumas = []
        for j in range(A.shape[1]):
            sum = 0
            for i in range(A.shape[0]):
                sum += abs(A[i][j])
            vectorSumas.append(sum)
        return np.max(vectorSumas)

    if p == 'inf':
        vectorSumas = []
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum += abs(A[i][j])
            vectorSumas.append(sum)
        return np.max(vectorSumas)

def condMC(A, p, cantVect):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    inversa = np.linalg.inv(A)
    return normaMatMC(A, p, p, cantVect)[0] * normaMatMC(inversa, p, p, cantVect)[0]

def condExacta(A, p):
    """
    Que devuelve el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    inversa = np.linalg.inv(A)
    return normaExacta(A, p) * normaExacta(inversa, p)

# labo 4

def calculaLU(A):
    nops = 0
    upper = matrizDeCeros(A.shape[0], A.shape[1]) + A
    lower = escala([1 for _ in range(A.shape[0])])

    for fila in range(upper.shape[0]):
        numDiagonal = upper[fila][fila]

        if np.abs(numDiagonal)  < 1e-08:
            return [None, None, 0]

        for fila2 in range(fila+1, upper.shape[0]):
            nops += 1

            coef = upper[fila2][fila]/numDiagonal
            lower[fila2][fila] = coef

            upper[fila2][fila] = 0.0

            for columna in range(fila+1, upper.shape[1]):
                upper[fila2][columna] = upper[fila2][columna] - coef*upper[fila][columna]
                nops += 2

    return [lower, upper, nops]


def res_tri(L, b, inferior=True):

    n = L.shape[1]
    x_vector =  np.zeros(n)
    
    if inferior:
        for i in range(n):
            x_actual = b[i]
            #idea si es triangular inferior la solucion es de la pinta (b1/L11 , (b2-L21.X1)/L22 , b3-L32.X2-L31.X1   cada x_n se le restan todos los anteriores x_n
            for j in range(i):
                x_actual -= L[i][j]*x_vector[j]
            x_actual = x_actual/L[i][i]
            x_vector[i] = x_actual
    else:
        for i in range(n-1,-1,-1):
            x_actual = b[i]
            for j in range(i+1,n):
                x_actual -= L[i][j]*x_vector[j]
            x_actual = x_actual/L[i][i]
            x_vector[i] = x_actual

    return x_vector

def inversa(A):

    descomposicion = calculaLU(A)
    L = descomposicion[0]
    U = descomposicion[1]

    if L is None or U is None:
        return None

    filas, columnas = A.shape
    inversa = np.zeros((filas, columnas))
    for columna in range(columnas):
        vector_canonico = np.zeros(filas)
        vector_canonico[columna] = 1.
        y = res_tri(L, vector_canonico)
        inversa[columna] = res_tri(U, y, False)
    return traspuesta(inversa)

def calculaLDV(A):
    nops = 0
    descomposicion = calculaLU(A)
    L = descomposicion[0]
    U = descomposicion[1]
    nops += descomposicion[2]

    if L is None or U is None:
        return [None, None, None, 0]

    U_t = traspuesta(U)
    descomposicionU_t = calculaLU(U_t)

    V_t = descomposicionU_t[0]
    D = descomposicionU_t[1]
    nops += descomposicionU_t[2]

    V = traspuesta(V_t)

    return [L, D, V, nops]

def esSDP(A, atol=1e-08):
    if not esSimetrica(A):
        return False

    descLDV = calculaLDV(A)

    if descLDV[0] is None:
        return False

    D = descLDV[1]

    for i in range(D.shape[0]):
        if D[i][i] <= 0:
            return False

    return True

# LABO-6

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    nops = 0 # TODO: Ver si los nops estan bien calculados porque no hay tests sobre ellos
    n = A.shape[0]

    if n != A.shape[1]:
        return None

    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)

    for j in range(0, n):
        Q[:, j] = A[:, j]
        for k in range(j):
            R[k, j] = np.dot(Q[:, k], Q[:, j])
            nops += n * 2 - 1 # n mult y n-1 sumas
            Q[:, j] = Q[:, j] - vectorPorEscalar(Q[:, k], R[k, j])
            nops += n * 2 # n mult y n restas

        R[j, j] = norma(Q[:, j], 2)
        nops += n * 2 # n mult y n-1 sumas y una raiz
        Q[:, j] = vectorPorEscalar(Q[:, j], 1/R[j, j])
        nops += n # n mult

    if retorna_nops:
        return [Q, R, nops]

    return [Q, R]

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    m = A.shape[0]
    n = A.shape[1]
    if m < n:
        return None

    R = A
    Q = np.eye(m)

    for k in range(n):
        x = R[k:m, k]
        alfa = (-1) * np.sign(x[0]) / norma(x, 2)
        u = x - alfa * np.eye(m - k)[0]
        norma_u = norma(u, 2)
        if norma_u > tol:
            u = u / norma_u
            Hk = np.eye(m-k) - 2 * multiplacionMatricialDeVectores(u, u)
            Hk_p = np.eye(m)
            Hk_p[k:m, k:m] = Hk 
            R = multiplicar(Hk_p, R)
            Q = multiplicar(Q, Hk_p)

    return [Q, R]

def calculaQR(A, metodo='RH',tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'GS':
        return QR_con_GS(A, tol)
    elif metodo == 'RH':
        return QR_con_HH(A, tol)
    else:
        return None


def matrizPorEscalar(A, c):
    res = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res[i][j] = A[i][j] * c

    return res

def aplicarMatrizYNormalizar(A, v):
    w = calcularAx(A, v)
    wNorma = norma(w, 2)
    if wNorma <= 1e-15:
        return np.zeros(w.shape[0])
    else:
        w = vectorPorEscalar(w, 1/wNorma)
    return w

def aplicarMatrizKVecesYNormalizar(A, v, k):
    w = v
    for i in range(k):
        w = aplicarMatrizYNormalizar(A, w)
    return w

def metpot2k(A, tol=1e-15, K=1000):
    v = np.random.rand(A.shape[1])
    vPrima = aplicarMatrizKVecesYNormalizar(A, v, 2)

    e = productoEscalar(vPrima, v)
    k = 0
    while np.abs(e-1) >= tol and k < K:
        v = vPrima
        vPrima = aplicarMatrizKVecesYNormalizar(A, v, 2)
        e = productoEscalar(vPrima, v)
        k += 1

    autovalor = productoEscalar(vPrima, calcularAx(A, vPrima))

    if autovalor < tol:
        autovalor = 0.0

    error = e-1
    return [vPrima, autovalor, error]

def restarVectores (a, b):
    if a.shape != b.shape:
        return None

    res = np.zeros(a.shape)

    for i in range(a.shape[0]):
        res[i] = a[i] - b[i]

    return res

def diagRH(A, tol=1e-15, K=1000):

    if not esSimetrica(A):
        return None

    autovector, lamda, _ =  metpot2k(A, tol, K)

    n = A.shape[0]
    u = restarVectores(np.eye(n)[0], autovector)
    uNormaAl2 = norma(u, 2)**2
    
    aRestar = matrizPorEscalar(multiplacionMatricialDeVectores(u, u), (2/uNormaAl2))
    reflectorHouseholder = restar(np.eye(n), aRestar)

    if n == 2:
        S = reflectorHouseholder
        D = multiplicar(reflectorHouseholder, multiplicar(A, traspuesta(reflectorHouseholder)))

    else:
        B = multiplicar(reflectorHouseholder, multiplicar(A, traspuesta(reflectorHouseholder)))
        APrima = B[1:n, 1:n]
        SPrima, DPrima = diagRH(APrima, tol, K)
        D = matrizPorEscalar(np.eye(A.shape[0]), lamda)
        D[1:n, 1:n] = DPrima
        S = np.eye(A.shape[0])
        S[1:n, 1:n] = SPrima
        S = multiplicar(reflectorHouseholder, S)

    return S, D

# LABO-7

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    res = matrizDeCeros(n,n)

    for j in range(n):
        suma = 0

        for i in range(n):
            res[i, j] = np.random.rand()
            suma += res[i, j]

        if suma != 0:
            for i in range(n):
                res[i,j] = res[i,j] / suma

    return res

def transiciones_al_azar_uniformes(n, thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas.
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres.
    Todos los elementos de la columna $j$ son iguales
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    res = matrizDeCeros(n, n)

    for j in range(n):
        suma = 0

        for i in range(n):
            res[i, j] = 1 if np.random.rand() <= thres else 0
            suma += res[i, j]

        if suma != 0:
            for i in range(n):
                res[i, j] = res[i, j] / suma

    return res

def nucleo(A, tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    A_t = traspuesta(A)
    M = multiplicar(A_t, A)
    S, D = diagRH(M, tol)

    autovectores_nucleo = []

    for i in range(D.shape[0]):
        if np.abs(D[i, i]) <= tol:
            autovectores_nucleo.append(S[:, i])

    if len(autovectores_nucleo) == 0:
        return np.array([])

    res = matrizDeCeros(A.shape[1], len(autovectores_nucleo))

    for i in range(len(autovectores_nucleo)):
        for j in range(res.shape[0]):
            res[j][i] = autovectores_nucleo[i][j]

    return vectorAMatriz(res)

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default.
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    elems = {}

    if not len(listado) == 0:
        coord_i, coord_j, valores = listado

        for k in range(len(coord_i)):
            if np.abs(valores[k]) > tol:
                elems[(coord_i[k], coord_j[k])] = valores[k]

    return elems, (m_filas, n_columnas)

def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v
    """

    res = np.zeros(v.shape[0])

    for key, value in A[0].items():
        i = key[0]
        j = key[1]
        res[i] += value * v[j]

    return res

# Labo 8

def svd_reducida(A, k="max", tol=1e-15):
    A_t = traspuesta(A)
    M = multiplicar(A_t, A)
    S, D = diagRH(M, tol)

    autovectores = []
    autovalores = []

    if k == "max":
        k = D.shape[0]

    i = 0
    while i < k and np.abs(D[i, i]) >= tol:

        if np.abs(D[i, i]) >= tol and not (D[i, i] < 0):
            autovectores.append(S[:, i])
            autovalores.append(np.sqrt(D[i, i]))

        i += 1

    res = matrizDeCeros(A.shape[1], len(autovectores))

    for i in range(len(autovectores)):
        for j in range(res.shape[0]):
            res[j][i] = autovectores[i][j]

    V_s = vectorAMatriz(res)
    E_s = autovalores

    B = multiplicar(A, V_s)

    for j in range(B.shape[1]):
        B[:, j] = vectorPorEscalar(B[:, j], 1/norma(B[:, j], 2))

    U_s = B

    return U_s, E_s, V_s

# Tests para los labos

# funciones extras para los tests
def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y),0, atol=atol)

def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True

def esNucleo(A, S, tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True

def correrTestsLabos():
    #test_labo1()
    #test_labo2()
    #test_labo3()
    #test_labo4()
    #test_labo5()
    test_labo6()
    #test_labo7()
    #test_labo8()

def test_labo1():
    assert(not sonIguales(1,1.1))
    assert(sonIguales(1,1 + np.finfo('float64').eps))
    assert(not sonIguales(1,1 + np.finfo('float32').eps))
    assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
    assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

    assert(np.allclose(error_relativo(1,1.1),0.1))
    assert(np.allclose(error_relativo(2,1),0.5))
    assert(np.allclose(error_relativo(-1,-1),0))
    assert(np.allclose(error_relativo(1,-1),2))

    assert(matricesIguales(np.diag([1,1]),np.eye(2)))
    assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
    assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

def test_labo2():
    # Tests para rota
    assert(np.allclose(rota(0), np.eye(2)))
    assert(np.allclose(rota(np.pi/2), np.array([[0, -1],[1, 0]])))
    assert(np.allclose(rota(np.pi), np.array([[-1, 0],[0, -1]])))

    # Tests para escala
    assert(np.allclose(escala([2,3]), np.array([[2,0],[0,3]])))
    assert(np.allclose(escala([1,1,1]), np.eye(3)))
    assert(
        np.allclose(escala([0.5,0.25]), np.array([[0.5,0],[0,0.25]]))
    )

    # Tests para rota-y-escala
    assert(
        np.allclose(rota_y_escala(0,[2,3]), np.array([[2,0],[0,3]]))
    )
    assert(np.allclose(
        rota_y_escala(np.pi/2,[1,1]), np.array([[0,-1],[1,0]])
    ))
    assert(np.allclose(
        rota_y_escala(np.pi,[2,2]), np.array([[-2,0],[0,-2]]))
    )

    # Tests para afin
    assert(np.allclose(
        afin(0,[1,1],[1,2]),
        np.array([[1,0,1],
                  [0,1,2],
                  [0,0,1]]))
    )

    assert(np.allclose(afin(np.pi/2,[1,1],[0,0]),
                       np.array([[0,-1,0],
                                 [1,0,0],
                                 [0,0,1]]))
           )

    assert(np.allclose(afin(0,[2,3],[1,1]),
                       np.array([[2,0,1],
                                 [0,3,1],
                                 [0,0,1]]))
           )

    # Tests para trans_afin
    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2,[1,1],[0,0]),
        np.array([0,1])
    ))

    assert(np.allclose(
        trans_afin(np.array([1,1]), 0,[2,3],[0,0]),
        np.array([2,3])
    ))

    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2,[3,2],[4,5]),
        np.array([4,8])
    ))

def test_labo3():
    # Tests norma
    assert (np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2)))
    assert (np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10)))
    assert (norma(np.random.rand(10), 2) <= np.sqrt(10))
    assert (norma(np.random.rand(10), 2) >= 0)

    # Tests normaliza
    # Tests normaliza
    for x in normaliza([np.array([1] * k) for k in range(1, 11)], 2):
        assert (np.allclose(norma(x, 2), 1))
    for x in normaliza([np.array([1] * k) for k in range(2, 11)], 1):
        assert (not np.allclose(norma(x, 2), 1))
    for x in normaliza([np.random.rand(k) for k in range(1, 11)], 'inf'):
        assert (np.allclose(norma(x, 'inf'), 1))

    # Tests normaExacta

    assert (np.allclose(normaExacta(np.array([[1, -1], [-1, -1]]), 1), 2))
    assert (np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 1), 6))
    assert (np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 'inf'), 7))
    assert (normaExacta(np.array([[1, -2], [-3, -4]]), 2) is None)
    assert (normaExacta(np.random.random((10, 10)), 1) <= 10)
    assert (normaExacta(np.random.random((4, 4)), 'inf') <= 4)

    # Test normaMC

    nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
    assert(np.allclose(nMC[0],1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

    nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
    assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

    A = np.array([[1,2],[3,4]])
    nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
    assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

    # Test condMC

    A = np.array([[1, 1], [0, 1]])
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaMatMC(A, 2, 2, 10000)
    normaA_ = normaMatMC(A_, 2, 2, 10000)
    condA = condMC(A, 2, 10000)
    print("ANTES DE ASSERT condMC 1: ")
    print("normaA: ", normaA)
    print("normaA_: ", normaA_)
    print("mult normas: ", normaA[0] * normaA_[0])
    print("condA: ", condA)
    assert (np.allclose(normaA[0] * normaA_[0], condA, atol=1e-3))

    A = np.array([[3, 2], [4, 1]])
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaMatMC(A, 2, 2, 10000)
    normaA_ = normaMatMC(A_, 2, 2, 10000)
    condA = condMC(A, 2, 10000)
    assert (np.allclose(normaA[0] * normaA_[0], condA, atol=1e-3))

    # Test condExacta

    A = np.random.rand(10, 10)
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaExacta(A, 1)
    normaA_ = normaExacta(A_, 1)
    condA = condExacta(A, 1)
    assert (np.allclose(normaA * normaA_, condA))

    A = np.random.rand(10, 10)
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaExacta(A, 'inf')
    normaA_ = normaExacta(A_, 'inf')
    condA = condExacta(A, 'inf')
    assert (np.allclose(normaA * normaA_, condA))

def test_labo4():
    # Tests LU

    L0 = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    U0 = np.array([[10, 1, 0], [0, 2, 1], [0, 0, 1]])
    A = L0 @ U0
    L, U, nops = calculaLU(A)
    assert (np.allclose(L, L0))
    assert (np.allclose(U, U0))

    L0 = np.array([[1, 0, 0], [1, 1.001, 0], [1, 1, 1]])
    U0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    A = L0 @ U0
    L, U, nops = calculaLU(A)
    assert (not np.allclose(L, L0))
    assert (not np.allclose(U, U0))
    assert (np.allclose(L, L0, atol=1e-3))
    assert (np.allclose(U, U0, atol=1e-3))
    assert (nops == 13)

    L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    U0 = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
    A = L0 @ U0
    L, U, nops = calculaLU(A)
    assert (L is None)
    assert (U is None)
    assert (nops == 0)

    ## Tests res_tri

    A = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    b = np.array([1, 1, 1])
    assert (np.allclose(res_tri(A, b), np.array([1, 0, 0])))
    b = np.array([0, 1, 0])
    assert (np.allclose(res_tri(A, b), np.array([0, 1, -1])))
    b = np.array([-1, 1, -1])
    assert (np.allclose(res_tri(A, b), np.array([-1, 2, -2])))
    b = np.array([-1, 1, -1])
    assert (np.allclose(res_tri(A, b, inferior=False), np.array([-1, 1, -1])))

    A = np.array([[3, 2, 1], [0, 2, 1], [0, 0, 1]])
    b = np.array([3, 2, 1])
    assert (np.allclose(res_tri(A, b, inferior=False), np.array([1 / 3, 1 / 2, 1])))

    A = np.array([[1, -1, 1], [0, 1, -1], [0, 0, 1]])
    b = np.array([1, 0, 1])
    assert (np.allclose(res_tri(A, b, inferior=False), np.array([1, 1, 1])))

    # Test inversa

    ntest = 10
    iter = 0
    while iter < ntest:
        A = np.random.random((4, 4))
        A_ = inversa(A)
        if not A_ is None:
            assert (np.allclose(np.linalg.inv(A), A_))
            iter += 1

    # Matriz singular devería devolver None
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (inversa(A) is None)

    # Test LDV:

    L0 = np.array([[1, 0, 0], [1, 1., 0], [1, 1, 1]])
    D0 = np.diag([1, 2, 3])
    V0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    A = L0 @ D0 @ V0
    L, D, V, nops = calculaLDV(A)
    assert (np.allclose(L, L0))
    assert (np.allclose(D, D0))
    assert (np.allclose(V, V0))

    L0 = np.array([[1, 0, 0], [1, 1.001, 0], [1, 1, 1]])
    D0 = np.diag([3, 2, 1])
    V0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1.001]])
    A = L0 @ D0 @ V0
    L, D, V, nops = calculaLDV(A)
    assert (np.allclose(L, L0, 1e-3))
    assert (np.allclose(D, D0, 1e-3))
    assert (np.allclose(V, V0, 1e-3))

    # Tests SDP

    L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    D0 = np.diag([1, 1, 1])
    A = L0 @ D0 @ L0.T
    assert (esSDP(A))

    D0 = np.diag([1, -1, 1])
    A = L0 @ D0 @ L0.T
    assert (not esSDP(A))

    D0 = np.diag([1, 1, 1e-16])
    A = L0 @ D0 @ L0.T
    assert (not esSDP(A))

    L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    D0 = np.diag([1, 1, 1])
    V0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1 + 1e-10, 1]]).T
    A = L0 @ D0 @ V0
    assert (not esSDP(A))

def test_labo5():
    # --- Matrices de prueba ---
    A2 = np.array([[1., 2.],
                   [3., 4.]])

    A3 = np.array([[1., 0., 1.],
                   [0., 1., 1.],
                   [1., 1., 0.]])

    A4 = np.array([[2., 0., 1., 3.],
                   [0., 1., 4., 1.],
                   [1., 0., 2., 0.],
                   [3., 1., 0., 2.]])

    # --- TESTS PARA QR_by_GS2 ---
    Q2, R2 = QR_con_GS(A2)
    check_QR(Q2, R2, A2)

    Q3, R3 = QR_con_GS(A3)
    check_QR(Q3, R3, A3)

    Q4, R4 = QR_con_GS(A4)
    check_QR(Q4, R4, A4)

    # --- TESTS PARA QR_by_HH ---
    Q2h, R2h = QR_con_GS(A2)
    check_QR(Q2h, R2h, A2)

    Q3h, R3h = QR_con_HH(A3)
    check_QR(Q3h, R3h, A3)

    Q4h, R4h = QR_con_HH(A4)
    check_QR(Q4h, R4h, A4)

    # --- TESTS PARA calculaQR ---
    Q2c, R2c = calculaQR(A2, metodo='RH')
    check_QR(Q2c, R2c, A2)

    Q3c, R3c = calculaQR(A3, metodo='GS')
    check_QR(Q3c, R3c, A3)

    Q4c, R4c = calculaQR(A4, metodo='RH')
    check_QR(Q4c, R4c, A4)


def test_labo6():
    #### TESTEOS
    # Tests metpot2k
    S = np.vstack([
        np.array([2,1,0])/np.sqrt(5),
        np.array([-1,2,5])/np.sqrt(30),
        np.array([1,-2,1])/np.sqrt(6)
                ]).T

    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(100):
        D = np.diag(np.random.random(3)+1)*100
        A = S@D@S.T
        v,l,_ = metpot2k(A,1e-15,1e5)
        if np.abs(l - np.max(D))< 1e-8:
            exitos += 1
    assert exitos > 95

    #Test con HH
    exitos = 0
    for i in range(100):
        v = np.random.rand(9)
        #v = np.abs(v)
        #v = (-1) * v
        ixv = np.argsort(-np.abs(v))
        D = np.diag(v[ixv])
        I = np.eye(9)
        H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

        A = H@D@H.T
        v,l,_ = metpot2k(A, 1e-15, 1e5)
        #max_eigen = abs(D[0][0])
        if abs(l - D[0,0]) < 1e-8:         
            exitos +=1
    assert exitos > 95

    # Tests diagRH
    D = np.diag([1,0.5,0.25])
    S = np.vstack([
        np.array([1,-1,1])/np.sqrt(3),
        np.array([1,1,0])/np.sqrt(2),
        np.array([1,-1,-2])/np.sqrt(6)
                ]).T

    A = S@D@S.T
    SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
    assert np.allclose(D,DRH)
    assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)

    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(100):
        A = np.random.random((5,5))
        A = 0.5*(A+A.T)
        S,D = diagRH(A,tol=1e-15,K=10000)
        ARH = S@D@S.T
        e = normaExacta(ARH-A, p='inf')
        if e < 1e-5:
            exitos += 1
    assert exitos >= 95

def test_labo7():
    # nucleo
    A = np.eye(3)
    S = nucleo(A)
    assert S.shape[0] == 0, "nucleo fallo para matriz identidad"
    A[1, 1] = 0
    S = nucleo(A)
    msg = "nucleo fallo para matriz con un cero en diagonal"
    assert esNucleo(A, S), msg
    assert S.shape == (3, 1), msg
    assert abs(S[2, 0]) < 1e-2, msg
    assert abs(S[0, 0]) < 1e-2, msg

    v = np.random.random(5)
    v = v / np.linalg.norm(v)
    H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
    S = nucleo(H)
    msg = "nucleo fallo para matriz de proyeccion ortogonal"
    assert S.shape == (5, 1), msg
    v_gen = S[:, 0]
    v_gen = v_gen / np.linalg.norm(v_gen)
    assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg

    # crea rala
    listado = [[0, 17], [3, 4], [0.5, 0.25]]
    A_rala_dict, dims = crea_rala(listado, 32, 89)
    assert dims == (32, 89), "crea_rala fallo en dimensiones"
    assert A_rala_dict[(0, 3)] == 0.5, "crea_rala fallo"
    assert A_rala_dict[(17, 4)] == 0.25, "crea_rala fallo"
    assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

    listado = [[32, 16, 5], [3, 4, 7], [7, 0.5, 0.25]]
    A_rala_dict, dims = crea_rala(listado, 50, 50)
    assert dims == (50, 50), "crea_rala fallo en dimensiones con tol"
    assert A_rala_dict.get((32, 3)) == 7
    assert A_rala_dict[(16, 4)] == 0.5
    assert A_rala_dict[(5, 7)] == 0.25

    listado = [[1, 2, 3], [4, 5, 6], [1e-20, 0.5, 0.25]]
    A_rala_dict, dims = crea_rala(listado, 10, 10)
    assert dims == (10, 10), "crea_rala fallo en dimensiones con tol"
    assert (1, 4) not in A_rala_dict
    assert A_rala_dict[(2, 5)] == 0.5
    assert A_rala_dict[(3, 6)] == 0.25
    assert len(A_rala_dict) == 2

    # caso borde: lista vacia. Esto es una matriz de 0s
    listado = []
    A_rala_dict, dims = crea_rala(listado, 10, 10)
    assert dims == (10, 10), "crea_rala fallo en dimensiones con lista vacia"
    assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

    # multiplica rala vector
    listado = [[0, 1, 2], [0, 1, 2], [1, 2, 3]]
    A_rala = crea_rala(listado, 3, 3)
    v = np.random.random(3)
    v = v / np.linalg.norm(v)
    res = multiplica_rala_vector(A_rala, v)
    A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    res_esperado = A @ v
    assert np.allclose(res, res_esperado), "multiplica_rala_vector fallo"

    A = np.random.random((5, 5))
    A = A * (A > 0.5)
    listado = [[], [], []]
    for i in range(5):
        for j in range(5):
            listado[0].append(i)
            listado[1].append(j)
            listado[2].append(A[i, j])

    A_rala = crea_rala(listado, 5, 5)
    v = np.random.random(5)
    assert np.allclose(multiplica_rala_vector(A_rala, v), A @ v)


def genera_matriz_para_test(m, n=2, tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m, n))
    else:
        A = np.random.random((m, tam_nucleo))
        A = np.hstack([A, A])
    return (A)

# Cambiamos tol acá
def test_svd_reducida_mn(A, tol=1e-14):
    m, n = A.shape
    hU, hS, hV = svd_reducida(A, tol=tol)
    nU, nS, nVT = np.linalg.svd(A)
    r = len(hS) + 1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU)) - 1) < 10 ** r * tol), 'Revisar calculo de hat U en ' + str(
        (m, n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV)) - 1) < 10 ** r * tol), 'Revisar calculo de hat V en ' + str(
        (m, n))
    assert len(hS) == len(nS[np.abs(nS) > tol]), 'Hay cantidades distintas de valores singulares en ' + str((m, n))
    assert np.all(
        np.abs(hS - nS[np.abs(nS) > tol]) < 10 ** r * tol), 'Hay diferencias en los valores singulares en ' + str(
        (m, n))

def test_labo8():
    # Matrices al azar
    for m in [2, 5, 10, 20]:
        for n in [2, 5, 10, 20]:
            for _ in range(10):
                A = genera_matriz_para_test(m, n)
                test_svd_reducida_mn(A)

    # Matrices con nucleo
    m = 12
    for tam_nucleo in [2, 4, 6]:
        for _ in range(10):
            A = genera_matriz_para_test(m, tam_nucleo=tam_nucleo)
            test_svd_reducida_mn(A)

    # Tamaños de las reducidas
    A = np.random.random((8, 6))
    for k in [1, 3, 5]:
        hU, hS, hV = svd_reducida(A, k=k)
        assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
        assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
        assert len(hS) == k, 'Tamaño de hS incorrecto'

correrTestsLabos()
