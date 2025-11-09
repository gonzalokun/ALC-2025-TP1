import numpy as np
import math

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
            else:
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


def esSimetrica(a, tol=1e-10):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    tras = traspuesta(a)

    dif = restar(a, tras)

    for i in range(filas):
        for j in range(columnas):
            if np.abs(dif[i][j]) >= tol:
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
        matriz[fila1][j] += num * matriz[fila2][j]


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
            result[i][j] = vector[(j - i) % vector.shape[0]]

    return result


def matrizVandermonde(vector):
    result = matrizDeCeros(vector.shape[0], vector.shape[0])

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[j] ** i

    return result


def numeroAureo(n):
    a = 0
    b = 1

    for i in range(n + 1):
        tmp = a
        a = b
        b += tmp

    if a == 0:
        return 0

    return b / a


def multiplicar(matrizA, matrizB):
    if matrizA.shape[1] != matrizB.shape[0]:
        raise Exception("No se puede :(")

    res = matrizDeCeros(matrizA.shape[0], matrizB.shape[1])

    for i in range(matrizA.shape[0]):
        for j in range(matrizB.shape[1]):
            for k in range(matrizA.shape[1]):
                res[i][j] += matrizA[i][k] * matrizB[k][j]

    return np.array(res)


def multiplacionMatricialDeVectores(vectorA, vectorB):
    res = np.zeros((vectorA.shape[0], vectorB.shape[0]))

    for i in range(vectorA.shape[0]):
        for j in range(vectorB.shape[0]):
            res[i][j] = vectorA[i] * vectorB[j]
    return res


def productoEscalar(vectorA, vectorB):
    if vectorA.shape[0] != vectorB.shape[0]:
        return None
    else:
        res = 0.0
        for i in range(vectorA.shape[0]):
            res += vectorA[i] * vectorB[i]
        return res


def vectorPorEscalar(x, s):
    res = []
    for i in range(len(x)):
        res.append(x[i] * s)
    return np.array(res)


# labo 1

def error(x, y):
    return abs(x - y)


def error_relativo(x, y):
    return abs(x - y) / abs(x)


def matricesIguales(A, B, tol=1e-08):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if A.shape != B.shape:
        return False

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i][j] - B[i][j]) >= tol:
                return False

    return True


# labo 2

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    res = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
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


def rota_y_escala(theta, s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    res = multiplicar(rota(theta), escala(s))
    return res


def afin(theta, s, b):
    """
    Recibe un angulo theta , una tira de numeros s (en R2) , y un vector
    b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor
    fijo b
    """
    matriz2x2 = rota_y_escala(theta, s)
    matriz3x3 = np.array(matrizDeCeros(3, 3))
    matriz3x3[:2, :2] = matriz2x2
    matriz3x3[0][2] = b[0]
    matriz3x3[1][2] = b[1]
    matriz3x3[2][2] = 1
    return np.array(matriz3x3)


def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a
    v
    """
    transf = afin(theta, s, b)
    vectorCon1 = np.append(v, 1)
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
        sum += abs(x[i]) ** p

    return sum ** (1 / p)


def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacios, y un escalar p. Devuelve
    una lista donde cada elemento corresponde a normalizar los
    elementos de X con la norma p.
    """
    vectoresNormalizados = []
    for i in range(len(X)):
        vectorActual = X[i]
        vectoresNormalizados.append(vectorPorEscalar(vectorActual, (1 / norma(vectorActual, p))))
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
    if not p in [1, 'inf']:
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

        if np.abs(numDiagonal) < 1e-08:
            return [None, None, 0]

        for fila2 in range(fila + 1, upper.shape[0]):
            nops += 1

            coef = upper[fila2][fila] / numDiagonal
            lower[fila2][fila] = coef

            upper[fila2][fila] = 0.0

            for columna in range(fila + 1, upper.shape[1]):
                upper[fila2][columna] = upper[fila2][columna] - coef * upper[fila][columna]
                nops += 2

    return [lower, upper, nops]


def res_tri(L, b, inferior=True):
    n = L.shape[1]
    x_vector = np.zeros(n)

    if inferior:
        for i in range(n):
            x_actual = b[i]
            # idea si es triangular inferior la solucion es de la pinta (b1/L11 , (b2-L21.X1)/L22 , b3-L32.X2-L31.X1   cada x_n se le restan todos los anteriores x_n
            for j in range(i):
                x_actual -= L[i][j] * x_vector[j]
            x_actual = x_actual / L[i][i]
            x_vector[i] = x_actual
    else:
        for i in range(n - 1, -1, -1):
            x_actual = b[i]
            for j in range(i + 1, n):
                x_actual -= L[i][j] * x_vector[j]
            x_actual = x_actual / L[i][i]
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
    nops = 0  # TODO: Ver si los nops estan bien calculados porque no hay tests sobre ellos
    n = A.shape[0]

    if n != A.shape[1]:
        return None

    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)

    for j in range(0, n):
        Q[:, j] = A[:, j]
        for k in range(j):
            R[k, j] = np.dot(Q[:, k], Q[:, j])
            nops += n * 2 - 1  # n mult y n-1 sumas
            Q[:, j] = Q[:, j] - vectorPorEscalar(Q[:, k], R[k, j])
            nops += n * 2  # n mult y n restas

        R[j, j] = norma(Q[:, j], 2)
        nops += n * 2  # n mult y n-1 sumas y una raiz
        Q[:, j] = vectorPorEscalar(Q[:, j], 1 / R[j, j])
        nops += n  # n mult

    if retorna_nops:
        return [Q, R, nops]

    return [Q, R]


def QR_con_HH(A, tol=1e-12):
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
            Hk = np.eye(m - k) - 2 * multiplacionMatricialDeVectores(u, u)
            Hk_p = np.eye(m)
            Hk_p[k:m, k:m] = Hk
            R = multiplicar(Hk_p, R)
            Q = multiplicar(Q, Hk_p)

    return [Q, R]


def calculaQR(A, metodo='RH', tol=1e-12):
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
        w = vectorPorEscalar(w, 1 / wNorma)
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
    while np.abs(e - 1) > tol and k < K:
        v = vPrima
        vPrima = aplicarMatrizKVecesYNormalizar(A, v, 2)
        e = productoEscalar(vPrima, v)
        k += 1

    autovalor = productoEscalar(vPrima, calcularAx(A, vPrima))

    # if autovalor < tol:
    #     autovalor = 0.0

    error = e - 1
    return [vPrima, autovalor, error]


def restarVectores(a, b):
    if a.shape != b.shape:
        return None

    res = np.zeros(a.shape)

    for i in range(a.shape[0]):
        res[i] = a[i] - b[i]

    return res


def diagRH(A, tol=1e-15, K=1000):
    if not esSimetrica(A):
        return None

    autovector, lamda, _ = metpot2k(A, tol, K)

    n = A.shape[0]
    u = restarVectores(np.eye(n)[0], autovector)
    uNormaAl2 = norma(u, 2) ** 2

    aRestar = matrizPorEscalar(multiplacionMatricialDeVectores(u, u), (2 / uNormaAl2))
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
    res = matrizDeCeros(n, n)

    for j in range(n):
        suma = 0

        for i in range(n):
            res[i, j] = np.random.rand()
            suma += res[i, j]

        if suma != 0:
            for i in range(n):
                res[i, j] = res[i, j] / suma

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
        B[:, j] = vectorPorEscalar(B[:, j], 1 / norma(B[:, j], 2))

    U_s = B

    return U_s, E_s, V_s
