import math
import numpy as np

# Делаем красивый вывод
np.set_printoptions(formatter={'float': '{: 0.7f}'.format})

variant = int(input("Вариант: "))
SZ = 1
tx = -1
if variant == 1:
    SZ = int(input("Размер: "))
elif variant == 2:
    tx = float(input("Параметр х: "))
    SZ = 100




# Ax = b
A = np.ndarray(shape=(SZ, SZ), dtype=float)
invA = np.ndarray(shape=(SZ, SZ), dtype=float)
b = np.zeros((SZ,), dtype=float)
x = np.zeros((SZ,), dtype=float)


# Класс счетчика операций

class Counter:
    def __init__(self, name):
        self.add = 0
        self.mult = 0
        self.div = 0
        self.name = name

    def printInfo(self):
        print("Метод {0} выполнил {1} сложений, {2} умножений и {3} делений".format(
            self.name, self.add, self.mult, self.div))

    def addInfo(self, add=0, mult=0, div=0):
        self.add += add
        self.mult += mult
        self.div += div


gaussCnt = Counter("Гаусса")
iterCnt = Counter("Верхней релаксации")


def createMatrix():
    M = 4
    global SZ, tx
    qm = 1.001 - 2 * M * 1e-3
    tmpA = np.ndarray(shape=(SZ, SZ), dtype=float)
    tmpb = np.zeros((SZ,), dtype=float)
    # Генерируем матрицуч
    for ti in range(1, SZ + 1):
        for tj in range(1, SZ + 1):
            if ti != tj:
                tmpA[ti - 1][tj - 1] = qm ** (ti + tj) + 0.1 * (tj - ti)
            else:
                tmpA[ti - 1][tj - 1] = (qm - 1) ** (ti + tj)
    # Генерируем вектор правой части
    for ti in range(1, SZ + 1):
        tmpb[ti - 1] = SZ * math.exp(tx / ti) * math.cos(tx)
        # tmpb[ti - 1] = abs(tx - SZ/10.0) * ti * math.sin(tx)
    return tmpA, tmpb


def determ(matrix):
    ans = 1
    sgn = 1
    # Прямой ход метод Гаусса
    for ti in range(SZ - 1):
        # Ведущий элемент равен 0
        if matrix[ti][ti] == 0:
            # Ищем первый не 0 в столбике
            notZeros = np.nonzero(matrix[:, i][i:])[0]
            if notZeros.size == 0:
                return 0
            # Меняем местами строки
            goodRow = notZeros[0] + i
            matrix[[ti, goodRow]] = matrix[[goodRow, ti]]
            sgn *= -1
        # Вычитаем из следующих строчек текущую, умноженную на коэфициент

        for tj in range(ti + 1, SZ):
            matrix[tj] -= matrix[ti] * (matrix[tj][ti] / matrix[ti][ti])
    # Считаем определитель диагональной матрицы
    for ti in range(SZ):
        ans *= matrix[ti][ti]
    return ans * sgn

def matr_norm(matrix):
    sum = 0
    for i in range(SZ):
        for j in range(SZ):
            sum += matrix[i][j]**2
    return math.sqrt(sum)

def inverseMatrix(matrix):
    ansMatrix = np.identity(SZ, dtype=float)  # Единичная матрица
    # Прямой ход метода Гаусса
    for i in range(SZ - 1):
        # Ведущий элемент равен 0
        if matrix[i][i] == 0:
            # Ищем первый не 0 в столбике
            notZeros = np.nonzero(matrix[:, i][i:])[0]
            if notZeros.size == 0:
                ans = 0
                return
            # Меняем местами строки
            goodRow = notZeros[0] + i
            matrix[[i, goodRow]] = matrix[[goodRow, i]]
            ansMatrix[[i, goodRow]] = ansMatrix[[goodRow, i]]
        # Вычитаем из следующих строчек текущую, умноженную на коэфициент
        for j in range(i + 1, SZ):
            ansMatrix[j] -= ansMatrix[i] * (matrix[j][i] / matrix[i][i])
            matrix[j] -= matrix[i] * (matrix[j][i] / matrix[i][i])

    # Обратный ход метода Гаусса
    for i in range(SZ - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            ansMatrix[j] -= ansMatrix[i] * (matrix[j][i] / matrix[i][i])

    for i in range(SZ):
        ansMatrix[i] /= matrix[i][i]
    return ansMatrix


# Метод Гаусса
def GaussMethod(matrix, b):
    ansMatrix = b  # Вектор правой части
    # Прямой ход метода Гаусса
    for i in range(SZ - 1):
        # Ведущий элемент равен 0
        if matrix[i][i] == 0:
            # Ищем первый не 0 в столбике
            notZeros = np.nonzero(matrix[:, i][i:])[0]
            if notZeros.size == 0:
                ans = 0
                return
            # Меняем местами строки
            goodRow = notZeros[0] + i
            matrix[[i, goodRow]] = matrix[[goodRow, i]]
            ansMatrix[[i, goodRow]] = ansMatrix[[goodRow, i]]
        # Вычитаем из следующих строчек текущую, умноженную на коэфициент
        for j in range(i + 1, SZ):
            ansMatrix[j] -= ansMatrix[i] * (matrix[j][i] / matrix[i][i])
            matrix[j] -= matrix[i] * (matrix[j][i] / matrix[i][i])
            gaussCnt.addInfo(SZ, SZ, 1)
    # Обратный ход метода Гаусса
    for i in range(SZ - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            ansMatrix[j] -= ansMatrix[i] * (matrix[j][i] / matrix[i][i])
            gaussCnt.addInfo(SZ, SZ, 1)
    for i in range(SZ):
        ansMatrix[i] /= matrix[i][i]
        gaussCnt.addInfo(div=1)
    return ansMatrix


def GaussWithSelect(matrix, b):
    ansMatrix = b  # Вектор правой части
    # Прямой ход метода Гаусса
    for i in range(SZ - 1):
        # Ищем максимальный элемент в столбике
        goodRow = np.argmax(np.absolute(matrix[:, i][i:])) + i
        matrix[[i, goodRow]] = matrix[[goodRow, i]]
        ansMatrix[[i, goodRow]] = ansMatrix[[goodRow, i]]

        # Вычитаем из следующих строчек текущую, умноженную на коэфициент
        for j in range(i + 1, SZ):
            ansMatrix[j] -= ansMatrix[i] * (matrix[j][i] / matrix[i][i])
            matrix[j] -= matrix[i] * (matrix[j][i] / matrix[i][i])

    # Обратный ход метода Гаусса
    for ti in range(SZ - 1, 0, -1):
        for tj in range(ti - 1, -1, -1):
            ansMatrix[tj] -= ansMatrix[ti] * (matrix[tj][ti] / matrix[ti][ti])

    for i in range(SZ):
        ansMatrix[i] /= matrix[i][i]
    return ansMatrix


#################################################################################

# Метод итераций

# Норма разности векторов
def diffNorm(v, matr):
    diff = 0.0
    global b
    for ti in range(SZ):
        now = 0.0
        for tj in range(SZ):
            now += matr[ti][tj] * v[tj]
        now -= b[ti]
        diff += now ** 2
    return math.sqrt(diff)

eqMatr = np.ndarray((1,))

def nextIter(omega, itCnt):
    global b, ansX, eqMatr
    newAns = np.zeros((SZ,), dtype=float)
    for ti in range(SZ):
        sum1 = 0.0
        for tj in range(ti):
            sum1 += eqMatr[ti][tj] * newAns[tj]
            itCnt.addInfo(add=1, mult=1)
        for tj in range(ti, SZ):
            sum1 += (eqMatr[ti][tj] * ansX[tj])
            itCnt.addInfo(add=1, mult=1)
        newAns[ti] = ansX[ti] + omega / eqMatr[ti][ti] * (b[ti] - sum1)
        itCnt.addInfo(add=2, mult=2, div=1)
    ansX = newAns
    # print("new batya: ", ansX)
    return diffNorm(ansX, eqMatr)


# Проверить данную омегу на сходимость
def checkOmega(omega):
    tmpCnt = Counter("Верхней релаксации")
    global isConverge, minIter, omegaAns, iterCnt, ansX, EPS, eqMatr
    isConverge = True
    curIter = 1

    while nextIter(omega, tmpCnt) > EPS:
        curIter += 1
        if curIter >= MAXITER:
            break
    for ti in range(SZ):
        if not np.isfinite(ansX[ti]):
            isConverge = False
    if isConverge:
        print("Сходится при омега = {0}, за {1} итераций".format(omega, curIter))
        if curIter < minIter:
            iterCnt = tmpCnt
            minIter = curIter
            omegaAns = omega
    else:
        print("Расходится при омега = ", omega)


######################################################################
if __name__ == "__main__":
    if variant == 1:
        input_data = input().split('\n')

        for i in range(SZ):
            tmp = list(input_data[i].split())
            A[i] = tmp[0: -1]
            b[i] = tmp[-1]
    elif variant == 2:
        A, b = createMatrix()
        print("newSZ : ", SZ)
        ans = np.zeros((SZ,), dtype=float)
    else:
        print("bad variant")

    print(A)
    det = determ(A.copy())
    print("Определитель = {0:.10f}".format(det))
    print("Обратная матрица: \n", inverseMatrix(A.copy()))
    norm1 = matr_norm(A.copy())
    norm2 = matr_norm(inverseMatrix(A.copy()))
    print("Число обусловленности: ", norm1 * norm2)
    print("Решаем СЛАУ обычным Гауссом: ", GaussMethod(A.copy(), b.copy()))
    print("Решаем СЛАУ Гауссом с выбором: ", GaussWithSelect(A.copy(), b.copy()))
    gaussCnt.printInfo()
    print("Решаем с помощью пакета Numpy: ", np.linalg.solve(A.copy(), b.copy()))

    EPS = 0.001  # Погрешность
    MAXITER = 40000  # Максимальное число итераций
    isConverge = False  # Флаг сходимостиок
    minIter = MAXITER  # Лучшее кол-во итераций
    omegaAns = 0.0  # Лучшая омега

    magicMatrix = np.matmul(A.transpose(), A)
    b = np.matmul(A.transpose(), b)
    ansX = np.zeros((SZ,), dtype=float)
    eqMatr = magicMatrix

    for omg in np.arange(1.1, 2.0, 0.1):
        isConverge = True
        checkOmega(omg)

    if minIter != MAXITER or diffNorm(ansX, magicMatrix) < EPS:
        print("Лучшая сходимость при омега = {0}, за {1} итераций".format(omegaAns, minIter))
        print("Решение СЛАУ методом верхней релаксации: ", end="")
        print(ansX)
        iterCnt.printInfo()
    else:
        print("Метод расходится(много итераций)")
