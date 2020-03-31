import math

import numpy as np

test = -1  # Номер теста
iterAmount = -1  # Кол-во итераций
eqAmount = -1  # Кол-во уравнений
step = -1  # Размер шага
x0 = 0.0
y0 = np.zeros(())
functions = []  # Массив функций


# Уравнение 1-6
def f1_6(x, y):
    return (x - x ** 2) * y[0]


def init_f1_6():
    global eqAmount, x0, y0
    eqAmount = 1
    x0 = 0
    y0 = np.zeros((eqAmount,), dtype=float)
    y0[0] = 1


def init_f2_17():
    global eqAmount, x0, y0
    eqAmount = 2
    y0 = np.zeros((eqAmount,), dtype=float)
    x0 = 0.0
    y0[0] = 1.0
    y0[1] = 0.5


def f2_17_1(x, y):
    return math.sin(1.4 * (y[0] ** 2)) - x + y[1]


def f2_17_2(x, y):
    return x + y[0] - 2.2 * (y[1] ** 2) + 1


def initTest(test):
    if test == 6:
        init_f1_6()
        functions.append(f1_6)
    elif test == 17:
        init_f2_17()
        functions.append(f2_17_1)
        functions.append(f2_17_2)


def printVec(x, y):
    print("({0:.10f}".format(x), end = '')
    for i in range(eqAmount):
        print("; {0:.10f}".format(y[i]), end='')
    print(")")


# Метод Рунге-Кутта второго порядка
def RungeKutta2():
    global x0, y0
    k = np.zeros((eqAmount,), dtype=float)
    y = np.zeros((eqAmount,), dtype=float)
    printVec(x0, y0)
    for i in range(iterAmount):
        for j in range(eqAmount):
            k[j] = y0[j] + functions[j](x0, y0) * step
        for j in range(eqAmount):
            y[j] = y0[j] + (functions[j](x0, y0) + functions[j](x0 + step, k)) * step / 2
        x0 += step
        printVec(x0, y)
        y0 = y

#
def RungeKutta4():
    global x0, y0
    y = np.zeros((eqAmount,), dtype=float)
    k1 = np.zeros((eqAmount,), dtype=float)
    k2 = np.zeros((eqAmount,), dtype=float)
    k3 = np.zeros((eqAmount,), dtype=float)
    k4 = np.zeros((eqAmount,), dtype=float)
    printVec(x0, y0)
    for i in range(iterAmount):
        for j in range(eqAmount):
            k1[j] = functions[j](x0, y0)
            y[j] = y0[j] + step * k1[j] / 2
        for j in range(eqAmount):
            k2[j] = functions[j](x0 + step / 2, y)
            y[j] = y0[j] + step * k2[j] / 2
        for j in range(eqAmount):
            k3[j] = functions[j](x0 + step / 2, y0)
            y[j] = y0[j] + step * k3[j]
        for j in range(eqAmount):
            k4[j] = functions[j](x0 + step, y0)
            y[j] = y0[j] + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) * step / 6
        x0 += step
        printVec(x0, y)
        y0 = y


if __name__ == "__main__":
    test = int(input("Номер теста(6, 17): "))
    iterAmount = int(input("Кол-во итераций: "))
    step = float(input("Размер шага: "))
    initTest(test)
    print("Рунге-Кутта четвертого порядка: ")
    RungeKutta4()
    initTest(test)
    print("Рунге-Кутта второго порядка: ")
    RungeKutta2()

