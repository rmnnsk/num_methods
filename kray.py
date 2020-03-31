from time import sleep
from typing import Any, Union

import numpy as np


class Equation:

    # Инициализируем начальные переменные уравнения
    def __init__(self, p, q, f, aC, bC):
        self.p = p  # p(x)
        self.q = q  # q(x)
        self.f = f  # f(x)
        self.aCoef = aC  # a0, a1, a, A
        self.bCoef = bC  # b0, b1, b, B
        self.y = np.ndarray(())  # y(x) - массив решений

    def set_iterAmount(self, iA):
        self.iterAmount = iA

    # Печатаем решение
    def print_solution(self):
        x = self.aCoef[2]
        for i in range(self.iterAmount + 1):
            print("({0:.10f}; {1:.10f})".format(x, self.y[i]))
            x += self.step

    def boundary_solve(self):
        self.step = (self.bCoef[2] - self.aCoef[2] + 0.0) / self.iterAmount
        self.y = np.zeros((self.iterAmount + 1,), dtype=float)
        ksi = np.zeros((self.iterAmount + 1,), dtype=float)
        eta = np.zeros((self.iterAmount + 1,), dtype=float)
        x = self.aCoef[2]
        ksi[1] = -self.aCoef[1] / (self.aCoef[0] * self.step - self.aCoef[1])
        eta[1] = (self.aCoef[3] * self.step) / (self.aCoef[0] * self.step - self.aCoef[1])
        for i in range(1, self.iterAmount):
            x += self.step
            k = (1.0 / self.step ** 2) - (self.p(x) / (2 * self.step))
            l = (2.0 / self.step ** 2) - self.q(x)
            m = (1.0 / self.step ** 2) + (self.p(x) / (2 * self.step))
            n = self.f(x)
            ksi[i + 1] = m / (l - k * ksi[i])
            eta[i + 1] = (eta[i] * k - n) / (l - k * ksi[i])
        self.y[self.iterAmount] = (self.bCoef[1] * eta[self.iterAmount] + self.bCoef[3] * self.step) / \
                                  (self.bCoef[1] * (1.0 - ksi[self.iterAmount]) + self.bCoef[0] * self.step)
        for i in range(self.iterAmount, 0, -1):
            self.y[i - 1] = self.y[i] * ksi[i] + eta[i]

# y'' + y'/2 + 3y = 2x^2
p = lambda x: 0.5
q = lambda x: 3.0
f = lambda x: 2.0 * (float(x) ** 2)
     #a0    a1    a    A
aC = (1.0, -2.0, 1.0, 0.6)  # y(1) - 2y'(1) = 0.6
bC = (1.0, 0.0, 1.3, 1.0)  # y(1.3) = 1
eq = Equation(p, q, f, aC, bC)

if __name__ == "__main__":
    itAmount = int(input("Количество итераций: "))
    eq.set_iterAmount(itAmount)
    eq.boundary_solve()
    eq.print_solution()
