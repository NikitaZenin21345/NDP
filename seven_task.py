import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#left_integrate_border = 0
#right_integrate_border = 1
#lam = 0.5
#step = 1/20

#x = np.arange(left_integrate_border, right_integrate_border, step).reshape(-1, 1)
#n = len(x)

#K = lambda x_, s: x_ * s
#f = lambda x_: 5 * x_ / 6
#y_exact = lambda x_: x_
#y_exact_values = y_exact(x)
left_integrate_border = 0
right_integrate_border = 1
lam = -1
step = 1/20

x = np.arange(left_integrate_border, right_integrate_border, step).reshape(-1, 1)

f = lambda t: np.exp(t) - t
K = lambda x_, s: x_ * (np.exp(x_ * s) - 1)
y_exact_values = np.ones(len(x))

def Fredholm_method(K, f, a, b, h):
    x = np.arange(a, b, h).reshape(-1, 1)
    n = len(x)
    wt = 1/2
    wj = 1
    A = np.zeros((n, n))

    for i in range(n):
        A[i][0] = -h * wt * lam * K(x[i], x[0])
        for j in range(1, n-1):
            A[i][j] = -h * wj * lam * K(x[i], x[j])
        A[i][n-1] = -h * wt * lam * K(x[i], x[n-1])
        A[i][i] += 1

    B = np.zeros((n, 1))
    for j in range(n):
        B[j][0] = f(x[j])

    y_approx = np.linalg.solve(A, B)
    return y_approx

y_approx = Fredholm_method(K, f, left_integrate_border, right_integrate_border, step)
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact_values, '-g', linewidth=2, label='exact')
plt.plot(x, y_approx, 'or', label='approx')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1, 1), loc='best')
plt.ylim([0, 2])
plt.show()

left_integrate_border = 0
right_integrate_border = 1
lam = -1
step = 1/20

x = np.arange(left_integrate_border, right_integrate_border, step).reshape(-1, 1)

alpha = lambda x: [x**2, x**3, x**4]
beta = lambda s: [s, s**2/2, s**3/6]
f = lambda t: np.exp(t) - t
y_exact = np.ones(len(x))

def bfun(t, m, f):
    return beta(t)[m] * f(t)

def Aijfun(t, m, k):
    return beta(t)[m] * alpha(t)[k]

def Solve(f, t, Lambda, a, b):
    m = len(alpha(0))
    M = np.zeros((m, m))
    r = np.zeros((m, 1))
    for i in range(m):
        r[i] = integrate.quad(bfun, a, b, args=(i, f))[0]
        for j in range(m):
            M[i][j] = -Lambda * integrate.quad(Aijfun, a, b, args=(i, j))[0]
    for i in range(m):
        M[i][i] = M[i][i] + 1
    c = np.linalg.solve(M, r)
    return Lambda * (c[0] * alpha(t)[0] + c[1] * alpha(t)[1]) + f(t)


y_approx = Solve(f, x, lam, left_integrate_border, right_integrate_border)
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, '-g', linewidth=2, label='exact')
plt.plot(x, y_approx, 'or', label='approx')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1, 1), loc='best')
plt.ylim([0, max(y_exact) + 0.2])
plt.show()

a = -1
b = 1
lam = 1
step = 1/20

x = np.arange(a, b, step).reshape(-1, 1)
n = len(x)

y_func = lambda x_: 6 * x_**2 + 1
y_exact = y_func(x)

phi_1 = lambda t: t
phi_2 = lambda t: t**2
psi_1 = lambda s: 1
psi_2 = lambda s: s

phi = [phi_1, phi_2]
psi = [psi_1, psi_2]


K = lambda x, s: x**2 + x*s
f = lambda s: 1

integral_func1 = lambda x, phi, psi: phi(x)*psi(x)
integral_func2 = lambda x, s, phi, psi: psi(x)*K(x,s)*phi(s)
integral_func3 = lambda x, s, psi: psi(x)*K(x,s)*f(s)

def aij(a, b, phi_i, psi_i, phi_j, func1, func2):
    first_integral = integrate.quad(func1, a, b, args=(phi_j, psi_i))
    second_integral = integrate.dblquad(func2, a, b, a, b, args=(phi_j, psi_i))
    return first_integral[0] - lam * second_integral[0]

def bi(a, b, psi_i, func):
    return lam * integrate.dblquad(func, a, b, a, b, args=(psi_i,))[0]


def GalerkinPetrov(t, a, b):
    n = len(psi)
    matrix_a = np.zeros((n,n))
    vector_b = np.zeros((n,1))
    for i in range(n):
        vector_b[i] = bi(a, b, psi[i], integral_func3)
        for j in range(n):
            matrix_a[i][j] = aij(a, b, phi[i], psi[i], phi[j], integral_func1, integral_func2)
    c = np.linalg.solve(matrix_a, vector_b)
    return 1 + c[0]*phi[0](t) + c[1]*phi[1](t)


y_approx = GalerkinPetrov(x, a, b)
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, '-g', linewidth=2, label='exact')
plt.plot(x, y_approx, 'or', label='approx')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1, 1), loc='best')
plt.show()
