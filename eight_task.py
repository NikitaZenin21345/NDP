import numpy as np
import matplotlib.pyplot as plt

def explicit_scheme(Nt,Nx, alpha, L, T, left_boundary_condition, right_boundary_condition, q):
    dx = L / Nx
    dt = T / Nt
    u = np.zeros((Nt + 1, Nx + 1))
    u_snapshots = [u[0].copy()]
    u[-1] = right_boundary_condition(0)
    u[0] = left_boundary_condition(0)
    for n in range(Nt):
        for i in range(1, Nx):
            u[n + 1, i] = u[n, i] + alpha * dt / dx ** 2 * (u[n, i - 1] - 2 * u[n, i] + u[n, i + 1]) + q * dt
        u[n + 1, 0], u[n + 1, -1] = left_boundary_condition(n * dt), right_boundary_condition(n * dt)
        if n % 200 == 0:
            u_snapshots.append(u[n+1].copy())
    return u_snapshots

L = 1.0
T = 20.0
Nx = 100
Nt = 5000
alpha = 0.01
dx = L / Nx
dt = T / Nt
q = 0

#u_snapshots = explicit_scheme(Nt,Nx, alpha, L, T, lambda t: 0 - 10 * t/ T, lambda t: 20 - 10 * t/ T, q)
u_snapshots = explicit_scheme(Nt, Nx, alpha, L, T, lambda t: 0, lambda t: 20, q)

plt.figure(figsize=(10, 8))
for i, snapshot in enumerate(u_snapshots):
    plt.plot(np.linspace(0, L, Nx + 1), snapshot, label=f't={i * 200 * dt:.2f}s')
plt.title('Temperature distribution explicit heat equation')
plt.xlabel('x,m')
plt.ylabel('T, C`')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

def heat_equation_implicit(N, M, L, T, alpha, q, right_boundary_condition, left_boundary_condition):
    dx = L / N
    dt = T / M
    r = alpha * dt / dx**2


    u = np.linspace(0, L, N+1)
    new_u = np.zeros(N+1)
    u[-1] = right_boundary_condition(0)
    u[0] = left_boundary_condition(0)

    A = np.zeros((N+1, N+1))
    np.fill_diagonal(A[1:-1, 1:-1], 1 + 2*r)
    np.fill_diagonal(A[1:-1, :-2], -r)
    np.fill_diagonal(A[1:-1, 2:], -r)
    A[0, 0], A[-1, -1] = 1, 1

    results = []
    for i in range(M):
        b = u + dt * q
        b[0], b[-1] = right_boundary_condition(i*dt), left_boundary_condition(i*dt)
        new_u = solve(A, b)
        u[:] = new_u
        results.append(u.copy())

    return results

L = 1.0
T = 20.0
Nx = 100
Nt = 5000
alpha = 0.01
q = 0.1

#results = heat_equation_implicit(N, M, L, T, alpha, q, lambda t: 0 - 10 * t/ T, lambda t: 20 - 10 * t/ T)
results = heat_equation_implicit(Nx, Nt, L, T, alpha, q, lambda t: 0 , lambda t: 20 )
x = np.linspace(0, L, Nx+1)
plt.figure(figsize=(10, 5))
for u in results[::len(results)//10]:
    plt.plot(x, u, label=f"t = {np.round(u.sum()/Nx, 2)}s")
plt.title('Temperature distribution implicit heat equation')
plt.xlabel('x,m')
plt.ylabel('T, C`')
plt.grid(True)
plt.legend()
plt.show()