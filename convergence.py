"""Grid convergence test for the finite-difference QHO solver.

Ground-state error |E0 - 1/2| should scale as dx^2 (3-point stencil),
with the leading term predicted analytically as -dx^2/32.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

L = 8.0
Ns = [100, 200, 400, 800, 1600]
dxs, errs = [], []

for N in Ns:
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]                      # = 2L/(N-1)
    diag = 1.0 / dx**2 + 0.5 * x**2       # diagonal of H = -1/2 D2 + V
    off = -0.5 / dx**2 * np.ones(N - 1)   # off-diagonal of H
    E0 = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))[0][0]
    dxs.append(dx)
    errs.append(abs(E0 - 0.5))
    print(f"N={N:5d}  dx={dx:.4f}  E0={E0:.8f}  |E0-1/2|={abs(E0-0.5):.3e}  "
          f"predicted dx^2/32={dx**2/32:.3e}")

dxs, errs = np.array(dxs), np.array(errs)
slope = np.polyfit(np.log(dxs), np.log(errs), 1)[0]
print(f"fitted convergence order: {slope:.3f}  (expected 2)")

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(dxs, errs, "o-", label=f"numerical (slope {slope:.2f})")
ax.loglog(dxs, dxs**2 / 32, "--", label=r"predicted $dx^2/32$")
ax.set_xlabel("grid spacing dx")
ax.set_ylabel(r"$|E_0 - 1/2|$")
ax.set_title("QHO ground state: grid convergence")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/convergence.png", dpi=150)
print("saved figures/convergence.png")
