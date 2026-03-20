"""验证 Burgers 1D 求解器：质量守恒、能量耗散、收敛阶。"""
import sys
import os
import unittest
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pde.burgers_1d import setup_burger, integrate_burger, build_diffusion_matrix

L  = 2 * np.pi
nx = 128
dx = L / nx
nu = 1e-3
dt = 0.5 * dx


def _run(n_steps, rng_seed=7, nx_=128, nu_=1e-3, dt_=None):
    dx_ = L / nx_
    dt_ = dt_ or 0.5 * dx_
    A = build_diffusion_matrix(nx_, dt_, dx_, nu_)
    u, _, _ = setup_burger(nx_, dx_, dt_, L, nu_, alpha=2.5, u_mean=0.0, rng_seed=rng_seed)
    for _ in range(n_steps):
        u = integrate_burger(u, dt_, dx_, nu_, A)
    return u


class TestBurgers1D(unittest.TestCase):

    def test_mass_conservation(self):
        """周期边界下平均值应精确守恒。"""
        A  = build_diffusion_matrix(nx, dt, dx, nu)
        u0, _, _ = setup_burger(nx, dx, dt, L, nu, alpha=2.5, u_mean=0.0, rng_seed=7)
        mass0 = np.mean(u0)
        u = u0.copy()
        for _ in range(200):
            u = integrate_burger(u, dt, dx, nu, A)
        drift = abs(np.mean(u) - mass0)
        self.assertLess(drift, 1e-12, f"mass drift = {drift:.2e}")

    def test_energy_dissipation(self):
        """nu > 0 时 ||u|| 应随时间下降。"""
        A  = build_diffusion_matrix(nx, dt, dx, nu)
        u0, _, _ = setup_burger(nx, dx, dt, L, nu, alpha=2.5, u_mean=0.0, rng_seed=7)
        e0 = np.linalg.norm(u0)
        u  = u0.copy()
        for _ in range(500):
            u = integrate_burger(u, dt, dx, nu, A)
        self.assertLess(np.linalg.norm(u), e0, "energy did not decrease")

    def test_convergence_order(self):
        """纯扩散极限下1阶时间格式，加密步数后误差比应 > 1.8。"""
        nx2 = 128
        dx2 = L / nx2
        nu2 = 0.01
        T   = 1.0

        def run_conv(n_steps):
            dt_r = T / n_steps
            A_r  = build_diffusion_matrix(nx2, dt_r, dx2, nu2)
            u, _, _ = setup_burger(nx2, dx2, dt_r, L, nu2, alpha=3.0, u_mean=0.0, rng_seed=99)
            u *= 0.01
            for _ in range(n_steps):
                u = integrate_burger(u, dt_r, dx2, nu2, A_r)
            return u

        u_ref = run_conv(2000)
        ns    = [25, 50, 100, 200]
        errs  = [np.linalg.norm(run_conv(n) - u_ref) for n in ns]
        ratio = errs[0] / errs[1]
        self.assertGreater(ratio, 1.8, f"convergence ratio = {ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
