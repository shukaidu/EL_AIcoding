"""验证 wave_2d_linear 求解器：能量守恒、色散关系、收敛阶。"""
import sys
import os
import unittest
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pde.wave_2d_linear import setup_wave2d, advance_tscreen

NX, NY = 128, 128
Lx, Ly = 2 * np.pi, 2 * np.pi
c  = 1.0
dt = 0.5 * (Lx / NX) / c


class TestWave2DLinear(unittest.TestCase):

    def test_energy_conservation(self):
        """无耗散谱方法，能量相对漂移应 < 1e-3。"""
        uhat, vhat, omega2, *_ = setup_wave2d(NX, NY, Lx, Ly, c=c,
                                               initial_condition="ring", rng_seed=1)
        K2 = -omega2 / c**2
        E0 = np.sum(np.abs(vhat)**2 + c**2 * K2 * np.abs(uhat)**2)
        uhat, vhat, _, _ = advance_tscreen(uhat, vhat, omega2, dt, 20)
        E1 = np.sum(np.abs(vhat)**2 + c**2 * K2 * np.abs(uhat)**2)
        rel_drift = abs(E1 - E0) / E0
        self.assertLess(rel_drift, 1e-3, f"rel drift = {rel_drift:.2e}")

    def test_dispersion_relation(self):
        """激发 (kx,ky)=(2,0) 单模，积分 T/4 后 u 应接近零点。"""
        kx_mode = 2
        x = np.linspace(0, Lx, NX, endpoint=False)
        y = np.linspace(0, Ly, NY, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        k         = 2 * np.pi / Lx * kx_mode
        T_exact   = 2 * np.pi / (c * k)
        u_single  = np.cos(kx_mode * 2 * np.pi / Lx * xx)
        uhat2     = np.fft.fft2(u_single)
        vhat2     = np.fft.fft2(np.zeros_like(u_single))
        kx2 = (2*np.pi/Lx) * np.concatenate([np.arange(0, NX//2+1), np.arange(-NX//2+1, 0)])
        ky2 = (2*np.pi/Ly) * np.concatenate([np.arange(0, NY//2+1), np.arange(-NY//2+1, 0)])
        KX2, KY2  = np.meshgrid(kx2, ky2, indexing="ij")
        omega2_2  = -(KX2**2 + KY2**2) * c**2
        n_quarter = max(1, round(T_exact / 4 / dt))
        _, _, u2, _ = advance_tscreen(uhat2, vhat2, omega2_2, dt, n_quarter)
        self.assertLess(np.mean(np.abs(u2)), 0.2, f"|u| at T/4 = {np.mean(np.abs(u2)):.4f}")

    def test_convergence_order(self):
        """RK4 积分应达到 4 阶收敛，加密步数后误差比应 > 10。"""
        NX_c, NY_c = 64, 64
        T_conv = 2.0

        def run_conv(n_steps):
            dt_r = T_conv / n_steps
            uh, vh, om2, *_ = setup_wave2d(NX_c, NY_c, Lx, Ly, c=c,
                                            initial_condition="ring", rng_seed=3)
            _, _, u, _ = advance_tscreen(uh, vh, om2, dt_r, n_steps)
            return u

        u_ref = run_conv(2560)
        ns    = [40, 80, 160, 320, 640]
        errs  = [np.linalg.norm(run_conv(n) - u_ref) for n in ns]
        ratio = errs[1] / errs[2]
        self.assertGreater(ratio, 10.0, f"convergence ratio = {ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
