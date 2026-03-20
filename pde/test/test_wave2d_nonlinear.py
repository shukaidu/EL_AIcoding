"""验证 IMEX Strang splitting 正确性：一致性、质量守恒、收敛阶。"""
import sys
import os
import unittest
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pde.wave_2d_nonlinear import setup_wave2d_nonlinear

COMMON = dict(Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32,
              g=9.8, h0=1.0, f_coriolis=0.0, nu_h=0.0, nu_q=0.0,
              nudging_coeff=0.0, initial_condition="random", rng_seed=42)

_dx      = COMMON["Lx"] / COMMON["nx"]
_c       = np.sqrt(COMMON["g"] * COMMON["h0"])
DT_TEST  = 0.5 * _dx / _c


class TestWave2DNonlinear(unittest.TestCase):

    def test_imex_rk4_consistency(self):
        """相同初始条件下 IMEX 与 RK4 的 h 场相对 L2 误差应 < 1e-3。"""
        h_i, qx_i, qy_i, _, adv_i, dt, _, _ = setup_wave2d_nonlinear(
            **COMMON, integrator="imex", dt=DT_TEST)
        h_r, qx_r, qy_r, _, adv_r, _,  _, _ = setup_wave2d_nonlinear(
            **COMMON, integrator="rk4",  dt=DT_TEST)
        h_i, qx_i, qy_i = adv_i(h_i, qx_i, qy_i, dt, 20)
        h_r, qx_r, qy_r = adv_r(h_r, qx_r, qy_r, dt, 20)
        rel_err = np.linalg.norm(h_i - h_r) / np.linalg.norm(h_r)
        self.assertLess(rel_err, 1e-3, f"rel L2 = {rel_err:.2e}")

    def test_mass_conservation(self):
        """IMEX 积分 100 步后质量漂移应 < 1e-10。"""
        h, qx, qy, _, adv, dt, _, _ = setup_wave2d_nonlinear(
            **COMMON, integrator="imex", dt=DT_TEST)
        mass0 = np.mean(h)
        h, qx, qy = adv(h, qx, qy, dt, 100)
        drift = abs(np.mean(h) - mass0)
        self.assertLess(drift, 1e-10, f"drift = {drift:.2e}")

    def test_convergence_order(self):
        """IMEX 2阶格式，加密步数后误差比应 > 3.0（期望 ~4）。"""
        CONV = dict(Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32,
                    g=9.8, h0=1.0, f_coriolis=0.0, nu_h=0.0, nu_q=0.0,
                    nudging_coeff=0.0, initial_condition="ring", rng_seed=1)
        dt_base = 0.5 * (CONV["Lx"] / CONV["nx"]) / np.sqrt(CONV["g"] * CONV["h0"])
        T       = 5 * dt_base

        def run_conv(n_steps):
            dt_run = T / n_steps
            h, qx, qy, _, adv, _, _, _ = setup_wave2d_nonlinear(
                **CONV, integrator="imex", dt=dt_run)
            return adv(h, qx, qy, dt_run, n_steps)[0]

        h_ref = run_conv(640)
        ns    = [5, 10, 20, 40, 80, 160]
        errs  = [np.linalg.norm(run_conv(n) - h_ref) for n in ns]
        ratio = errs[1] / errs[2]
        self.assertGreater(ratio, 3.0, f"convergence ratio = {ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
