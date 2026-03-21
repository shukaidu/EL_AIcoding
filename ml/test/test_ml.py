"""
ml 模块单元测试
覆盖：load_wave_2d_nonlinear（ml/data_io.py）、
      模型前向传播形状与反向传播（UNet、CNN、MLP）、snapshot 保存/加载

运行方式：
    python -m unittest discover ml/test/ -v
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ml.data_io import load_wave_2d_nonlinear
from ml.models import CNN, UNet, MLP
from ml.snapshot import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# 1. load_wave_2d_nonlinear — 测真实函数，mock 文件读取，关闭 DataLoader shuffle
#    （关闭 shuffle 后可逐样本精确比较，避免 sort 掩盖顺序错误）
# ---------------------------------------------------------------------------

def _make_fake_mat(N=40, C=3, Nx=12, nx=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "input_tensor":  rng.standard_normal((N, C, Nx, Nx)).astype(np.float32),
        "output_tensor": rng.standard_normal((N, C, nx, nx)).astype(np.float32),
    }


_OrigDataLoader = torch.utils.data.DataLoader  # patch 前保存原始引用，避免递归

def _no_shuffle_dl(ds, batch_size, shuffle):
    """替换 DataLoader，强制 shuffle=False，让测试中样本顺序确定。"""
    return _OrigDataLoader(ds, batch_size=batch_size, shuffle=False)


class TestLoadWave2dNonlinear(unittest.TestCase):

    def _load(self, mat, residual=False):
        device = torch.device("cpu")
        with patch("scipy.io.loadmat", return_value=mat), \
             patch.object(torch.utils.data, "DataLoader", _no_shuffle_dl):
            return load_wave_2d_nonlinear("fake.mat", device, 100, 0.2, residual)

    def test_loader_tensor_shapes(self):
        """train/val loader 第一批数据的 tensor shape 正确。"""
        mat = _make_fake_mat(N=40, Nx=12, nx=8)
        tl, vl, *_ = self._load(mat)
        xb, yb = next(iter(tl))
        self.assertEqual(tuple(xb.shape[1:]), (3, 12, 12))
        self.assertEqual(tuple(yb.shape[1:]), (3,  8,  8))

    def test_stats_shape(self):
        mat = _make_fake_mat()
        *_, stats = self._load(mat)
        self.assertEqual(stats["ch_mean"].shape, (3,))
        self.assertEqual(stats["ch_std"].shape,  (3,))

    def test_std_no_zero(self):
        mat = _make_fake_mat()
        mat["input_tensor"][:, 1, :, :] = 0.0  # 通道1方差=0
        *_, stats = self._load(mat)
        self.assertTrue((stats["ch_std"] >= 1e-8).all())

    def test_residual_false_target_exact(self):
        """residual=False，shuffle 关闭后可精确比较每个训练样本的 target。"""
        from sklearn.model_selection import train_test_split as _split
        N = 40
        mat = _make_fake_mat(N=N)
        tl, *_, stats = self._load(mat, residual=False)

        xi, xo = mat["input_tensor"], mat["output_tensor"]
        idx_tr, _ = _split(np.arange(N), test_size=0.2, random_state=42)
        mean = stats["ch_mean"].reshape(1, -1, 1, 1)
        std  = stats["ch_std"].reshape(1, -1, 1, 1)
        expected = ((xo - mean) / std)[idx_tr]  # 与 loader 同顺序

        actual = torch.cat([t for _, t in tl], dim=0).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_residual_true_target_exact(self):
        """residual=True，target = xo_norm - center_crop(xi_norm)，逐样本精确验证。"""
        from sklearn.model_selection import train_test_split as _split
        N = 40
        mat = _make_fake_mat(N=N)
        tl, vl, _, _, _, Nx, _, nx, _, stats = self._load(mat, residual=True)

        xi, xo = mat["input_tensor"], mat["output_tensor"]
        idx_tr, _ = _split(np.arange(N), test_size=0.2, random_state=42)
        mean = stats["ch_mean"].reshape(1, -1, 1, 1)
        std  = stats["ch_std"].reshape(1, -1, 1, 1)
        xi_norm = (xi - mean) / std
        xo_norm = (xo - mean) / std
        o = (Nx - nx) // 2
        expected = (xo_norm - xi_norm[:, :, o:o + nx, o:o + nx])[idx_tr]

        actual = torch.cat([t for _, t in tl], dim=0).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_residual_modes_differ(self):
        mat = _make_fake_mat()
        tl_false, *_ = self._load(mat, residual=False)
        tl_true,  *_ = self._load(mat, residual=True)
        t_false = torch.cat([t for _, t in tl_false], dim=0)
        t_true  = torch.cat([t for _, t in tl_true],  dim=0)
        self.assertFalse(torch.allclose(t_false, t_true))


# ---------------------------------------------------------------------------
# 4. 模型前向传播形状与反向传播
#    使用固定小参数（与 config 解耦），快速、无外部依赖
# ---------------------------------------------------------------------------

# 固定小参数：nst=4, nwd=8 → patch_side=16，base=8
_T_NWD   = 8
_T_PSZ   = 16   # patch_side = nwd + 2*nst = 8 + 2*4
_T_BASE  = 8
_T_CIN   = 3
_T_COUT  = 3
_T_B     = 2


class TestUNetShape(unittest.TestCase):

    def _fwd(self, pooling="max"):
        model = UNet(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE,
                     Nx=_T_PSZ, nx=_T_NWD, pooling=pooling)
        x = torch.zeros(_T_B, _T_CIN, _T_PSZ, _T_PSZ)
        with torch.no_grad():
            return model(x)

    def test_max_pooling_shape(self):
        self.assertEqual(self._fwd("max").shape,    (_T_B, _T_COUT, _T_NWD, _T_NWD))

    def test_avg_pooling_shape(self):
        self.assertEqual(self._fwd("avg").shape,    (_T_B, _T_COUT, _T_NWD, _T_NWD))

    def test_stride_pooling_shape(self):
        self.assertEqual(self._fwd("stride").shape, (_T_B, _T_COUT, _T_NWD, _T_NWD))

    def test_invalid_pooling_raises(self):
        with self.assertRaises(ValueError):
            UNet(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE,
                 Nx=_T_PSZ, nx=_T_NWD, pooling="unknown")

    def test_output_finite(self):
        self.assertTrue(torch.isfinite(self._fwd()).all())

    def test_backward_pass(self):
        model = UNet(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE, Nx=_T_PSZ, nx=_T_NWD, pooling="max")
        x = torch.randn(_T_B, _T_CIN, _T_PSZ, _T_PSZ, requires_grad=True)
        model(x).mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestCNNShape(unittest.TestCase):

    def test_output_shape(self):
        model = CNN(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE, Nx=_T_PSZ, nx=_T_NWD)
        with torch.no_grad():
            out = model(torch.zeros(_T_B, _T_CIN, _T_PSZ, _T_PSZ))
        self.assertEqual(out.shape, (_T_B, _T_COUT, _T_NWD, _T_NWD))

    def test_invalid_halo_raises(self):
        """(Nx - nx) % 4 != 0 时 CNN.__init__ 触发 assert。"""
        with self.assertRaises(AssertionError):
            CNN(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE, Nx=17, nx=8)  # 17-8=9，9%4≠0

    def test_output_finite(self):
        model = CNN(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE, Nx=_T_PSZ, nx=_T_NWD)
        with torch.no_grad():
            self.assertTrue(torch.isfinite(
                model(torch.zeros(_T_B, _T_CIN, _T_PSZ, _T_PSZ))).all())

    def test_backward_pass(self):
        model = CNN(Cin=_T_CIN, Cout=_T_COUT, base=_T_BASE, Nx=_T_PSZ, nx=_T_NWD)
        x = torch.randn(_T_B, _T_CIN, _T_PSZ, _T_PSZ, requires_grad=True)
        model(x).mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestMLPShape(unittest.TestCase):

    def test_output_shape(self):
        model = MLP(128, 64, 64, 3, "relu")
        with torch.no_grad():
            self.assertEqual(model(torch.zeros(8, 128)).shape, (8, 64))

    def test_batch_size_varies(self):
        model = MLP(32, 16, 32, 2, "relu")
        for B in [1, 5, 100]:
            with torch.no_grad():
                self.assertEqual(model(torch.zeros(B, 32)).shape, (B, 16))

    def test_tanh_activation(self):
        model = MLP(64, 32, 64, 2, "tanh")
        with torch.no_grad():
            self.assertEqual(model(torch.randn(4, 64)).shape, (4, 32))

    def test_invalid_activation_raises(self):
        with self.assertRaises(KeyError):
            MLP(64, 32, 64, 2, "sigmoid")

    def test_backward_pass(self):
        model = MLP(32, 16, 32, 2, "relu")
        x = torch.randn(4, 32, requires_grad=True)
        model(x).mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


# ---------------------------------------------------------------------------
# 5. Snapshot
#    注：load_checkpoint 只恢复 model weights，不恢复 optimizer state。
# ---------------------------------------------------------------------------

class TestSnapshot(unittest.TestCase):

    def _make_model_and_opt(self):
        model = MLP(16, 8, 16, 2, "relu")
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, opt

    def test_model_weights_restored_after_reload(self):
        """load_checkpoint 完整恢复所有模型参数（不含 optimizer state）。"""
        model, opt = self._make_model_and_opt()
        model(torch.randn(4, 16)).sum().backward()
        opt.step()

        original = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, opt, epoch=1, hist_tr=[0.5], hist_te=[0.4], path=path)
            new_model, _ = self._make_model_and_opt()
            load_checkpoint(new_model, path)

            # key 集合完全一致（完整加载，无缺失无多余）
            self.assertEqual(set(new_model.state_dict().keys()), set(original.keys()))
            for k, v in new_model.state_dict().items():
                torch.testing.assert_close(v, original[k])
        finally:
            os.remove(path)

    def test_extra_keys_stored(self):
        model, opt = self._make_model_and_opt()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, opt, epoch=5, hist_tr=[], hist_te=[],
                            path=path, base=32, model_type="mlp")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["base"], 32)
            self.assertEqual(ckpt["model_type"], "mlp")
            self.assertEqual(ckpt["epoch"], 5)
        finally:
            os.remove(path)

    def test_none_extra_keys_not_stored(self):
        """extra 中值为 None 的键不写入 checkpoint。"""
        model, opt = self._make_model_and_opt()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, opt, epoch=1, hist_tr=[], hist_te=[],
                            path=path, pooling=None, base=16)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("pooling", ckpt)
            self.assertIn("base", ckpt)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
