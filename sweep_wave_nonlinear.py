"""Parameter sweep: smooth_weight x base for wave_2d_nonlinear.

Usage:
    python sweep_wave_nonlinear.py
"""
import sys, io, os, re, shutil
import torch
import config.wave_2d_nonlinear_config as cfg
from ml.data_io import load_wave_2d_nonlinear
from ml.models import CNN
from ml.train import _run
from compare import _compare_wave_2d_nonlinear
from ml.train_loop import get_device

PARAM_GRID = [
    {"smooth_weight": 0.0,  "base": 32},
    {"smooth_weight": 0.01, "base": 32},
    {"smooth_weight": 0.05, "base": 32},
    {"smooth_weight": 0.1,  "base": 32},
    {"smooth_weight": 0.0,  "base": 48},
    {"smooth_weight": 0.05, "base": 48},
]

data_dir = "data/wave_2d_nonlinear"
sweep_dir = os.path.join(data_dir, "sweep")
os.makedirs(sweep_dir, exist_ok=True)

device = get_device()
print(f"Device: {device}")

# 数据只加载一次，所有 combo 复用
data_path = os.path.join(data_dir, cfg.data_mat)
if not os.path.isfile(data_path):
    print(f"Data not found: {data_path}. Run: python gen_data.py --problem wave_2d_nonlinear")
    sys.exit(1)

tl, vl, _, C_in, C_out, Nx, Ny, nx, ny = load_wave_2d_nonlinear(
    data_path, device, b_size=cfg.b_size
)

results = []
for params in PARAM_GRID:
    sw, base = params["smooth_weight"], params["base"]
    tag = f"sw{sw}_b{base}"
    model_rel = f"sweep/model_{tag}.pth"
    cmp_dir = os.path.join(sweep_dir, f"compare_{tag}")
    os.makedirs(cmp_dir, exist_ok=True)

    # 临时覆盖 config 属性（模块单例，compare 会自动读取最新值）
    cfg.smooth_weight = sw
    cfg.base = base
    cfg.model_pth = model_rel

    print(f"\n=== smooth_weight={sw}, base={base} ===")
    model = CNN(Cin=C_in, Cout=C_out, base=base, Nx=Nx, nx=nx).to(device)
    _run(model, tl, vl, cfg, data_dir, base=base)

    # Compare，重定向 stdout 捕获 L1
    buf = io.StringIO()
    sys.stdout, old = buf, sys.stdout
    _compare_wave_2d_nonlinear(data_dir, cmp_dir)
    sys.stdout = old
    output = buf.getvalue()
    print(output)

    m = re.search(r"L1 error mean: ([0-9.]+)", output)
    l1 = float(m.group(1)) if m else float("inf")
    results.append({"smooth_weight": sw, "base": base, "l1_mean": l1, "tag": tag})
    print(f"  -> L1 mean = {l1:.6f}")

# 汇总排序
print("\n=== 参数搜索汇总（按 L1 升序）===")
for r in sorted(results, key=lambda x: x["l1_mean"]):
    print(f"  sw={r['smooth_weight']:5.3f}, base={r['base']} -> L1={r['l1_mean']:.6f}")

best = min(results, key=lambda x: x["l1_mean"])
print(f"\n最优: smooth_weight={best['smooth_weight']}, base={best['base']} (L1={best['l1_mean']:.6f})")

# 将最优模型覆盖到标准路径
shutil.copy(
    os.path.join(data_dir, f"sweep/model_{best['tag']}.pth"),
    os.path.join(data_dir, "data_wave_model.pth"),
)
print("最优模型已复制到 data_wave_model.pth")
print(f"请将 config 更新为: smooth_weight={best['smooth_weight']}, base={best['base']}")
