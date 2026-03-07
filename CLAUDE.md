# Claude Code 指令

## 沟通
- 始终用**中文**回复。

## 环境
- Python 路径：`/c/Users/dushu/anaconda3/python.exe`（bash 中使用正斜杠格式）
- 工作目录：仓库根目录（含 `gen_data.py`、`compare.py`、`config/`、`pde/`）

## 项目结构
三个 PDE 问题：`burgers_1d`、`wave_2d_linear`、`wave_2d_nonlinear`

- `config/` — 所有可调参数（改实验设置只改这里）
- `pde/` — PDE 求解器
- `common/` — 公共工具（数据加载、模型、训练循环）
- `ml/train.py` — 训练入口
- `gen_data.py` — 数据生成入口
- `compare.py` — 对比入口
- `data/<问题名>/` — 数据、模型、对比图输出
- `latex/figures/<问题名>/` — 报告引用的图

## Pipeline 规则
执行 pipeline 时**必须按以下顺序**：

1. **清理旧图**：删除 `data/` 和 `latex/figures/` 下所有 `.png`
2. **生成数据**：`python gen_data.py --problem <问题名>`（三个问题可并行）
3. **训练**：`python -m ml.train --problem <问题名>`
4. **对比**：`python compare.py --problem <问题名>`
5. **拷图**：将 `data/<问题名>/training_history.png` 和 `data/<问题名>/compare/*.png` 拷到 `latex/figures/<问题名>/`
6. **更新 `latex/report.tex`**：参数表、耗时/加速比数值、图引用数量必须与本次实验一致

**注意**：步骤 6 不可省略，每次跑完 compare 后都要更新 report.tex。**不需要编译 PDF。**

## 后台任务
- 用 `/c/Users/dushu/anaconda3/python.exe` 运行 Python（不要用反斜杠路径）
- 数据生成三个问题可并行后台运行；训练建议顺序执行（避免 GPU 争抢）
- 用 `TaskOutput` 等待后台任务，超时设为 600000ms
