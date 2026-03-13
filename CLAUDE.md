# Claude Code 指令

## 目录
- 不用每次都cd！已经在工作目录！

## 沟通
- 始终用**中文**回复。

## 环境
- Python：直接用 `python`（conda base 环境已激活）
- 工作目录：仓库根目录（含 `gen_data.py`、`compare.py`、`config/`、`pde/`）

## 项目结构
三个 PDE 问题：`burgers_1d`、`wave_2d_linear`、`wave_2d_nonlinear`

- `config/` — 所有可调参数（改实验设置只改这里）
- `pde/` — PDE 求解器
- `ml/` — 机器学习模块
  - `train.py` — 训练入口（`python -m ml.train --problem <问题名>`）
  - `train_loop.py` — 训练循环
  - `data_io.py` — 数据加载
  - `snapshot.py` — 模型快照保存/加载
  - `models/` — 模型定义（`cnn.py`、`mlp.py`）
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

## 并行与顺序
- 数据生成三个问题可同时并行执行；训练建议顺序执行（避免 GPU 争抢）

## Bash 命令规范
- 不要在 `cp`、`ls` 等命令末尾加 `&& echo "Done"` 之类的确认输出——命令成功即可，无需额外打印。这样可以避免链式命令因末尾非 `cp` 而绕过 allowlist 匹配。

## Git
- **不要主动 commit**：只有用户明确要求时才执行 `git commit`
