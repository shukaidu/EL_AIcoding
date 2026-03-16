# Claude Code 指令

## 目录
- 不用每次都 `cd`，默认工作目录就是仓库根目录。

## 沟通
- 始终用中文回复。

## 环境
- Python：直接用 `python`（conda base 环境已激活）
- 工作目录：仓库根目录（含 `gen_data.py`、`compare.py`、`config/`、`pde/`）

## 项目结构
三个 PDE 问题：`burgers_1d`、`wave_2d_linear`、`wave_2d_nonlinear`

- `config/`：所有可调参数
- `pde/`：PDE 求解器
- `ml/`：机器学习模块
  - `train.py`：训练入口（`python -m ml.train --problem <问题名>`）
  - `train_loop.py`：训练循环
  - `data_io.py`：数据加载
  - `snapshot.py`：模型快照保存/加载
  - `models/`：模型定义（`cnn.py`、`mlp.py`）
- `gen_data.py`：数据生成入口
- `compare.py`：对比入口
- `data/<问题名>/`：数据、模型、训练曲线、对比图输出
- `latex/figures/<问题名>/`：实验图片归档目录

## Pipeline 规则
执行 pipeline 时按以下顺序：

1. 清理旧图：删除 `data/` 和 `latex/figures/` 下所有 `.png`
2. 生成数据：`python gen_data.py --problem <问题名>`（三个问题可并行）
3. 训练：`python -m ml.train --problem <问题名>`
4. 对比：`python compare.py --problem <问题名>`
5. 拷图：将 `data/<问题名>/training_history.png` 和 `data/<问题名>/compare/*.png` 拷到 `latex/figures/<问题名>/`
6. 更新报告：将 `latex/report.tex` 中对应问题的参数表和结果数值更新到最新值

## 并行与顺序
- 数据生成三个问题可并行执行
- 训练建议顺序执行，避免 GPU 争抢

## Bash 命令规范
- 不要在 `cp`、`ls` 等命令末尾追加无意义确认输出

## Git
- 不要主动 commit，只有用户明确要求时才执行 `git commit`
