# Claude Code 指令

## 沟通
- 始终用中文回复。

## 环境
- Python：直接用 `python`（conda base 环境已激活）
- 工作目录：仓库根目录，不用每次 `cd`，直接写 `python gen_data.py`、`git status` 等短命令
- 安装包优先用 `conda install`，conda 找不到时才用 `pip install`

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
- `sweep.py`：超参搜索（针对 `wave_2d_nonlinear`）
- `pde/test/`：各问题单元测试及可视化脚本 `vis_pde.py`
- `data/<问题名>/`：数据、模型、训练曲线、对比图输出
- `latex/report.tex`：实验报告 LaTeX 源文件
- `latex/figures/<问题名>/`：实验图片归档目录

## Pipeline
执行顺序（数据生成三个问题可并行，训练建议顺序执行避免 GPU 争抢）：

1. 清理旧图：删除 `data/` 和 `latex/figures/` 下所有 `.png`
2. 生成数据：`python gen_data.py --problem <问题名>`
3. 训练：`python -m ml.train --problem <问题名>`
4. 对比：`python compare.py --problem <问题名>`
5. 拷图：将 `data/<问题名>/training_history.png` 和 `data/<问题名>/compare/*.png` 拷到 `latex/figures/<问题名>/`
6. 更新报告：pipeline 跑完后**必须主动**更新 `latex/report.tex`，不等用户提醒。逐段全文核查：
   - 参数表所有数值与 config 一致（TSCREEN、nst、nwd、njp、base、smooth_weight、smooth_mode、param_ratio、compare_TF、model_type 等）
   - 表格行标签准确（CNN vs UNet，Jumps vs Frames 等）
   - Abstract/正文中模型名称、TV 描述与 config 一致
   - 历史易错项：nonlinear 表格参数、burgers njp/nwd 混淆、Abstract 写 CNN 实为 UNet
7. 编译 PDF：`cd latex && tectonic report.tex`

## Bash 规范
- 不要在命令末尾追加无意义确认输出
- 不要把程序输出重定向到日志文件（`> file.log`、`2>&1` 等）
- 预计超过 30 秒的命令（训练、数据生成、nonlinear compare 等）默认后台运行

## Config 规则
- 所有参数必须从 `config/` 读取，不允许在脚本里写死数值
- 如果某个参数 config 里没有，先加到 config，再在脚本里引用

## Git
- 不要主动 commit，只有用户明确要求时才执行 `git commit`
- commit 前检查 `.claude/settings.local.json` 是否有改动，有则一并纳入暂存区
- 丢弃工作区改动用 `git restore <files>`，不用旧语法 `git checkout -- <files>`

## 代码规范
- 模块内部辅助函数（不被其他模块导入）加前置下划线，如 `_wrap`
- 测试文件用 `PYTHONPATH=. python pde/test/test_burgers_1d.py` 运行
- 新增函数参数时不加默认值，强制调用方显式传参；修改接口前先用 Grep 找全所有调用点，一次性全部更新
