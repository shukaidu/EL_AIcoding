# 仓库指南

## 沟通与环境
- 默认使用中文沟通。
- 在仓库根目录直接执行命令，使用已激活环境中的 `python`。
- 安装依赖优先用 `conda install`，找不到再用 `pip install`。

## 项目结构
仓库包含三个问题：`burgers_1d`、`wave_2d_linear`、`wave_2d_nonlinear`。

- `config/`：全部可调参数的唯一来源。
- `pde/`：PDE 求解器；测试和可视化脚本在 `pde/test/`。
- `ml/`：训练、数据加载、checkpoint 与模型，模型定义在 `ml/models/`。
- `gen_data.py`：生成数据。
- `compare.py`：生成 solver 与 NN rollout 对比。
- `sweep.py`：`wave_2d_nonlinear` 超参搜索。
- `data/<problem>/`：数据、模型、训练曲线、对比图。
- `latex/report.tex`：报告源文件；`latex/figures/<problem>/`：报告配图。

## 开发与运行命令
- `python gen_data.py --problem wave_2d_nonlinear`：生成指定问题数据。
- `python -m ml.train --problem wave_2d_nonlinear`：训练模型。
- `python compare.py --problem wave_2d_nonlinear`：生成对比图。
- `PYTHONPATH=. python pde/test/test_burgers_1d.py`：运行单个测试。
- `cd latex && tectonic report.tex`：编译 PDF 报告。

标准流程：清理旧 `.png`，再执行数据生成、训练、对比、拷图、更新报告、编译 PDF。预计超过 30 秒的命令默认后台运行，不要用 `> file.log` 或 `2>&1` 重定向输出。

## 代码与配置约定
- Python 统一 4 空格缩进。
- 私有辅助函数使用前导下划线，如 `_wrap`。
- 所有实验参数必须从 `config/` 读取，禁止在脚本里写死。
- 新增函数参数不要补默认值；修改接口前先全局搜索并一次性更新全部调用点。

## 测试与报告要求
- 新测试文件命名为 `test_<feature>.py`。
- 提交结果前，确认 config、图像输出、报告表格三者一致。
- 更新 `latex/report.tex` 时重点核对：`TSCREEN`、`nst`、`nwd`、`njp`、`base`、`smooth_weight`、`smooth_mode`、`param_ratio`、`compare_TF`、`model_type`。
- 历史易错项：nonlinear 参数表、burgers 的 `njp/nwd`、Abstract 中模型名和正文不一致。

## Git 与提交
- 不主动执行 `git commit`，只有用户明确要求时才提交。
- 丢弃工作区改动使用 `git restore <files>`，不要用 `git checkout -- <files>`。
- 提交信息沿用现有风格：单行、动词开头，例如 `Refactor code and add unit tests`。
- PR 需说明影响范围、配置改动原因、关键指标或图像变化，以及报告是否已同步更新。
