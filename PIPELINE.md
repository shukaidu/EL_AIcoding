# Element Learning 实验流程说明

本文档说明从数据生成到 LaTeX 报告的全流程，以及每一步的含义、命令和输出。

---

## 1. 流程概览

整体流程分为五步，**必须按顺序执行**（后一步依赖前一步的输出）：

| 步骤 | 内容 | 主要命令 / 说明 |
|------|------|------------------|
| 0 | **删除所有 png** | 清理 `data/` 与 `latex/figures/` 下的旧图，避免污染本次结果 |
| 1 | **生成数据** | `python gen_data.py --problem <问题名>` |
| 2 | **训练网络** | `python -m ml.train --problem <问题名>` |
| 3 | **运行对比** | `python compare.py --problem <问题名>` |
| 4 | **拷贝图片、更新 report、编译** | 拷贝图到 `latex/figures/`，**先更新 `report.tex`**（见 8.2），再编译 |

支持的问题名（`<问题名>`）有三个：

- `burgers_1d`：一维 Burgers 方程（有限体积参考解 vs MLP）
- `wave_2d_linear`：二维线性波动方程（谱方法 vs MLP）
- `wave_2d_nonlinear`：二维非线性浅水（谱方法 vs ShrinkCNN）

按下面各节依次执行命令即可完成全流程。

**提醒**：执行任何 pipeline 相关指令（含「从 compare 开始跑」「只跑某问题」等）时，请对照本文档。**步骤 4 务必包含「更新 report.tex」**（参数表、结果耗时/误差、图文件名与数量），再编译，否则报告与最新实验不一致。

---

## 2. 环境与依赖

- **Python**：建议 3.8+，需安装：`numpy`、`scipy`、`torch`、`matplotlib`。
- **本机 Python 路径（Anaconda）**：若终端中 `python` 不可用，请使用完整路径或先激活 conda 环境。当前环境使用的可执行文件为：
  ```text
  C:\Users\dushu\anaconda3\python.exe
  ```
  下文所有 `python` 命令可替换为上述路径，例如：`C:\Users\dushu\anaconda3\python.exe gen_data.py --problem burgers_1d`。
- **工作目录**：所有命令均在**仓库根目录**（即包含 `gen_data.py`、`compare.py`、`config/`、`pde/` 的目录）下执行。
- **配置**：所有可调参数均在 `config/` 下对应文件中，无需改脚本内部常量（详见第 3 节）。

---

## 3. 配置文件说明

所有设置集中在 `config/` 下，按问题分文件：

| 文件 | 对应问题 | 主要内容 |
|------|----------|----------|
| `config/burgers_1d_config.py` | 1D Burgers | PDE/网格（L, nx, dt, njp, nst, nwd, nu, alpha 等）、数据量（nsamp, n_trajectories）、训练（b_size, num_epochs, hidden_size, lr_schedule）、compare（compare_seed, compare_t_end, compare_n_times）、文件名 |
| `config/wave_2d_linear_config.py` | 2D 线性波 | 波速 c、网格与时间（NX, NY, Lx, Ly, dt, TF, TSCREEN）、预测窗口（nwd, njp）、数据（nsamp, ntest, ic_list）、训练、compare（compare_TF, compare_ic, compare_seed）、文件名 |
| `config/wave_2d_nonlinear_config.py` | 2D 非线性浅水 | 网格与 PDE（Lx, Ly, nx, ny, g, h0, TSCREEN）、数据与训练、compare、文件名 |

修改实验设置（如 epoch 数、学习率、对比用的初值或种子）时，**只需改对应 config 文件**，无需改 `gen_data.py`、`compare.py` 或 `ml/train.py`。

---

## 4. 步骤 0：删除旧图（避免污染本次结果）

在开始生成数据、训练、对比之前，先删除此前产生的所有 png，保证本次流程得到的图即为最新结果。

### 4.1 要删除的位置

- **data/** 下所有 `.png`：包括各问题下的 `training_history.png` 和 `compare/*.png`。
- **latex/figures/** 下所有 `.png`：报告引用的图，避免编译报告时仍显示旧图。

### 4.2 命令示例

在仓库根目录执行（二选一或都执行）：

**Windows（PowerShell）：**

```powershell
Get-ChildItem -Path data -Recurse -Filter *.png | Remove-Item -Force
Get-ChildItem -Path latex\figures -Recurse -Filter *.png | Remove-Item -Force
```

**Linux / macOS：**

```bash
find data -name "*.png" -delete
find latex/figures -name "*.png" -delete
```

执行完毕后，再按下面步骤 1～4 依次进行数据生成、训练、对比和拷贝、编译报告。

---

## 5. 步骤 1：生成数据

### 5.1 命令

```bash
python gen_data.py --problem burgers_1d
python gen_data.py --problem wave_2d_linear
python gen_data.py --problem wave_2d_nonlinear
```

每次只针对一个 `--problem`；要跑三个问题就执行三次。

### 5.2 作用

- 用**参考求解器**（FV 或谱方法）在 config 给定的网格、时间、初值下生成时间演化。
- 从中按 patch 采样，得到「输入 patch → 输出 patch」的样本对，数量由 config 中的 `nsamp`（及 2D 的 `ntest`）等决定。
- 数据以 `.mat` 形式写入 `data/<问题名>/`，文件名由 config 中的 `data_mat` 指定（如 `data_res.mat`、`data_wave.mat`）。

### 5.3 输出位置与用途

- **输出路径**：`data/<问题名>/<data_mat>`，例如 `data/burgers_1d/data_res.mat`。
- **用途**：步骤 2 的训练会读取该文件；若不存在，训练会报错并提示先运行 `gen_data.py`。
- **并行**：三个问题的数据生成均支持多进程（Burgers 按轨迹、2D 按 ntest 条仿真并行），进程数由 config 与 CPU 核心数自动决定。

---

## 6. 步骤 2：训练网络

### 6.1 命令

```bash
python -m ml.train --problem burgers_1d
python -m ml.train --problem wave_2d_linear
python -m ml.train --problem wave_2d_nonlinear
```

同样每次一个 `--problem`。

### 6.2 作用

- 从 `data/<问题名>/<data_mat>` 读入步骤 1 生成的数据，按 config 中的 `b_size`、`test_split` 等划分 train/test。
- 用 config 中的网络结构（如 Burgers/2D linear 的 MLP：`hidden_size`, `num_hidden_layers` / `num_layers`；2D nonlinear 的 ShrinkCNN：`base`）和优化设置（`num_epochs`, `lr_schedule`）进行训练。
- 训练过程中会保存 checkpoint 和训练曲线图。

### 6.3 输出位置与用途

- **模型**：`data/<问题名>/<model_pth>`，如 `data/burgers_1d/data_res_model.pth`。
- **训练曲线**：`data/<问题名>/training_history.png`。
- **用途**：步骤 3 的 compare 会加载 `<model_pth>`；若不存在，compare 会报错并提示先运行 `python -m ml.train --problem <问题名>`。

---

## 7. 步骤 3：运行对比（compare）

### 7.1 命令

```bash
python compare.py --problem burgers_1d
python compare.py --problem wave_2d_linear
python compare.py --problem wave_2d_nonlinear
```

### 7.2 作用

- **参考解**：用与数据生成相同的参考求解器，在 config 中 **compare** 相关参数下跑一条（或若干）轨迹（如 `compare_ic`, `compare_seed`, `compare_t_end` / `compare_TF`）。
- **NN 推演**：加载 `data/<问题名>/<model_pth>`，从同一条初值出发，用学到的 patch 映射做时间步进，得到 NN 的数值解。
- 在若干时刻对比参考解与 NN 解，并记录参考解与 NN 的耗时，用于算加速比。

### 7.3 输出位置与用途

- **对比图**：`data/<问题名>/compare/t0.png`, `t1.png`, … ，每个时刻一张图（Spectral/FV vs NN）。
- **用途**：步骤 4 会把这些图与 `training_history.png` 一起拷贝到 `latex/figures/<问题名>/`，供 LaTeX 报告引用。

---

## 8. 步骤 4：拷贝图片、更新 report、编译报告

**务必记住**：拷贝完图片后，**必须先根据 config 与最近一次 compare 输出更新 `latex/report.tex`**（见 8.2），再执行 `pdflatex`，否则报告中的参数、耗时、图引用会与当前实验不符。

### 8.1 拷贝图片

需要保证 LaTeX 能引用到的图片来自 `latex/figures/`，且与 `latex/report.tex` 中的路径一致。约定：

- 对每个问题名 `P`（如 `burgers_1d`, `wave_2d_linear`, `wave_2d_nonlinear`）：
  - 将 `data/P/training_history.png` 拷贝为 `latex/figures/P/training_history.png`；
  - 将 `data/P/compare/*.png` 全部拷贝到 `latex/figures/P/`（保持文件名 t0.png, t1.png, …）。

这样 `report.tex` 里使用 `figures/burgers_1d/...`、`figures/wave_2d_linear/...`、`figures/wave_2d_nonlinear/...` 即可。拷贝可手动完成，或在 Windows 上用 `xcopy data\burgers_1d\training_history.png latex\figures\burgers_1d\` 等命令，在 Linux/macOS 上用 `cp` 类推。

### 8.2 更新 report.tex 后编译（不可省略）

**生成 report 前必须做**：根据当前 **config** 与最近一次 **compare 输出** 修改 `latex/report.tex`，避免报告与实验不一致。

- **参数表**（report 第 2 节）：与 `config/*.py` 一致（PDE、网格、训练 epochs/hidden size/njp/nu、compare_ic/compare_t_end 等）。
- **结果中的数值**：与最近一次 `compare.py` 打印一致（FV/Spectral/NN 耗时、L1 mean、加速比）。
- **图引用**：compare 在 njp>1 时只生成 t0, t1, …, t5 等，report 中 `\includegraphics` 的文件名与数量须与 `latex/figures/<问题>/` 下实际一致（如 2D linear 为 t0/t3/t5，无 t10）。

然后编译报告：

```bash
cd latex
pdflatex report.tex
pdflatex report.tex
```

建议运行两次 `pdflatex` 以正确生成交叉引用。生成的 PDF 为 `latex/report.pdf`。

报告内容包含：实验背景、三组问题的参数表、训练曲线（三张 `training_history.png`）、以及各问题的参考解 vs NN 对比图（由 `report.tex` 中引用的 t0/t5/t9 等示例图组成）。可根据需要增删或替换 `report.tex` 中的图。

---

## 9. 目录结构（与流程相关部分）

```
EL_cursor/
├── config/
│   ├── burgers_1d_config.py      # 1D Burgers 全部配置
│   ├── wave_2d_linear_config.py  # 2D 线性波全部配置
│   └── wave_2d_nonlinear_config.py
├── data/
│   ├── burgers_1d/
│   │   ├── data_res.mat          # 步骤 1 生成
│   │   ├── data_res_model.pth    # 步骤 2 生成
│   │   ├── training_history.png  # 步骤 2 生成
│   │   └── compare/              # 步骤 3 生成
│   │       ├── t0.png, t1.png, ...
│   ├── wave_2d_linear/
│   │   ├── data_wave.mat
│   │   ├── data_wave_model.pth
│   │   ├── training_history.png
│   │   └── compare/
│   └── wave_2d_nonlinear/
│       └── ...
├── latex/
│   ├── figures/                  # 步骤 4 拷贝目标
│   │   ├── burgers_1d/
│   │   ├── wave_2d_linear/
│   │   └── wave_2d_nonlinear/
│   ├── report.tex
│   └── README.md
├── gen_data.py                   # 步骤 1 入口
├── compare.py                    # 步骤 3 入口
├── PIPELINE.md                   # 本文档
└── ml/
    └── train.py                  # 步骤 2 入口（python -m ml.train）
```

---

## 10. 常见问题与注意事项

1. **先数据、后训练、再对比**  
   若跳过数据生成，请确保对应 `data/<问题>/<data_mat>` 已存在且与当前 config 一致（如 nx、njp、nu 等），否则训练/对比结果可能不一致。

2. **改 config 后**  
   若修改了 PDE 参数、网格或 compare 设置，通常需要至少重新生成数据（若与数据相关）或重新训练/对比；改 compare 相关（如 `compare_ic`）只需重新运行 compare 并重新拷贝到 latex 即可。

3. **报告中的图**  
   `report.tex` 中引用的图名（如 t0, t5, t9）需与 `data/<问题>/compare/` 下实际生成的文件名一致；若 compare 输出帧数或命名方式变化，需同步修改 `report.tex` 中的 `\includegraphics` 路径或文件名。

4. **GPU**  
   训练与 compare 中的 NN 推理会自动使用 CUDA（若可用）；无 GPU 时使用 CPU，速度较慢但流程相同。

5. **随机种子**  
   数据生成、对比初值等均由 config 中的 `seed_base`、`compare_seed`、`rng_seeds` 等控制，固定这些即可复现结果。

6. **别忘了更新 report**  
   每次执行 pipeline（含「只跑 compare」「只跑某问题」）后，步骤 4 都要**先更新 `report.tex`**（参数、耗时、图引用），再编译。执行前建议先看一遍本文档，避免漏掉更新 report。

---

以上即从「生成数据 → 训练 → 对比 → 拷贝到 latex → **更新 report** → 编译」的完整流程与说明。所有可调设置均在 `config/` 中，按本文档依次执行各步命令即可完成实验与报告。
