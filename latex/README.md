# LaTeX 报告说明

## 编译

在 `latex` 目录下执行：

```bash
pdflatex report.tex
```

建议运行两次以正确生成目录和交叉引用：

```bash
pdflatex report.tex
pdflatex report.tex
```

## 内容

- **report.tex**：主文档，包含实验背景、参数表、训练曲线与对比图。
- **figures/**：三个子问题的图片（来自 `data/` 的 compare 与 training_history）。

## 依赖

标准宏包：`graphicx`, `booktabs`, `subcaption`, `float`, `hyperref`。若缺少 `subcaption`，可改用 `subfig` 或删除子图，仅保留单图。
