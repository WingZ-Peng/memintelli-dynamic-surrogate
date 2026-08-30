[English](README.md) · **简体中文**

# 理想电导下的动态三头代理模型

针对**一个固定权重和一个固定的理想电导张量**，用三个条件预测头刻画忆阻器交叉阵列矩阵-向量乘法在动态读取噪声与电压噪声下的完整输出分布——不只是均值，还包括方差，以及各输出坐标之间的相关结构。

所学习的对象是 `p(Y | X, G_ideal, hardware_config)`。写入变异与固定故障已被关闭：它们既不是模型输入，也不属于目标分布，因此这**不是**器件总体（device population）模型。

> 本文由英文版 [`README.md`](README.md) 翻译而来。**所有数值以英文版为准。** 表格中的指标名称（如 `conditional_three_head`）与 `outputs/three_head/report/three_head_results.json` 中的键名一一对应，因此保留英文原样，便于直接检索。

## 结果

在 64 个未参与训练的测试 context 上，代理模型复现仿真器输出协方差的精度，与仿真器自身独立重跑一次的精度相当：

| 方法 | 线性输出方差 | 线性输出协方差 |
|---|---:|---:|
| `exact_independent_split`（参考基准） | 1.000 | 1.000 |
| **`conditional_three_head`** | **1.038** | **1.003** |
| `analytic_correlation_three_head` | 1.035 | 1.014 |
| `shared_three_head`（不使用条件信息） | 6.941 | 2.502 |
| `shuffled_input_features`（破坏条件对应关系） | 9.806 | 3.508 |

以上是相对 **Exact-vs-Exact 有限样本基准**的比值：每个测试 context 的 2,048 个 Exact 样本被划分为 1,024 个候选样本与 1,024 个参考样本，基准值即为这两半之间的距离。比值为 1.000 表示代理模型与参考样本的差距，和"再跑一次同样规模的仿真"所产生的差距处于同一水平。

各预测头在测试 context 上的表现：

| 预测头 | Exact 基准 | 条件模型 | 共享基线 | 基准比值 |
|---|---:|---:|---:|---:|
| 均值 NRMSE | 0.0442 | **0.0335** | 0.1164 | 0.759 |
| 方差 L1 | 0.0494 | **0.0398** | 0.4597 | 0.805 |
| 相关性 Frobenius（支撑集） | 0.1963 | **0.1429** | 0.9895 | 0.728 |

需要强调：`exact_independent_split` 衡量的是**两次含噪声采样之间**的距离，因此它并不是一个下界——一个好的确定性预测器完全可以低于它。比值小于 1.0 并非不可能，比值大于 1.0 也不能单独作为"预测头不好"的证据。

## 环境要求

- Python 3.11
- 与显卡匹配的 CUDA 版 PyTorch（CPU 也能跑，但很慢）
- `numpy < 2` 与 `matplotlib`，两者由内置的仿真器引入
- 约 300 MB 空闲磁盘空间，用于重新生成的样本分片

已发布的结果在 Python 3.11.15、torch 2.11.0+cu128、单块 NVIDIA RTX 5060（计算能力 12.0）上产生。

## 1. 配置环境

```powershell
conda create -n surrogate python=3.11 -y
conda activate surrogate

# 先安装与你的 CUDA 版本匹配的 torch。下面是复现已发布结果所用的 cu128 版本；
# 请按需从 https://pytorch.org 选择。
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

在 Linux 或 macOS 上这三条命令同样适用，只是激活环境的语法略有差异。

## 2. 验证安装

```powershell
python -m unittest discover -s tests -v
```

预期结果是 **6 个测试、跳过 2 个**，耗时几秒。刚克隆下来时这两个跳过是正常的：它们需要读取采集好的样本分片，而分片不纳入 git 管理（见第 4 步）。实际执行的四个测试中包含一条完整的微型端到端流程——采集、训练、评估，然后断点续跑——全部在临时目录中完成，因此通过即说明整套环境可用。

## 3. 不重新计算，直接查看已发布结果

训练好的检查点与生成的报告随仓库一同发布，因此结果可以立即查阅：

- [`docs/SUMMARY.md`](docs/SUMMARY.md) —— 汇总文档：架构、方法、全部表格，以及完整的运行日志
- [`outputs/three_head/report/three_head_results.md`](outputs/three_head/report/three_head_results.md) —— 生成的报告
- `outputs/three_head/report/three_head_results.json` —— 同样的数值，机器可读
- `outputs/three_head/checkpoints/` —— 三个训练好的预测头

确认检查点与其训练所依据的协议一致：

```powershell
python .\scripts\write_summary.py
```

该命令会依据现有产物重新生成 `docs/SUMMARY.md`。任何指纹不一致都会直接报错。

## 4. 从零复现

```powershell
python .\scripts\run_three_head.py all --device cuda:0
```

在 RTX 5060 上预留约 **30 分钟**。其中采集阶段占绝大部分，约 28 分钟、共 24 个分片；三个预测头的训练与最终评估加起来不到 1 分钟。

三个阶段相互独立且都支持断点续跑，可以分开执行、随时中断：

```powershell
python .\scripts\run_three_head.py collect --device cuda:0   # 约 28 分钟，写入 254 MB
python .\scripts\run_three_head.py train   --device cuda:0   # 数秒
python .\scripts\run_three_head.py analyze --device cuda:0   # 数秒
```

`--device` 也接受 `auto` 与 `cpu`。

重跑已完成的阶段不会重新计算：`collect` 会校验每个已有分片并报告 `reused`；`train` 会直接跳转到已保存的最优模型并报告 `resumed_complete`。续跑是**逐位一致**而非近似的——优化器状态与随机数生成器状态都与权重一同保存。

**第 4 步与第 3 步的关系。** 执行 `analyze` 会覆盖 `outputs/three_head/report/`。在协议未改动的前提下，重新生成的报告与随仓库发布的版本逐字节一致；但如果你手工编辑过这些文件，请先备份。

## 实现方式

`configs/three_head_protocol.json` 是所有数量、种子、维度与超参数的唯一真实来源。它的原始字节被哈希成一个 `protocol_fingerprint`，并写入每一个分片、检查点与报告。任何指纹不一致的产物都会直接报错——没有宽松回退，也没有"过期就重算"的路径。**修改协议会使 `outputs/` 下的一切失效。**

特征是输入、`G_ideal`、tile 布局与硬件配置的确定性函数，绝不依赖于某次实现出的 `G_static`、写入种子或随机采样轨迹。流程会解析地计算 `V_read = v·(1+N(0,σ_v))` 与 `G_read = g·exp(N(0,σ_r))` 在 ADC 之前的前两阶矩，将其换算到 ADC 码空间，并估计局部的 ADC 跳变增益。

### 相关性预测头是核心

一个 `S_tile` 坐标由 `(k_block, row_index, out_index)` 确定。它读取的是自身 `k_block` 内、自身行上的电压，以及自身 `k_block` 内、自身输出列上的电导单元。因此，除非两个坐标**同时**位于同一个 `k_block` 且共享一行或一个输出列，否则它们是**互不相交**的随机变量的函数——对其余所有坐标对，协方差严格为零，这来自独立性本身，而不是线性化近似。

9,900 个非对角元素中只有 1,300 个可能非零。所以这个预测头不是稠密的低秩矩阵，而是建立在物理噪声源之上的因子模型，每个坐标写成单位方差的求和形式：

```text
S_d = <alpha_d, eps_voltage[k, row]> + <beta_d, eps_conductance[k, out]>
      + sqrt(1 - |alpha_d|^2 - |beta_d|^2) * eta_d
```

不共享噪声源的坐标严格不相关、矩阵对任意参数取值都半正定、对角线严格为 1、采样是原生的且无需 Cholesky 分解——这些性质全部由构造保证，而非靠拟合得到。解析形式的载荷已存放在特征中，因此训练是**从解析解出发**的，并对偏离解析解施加惩罚。

报告中的 `analytic_first_order` 一行就是这个零参数、无需训练的闭式解。它是用来区分"结果中有多少来自物理、有多少来自学习"的消融项。

## 仓库结构

```text
three_head/                        流程主体（自包含）
  core.py                          采集、协议校验、原子写入
  features.py                      ADC 前矩的解析特征
  structure.py                     坐标共享关系 + 解析协方差
  structured_correlation.py        噪声源因子模型
  training.py                      三个预测头，含学习率调度
  evaluation.py                    指标、消融、报告
  metrics.py                       共用的指标定义
  conditional_mean.py   \
  observable_dpe.py      >          内置的模型与 tail 代码
  tail.py               /
configs/three_head_protocol.json   唯一真实来源
scripts/run_three_head.py          入口脚本
scripts/write_summary.py           重新生成 docs/SUMMARY.md
tests/test_three_head.py           自包含性 + 流程 + 结构检查
outputs/three_head/                流程写出的全部内容
memintelli_surrogate_comparison/   内置的上游仿真器，以及协议按哈希锁定的三个文件
docs/                              文档
```

最后那个目录名并非随意保留：`configs/three_head_protocol.json` 中记录了
`memintelli_surrogate_comparison/artifacts/` 下三个文件的 SHA-256，并在启动时按这个确切的相对路径去解析它们。重命名该目录会改变协议，进而使随仓库发布的检查点失效，因此保持原样。

### 哪些文件随仓库发布，哪些需要重新生成

纳入版本管理的有：全部源码、协议文件、三个训练好的检查点（5 MB）、生成的报告、采集与训练清单，以及运行日志。不纳入的是 `outputs/three_head/context_shards/`，即约 254 MB 的 Exact 采样数据——`collect` 阶段可依据协议种子确定性地重新生成。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/SUMMARY.md`](docs/SUMMARY.md) | 汇总文档，由产物自动生成。请勿手工编辑——改用 `scripts/write_summary.py` 重新生成。 |
| [`docs/DESIGN.zh-CN.md`](docs/DESIGN.zh-CN.md) | 范围约定：什么是固定的、什么是变化的、什么被排除在外。 |
| [`docs/KEY_FINDINGS.md`](docs/KEY_FINDINGS.md) | 详细结论，包含负面结果。 |
| [`docs/PROGRESS_REPORT.md`](docs/PROGRESS_REPORT.md) | 完整的研发过程，以及每个设计决策背后的推理。 |

## 适用范围与局限

这是一个小规模验证性实验。它表明在严格受控的条件下，这个条件分布是可学习且校准良好的——仅此而已：

- 只有一个固定的数学权重和一个固定的理想电导张量。
- 写入变异与固定故障被有意排除。
- 只有一种固定的形状、tile 布局与硬件配置。
- 输入是合成的高斯数据，而非真实网络的激活值。
- 各预测头的调度策略与相关性正则项是在验证集上选定的；测试集只在 `analyze` 阶段读取一次。
- **未声称任何加速比。** 尚未与仿真器做过任何实际耗时对比。

## 致谢与来源

`memintelli_surrogate_comparison/upstream/` 下的器件仿真器是 **MemIntelli** 的未经修改的第三方快照，来自 <https://github.com/HUST-ISMD-Odyssey/Memintelli>，由华中科技大学信息存储材料与器件研究所缪向水教授与李祎教授课题组开发。如果你使用了它，请引用他们的论文：

> H. Zhou, L. Yang, et al. *MemIntelli: A Generic End-to-End Simulation
> Framework for Memristive Intelligent Systems.* arXiv:2511.17418.

本流程的导入路径上只用到 `memintelli.pimpy`；软件包的其余部分原样保留，以确保其导入行为与上游发布时完全一致。

**关于其许可。** 该快照附带了位于 `memintelli_surrogate_comparison/upstream/license.txt` 的 MIT 许可证，但上游 README 同时声明该模型"is made publicly available on a non-commercial basis"（以非商业方式公开提供）。这两处说明彼此并不一致，本仓库不试图代为裁定。本仓库不授予你对上游代码的任何超出其作者真实意图的权利；若你的用途取决于这个问题的答案，请先与他们确认。

`three_head/`、`configs/`、`scripts/`、`tests/` 与 `docs/` 下的全部内容为原创工作。
