# 五组 InstanceNav 消融：手动测评说明

准备日期：2026-11-10。代码和启动入口已准备；**没有启动导航测评，也没有发送真实模型请求**。每次命令只运行指定的一组100任务，没有自动跑全套的入口。

## 启动命令

先按 [部署说明](../DEPLOYMENT_ZH.md) 从全量数据恢复并校验固定100任务，再激活本机测评环境：

```bash
cd /path/to/InstanceNav_AVS
conda activate psgnav-instance
export PSGNAV_PYTHON="$(command -v python)"
python tools/restore_instance100.py --verify-only
```

按需逐条运行，前一组完成后再用同两张卡运行下一组。--gpus可以替换为其他两张卡。

```bash
# E1：关闭实例核验，类别名探索保持不变
bash scripts/run_instance_ablation.sh E1 --gpus 0,1 --timestamp avs5_E1_r1

# E2：VLM只核类别，k=1，不永久排除
bash scripts/run_instance_ablation.sh E2 --gpus 0,1 --timestamp avs5_E2_r1

# E3：完整实例核验，明确no永久排除，unknown保留，k=1
bash scripts/run_instance_ablation.sh E3 --gpus 0,1 --timestamp avs5_E3_r1

# E4：完整实例核验，no或unknown均永久排除，k=1
bash scripts/run_instance_ablation.sh E4 --gpus 0,1 --timestamp avs5_E4_r1

# E5：完整实例核验，累计2次yes，不永久排除
bash scripts/run_instance_ablation.sh E5 --gpus 0,1 --timestamp avs5_E5_r1
```

同版本基准B可单独手动运行，建议用它作本批直接对照，不用旧36/37/36自动替代：

```bash
bash scripts/run_instance_ablation.sh B --gpus 0,1 --timestamp avs5_B_r1
```

## 中断、检查与重复

中断后保持同一个实验、时间戳、GPU与源码，增加--resume；已完成任务不重新跑：

```bash
bash scripts/run_instance_ablation.sh E3 --gpus 0,1 --timestamp avs5_E3_r1 --resume
```

检查配置/依赖而不测评，或只打印完整命令：

```bash
bash scripts/run_instance_ablation.sh E3 --check-only
bash scripts/run_instance_ablation.sh E3 --timestamp avs5_E3_r1 --dry-run
```

--check-only不导航、不调用模型；它检查数据、模型文件、依赖和有效配置，但未验证远端API实时可用性。需要第二轮时另用r2时间戳，不用--resume代表新重复。本轮没有预定所有组重复3轮。

## 已固定的实验条件

所有组固定类别名探索、qwen3.5-flash、match_strict_no、positive_match、准入关闭、RAG=0、相对证据关闭、多图复验关闭、恢复清空正票、1000原子动作及原InstanceNav距离协议。每组仅改变指定变量；E1按原实现退回类别计数门槛。

类别核验使用独立提示词，不发送实例属性或环境描述，仍输出yes/no/unknown。E3/E4继续使用原实例提示词，只改变输出的处理方式。永久排除使用现有node/空间关联，持续本episode，恢复不清除、下一episode清除；不使用GT身份。原始回答、永久排除及后续拦截记录均保留。

**计票口径：** 原代码允许同策略帧重复检测累计；为落实实验计划“同帧不能凑齐k”的定义，本批所有实例核验组统一设置INSTANCE_CONFIRMATION_COUNT_UNIT=frame，每候选每策略帧最多一票。不同帧仍不等于独立视角。普通run_instance100.sh默认仍为detection，保留旧行为。此差异显式记入启动头和manifest，因此建议补本批B，后续k分析以本批数据为准。

新入口还统一开启实际模型输入记录，保存带编号框图像、完整请求/输出及缓存等元数据；这增加磁盘写入，比较墙钟成本时须说明记录开销，逻辑调用/API调用分别统计。E1没有实例核验请求，因此不会产生相应请求图像。

## 输出位置与同版本约束

以E3第一轮为例：

- 合并结果：data/results_instancenav_avs5_E3_r1/experiment_0/results.txt。
- 两卡日志：outputs/instancenav_val10_100_avs5_E3_r1_gpu0.log及gpu1.log。
- 请求证据：日志同目录的同名前缀.verification/，其中events.jsonl关联images/中的实际输入图像。
- 源码和配置归档：outputs/instance_ablation_runs/avs5_20261110/avs5_E3_r1/manifest.json、source.tar.gz。
- 共同比较批次：outputs/instance_ablation_runs/avs5_20261110/suite.json。

首次真正启动时才创建运行归档。归档包含本项目运行源码、配置和数据哈希，不复制模型权重、数据集或.env凭据；外部依赖仍使用本机环境，并记录子模块版本。代码在当前工作区执行，归档不代表隔离执行环境。

本批测评期间不要修改运行源码或配置；同一suite后续启动会检查源码/数据指纹和解释器，不一致则拒绝混跑。续跑也检查实验配置及GPU等是否一致。确实要修改方法时，用新的--suite名称和新时间戳创建新的比较批次，并重跑相应基准，不能用它绕过不一致而混表。

## 验证记录

- 本机测评解释器：Python 3.9、PyTorch 2.7.0+cu118。
- 子集恢复与指纹校验、核验/排除、候选与场景图路径、k票、恢复、原提示词兼容、启动配置和归档续跑测试：230通过、2跳过（完整公开源数据相关检查，非当前100任务预检）。
- B与E1–E5的无API预检均通过，本机详细记录位于 outputs/instance_ablation_preparation_20261110/（不入Git）。
- 未进行真实导航烟测或远端API探测，正式测评由用户手动启动。

另在扩大检查范围时遇到一个原有relative-evidence测试替身缺少collect_only_enabled方法的失败，使用HEAD版本的Agent方法亦复现。本批明确关闭relative evidence；未为该无关测试修改运行逻辑。

计划来源：本次讨论确定的五组精简实验；这里的代码、配置和说明随 Git 同步，outputs 下的原始记录仅保留在本机。
