# InstanceNav_AVS 部署说明

此仓库保存当前源码快照，旧 Git 历史不随仓库发布。GLIP、Habitat-Lab、GroundingDINO 和 Segment Anything 的必要源码及许可证随项目保留。

## 本地资源

数据集、模型权重、API 凭据和运行结果不包含在 Git 中，请按 README 准备：

- InstanceNav 数据：`data/datasets/instancenav/val10/val10.json.gz`。
- HM3D 场景：`data/scene_datasets/hm3d_v0.2/`，可链接到本机已有数据目录。
- GLIP 权重：`GLIP/MODEL/glip_large_model.pth`；SAM 和 GroundingDINO 权重：`data/models/`。
- BERT 缓存：`data/huggingface/`；NLTK 资源：`data/nltk_data/`。
- 将 `.env.example` 复制为 `.env`，仅在本机填写 API Key。
- InstanceNav 配置、固定执行顺序和子集清单位于 `configs/runtime/`。清单只保存任务标识与复现信息，不包含图像、场景或完整数据集。
- 运行输出写入 `outputs/` 和 `data/results_*/`，均不受 Git 管理。

## 从全量数据恢复固定100任务

100个任务的身份清单 `configs/runtime/val10.manifest.json` 和执行索引
`configs/runtime/evaluation_order_fixed100.json` **已经受 Git 管理**。后者是 Habitat
按场景分组后的执行顺序，不是子集 JSON 的原始行号；不要自行取全量数据的前100条。

在新机器 `git pull` 后，使用 Python 3.9 或以上运行（只用标准库，不调用模型）：

```bash
cd /path/to/InstanceNav_AVS
python tools/restore_instance100.py
```

默认读取已下载的 `data/datasets/instancenav/val/val_text.json.gz`（PSL文本属性）和
`data/datasets/instance_imagenav_hm3d_v3/val/content/`（InstanceImageNav任务分片）。
如果下载在其他目录，指定原文件位置即可：

```bash
python tools/restore_instance100.py \
  --attributes /path/to/instancenav/val/val_text.json.gz \
  --content /path/to/instance_imagenav_hm3d_v3/val/content
```

输出到启动器使用的 `data/datasets/instancenav/val10/val10.json.gz`，并安装配套manifest。
工具沿用原采样程序的 seed=77、fraction=0.1，在临时文件中重建；只有任务身份、
执行顺序和完整文件 SHA256 都与 Git 中冻结清单一致，才替换本地子集。全量数据版本
不一致会报错，不会悄悄换成另一批100任务。只检查已有子集可用：

```bash
python tools/restore_instance100.py --verify-only
```

HM3D场景、模型权重和API配置仍按上面的本地资源要求准备。
五组消融的可移植启动命令和比较口径见 [100任务消融说明](reproduction/instance_ablation_100.md)。

## 当前实验设置

- InstanceNav 固定 100 条，与源机相同；31 个 HM3D val 场景。
- 每条最多 1000 个原子动作；停止距离和成功距离均为 1 米。
- `rag_mode=0`；Qwen3.5-Flash；客户端关闭 thinking。
- API 并发 8，初始输出上限 4096，响应缓存 4096，精确感知缓存开启。
- 这份代码有文本实例搜索提示和视觉实例验证，没有完整 FaCTNav 因子归因模块。

## Python 环境要求

先创建并激活适合本机 CUDA 的 Python 环境，依赖见 `README.md`、`environment.yml` 和 `reproduction/requirements-runtime.txt`。历史验证环境使用 Python 3.9.23、Habitat-Sim 0.2.4、transformers 4.26.0、PyTorch3D 0.7.9；PyTorch/CUDA 版本应与本机驱动和 GPU 匹配。

启动脚本默认使用当前 PATH 中的 `python`；可用 `PSGNAV_PYTHON=/absolute/path/to/python` 显式指定解释器。GLIP 补丁见 `patches/glip_torch2.patch`，GroundingDINO 的 PyTorch 2.7 构建适配见 `patches/groundingdino-torch27-build.patch`。这些补丁用于构建兼容性，按本机依赖版本选择使用。

## 使用环境与运行

新增启动器位于 `scripts/run_instance100.sh`，固定当前 100 条配置；不要使用历史默认 200 条启动命令代替。

```bash
cd /path/to/InstanceNav_AVS
conda activate psgnav-instance
bash scripts/run_instance100.sh --gpus 0,1 --timestamp instance100_qwen35_run1
```

当前启动器默认使用“类别引导探索＋完整描述实例核验”。也可显式指定：

```bash
bash scripts/run_instance100.sh --enable-instance-description-for-exploration false \
  --timestamp instance100_category_exploration_run1
```

也可设置环境变量 `ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION=false`，命令行参数优先。
该开关只控制地标选择提示词是否加入实例描述，实例核验继续使用完整描述；当前启动器默认 `false`。
可附加 `--check-only` 进行预检而不启动导航。开关值会写入各 GPU 日志和结构化运行头。

### 相对证据决策（S8）

方法默认关闭。以下命令开启方法，并关闭探索阶段的实例描述；实例核验及停止时的候选比较仍使用完整描述：

```bash
bash scripts/run_instance100.sh --enable-relative-evidence true \
  --enable-instance-description-for-exploration false --timestamp relative_category_run1
```

正式运行前可附加 `--check-only`。总开关关闭时保持当前基线；开启时要求实例核验开启、
`positive_match`、候选准入关闭、阈值 1、后置复验/受困证据保留/collect-only 关闭、RAG=0。
冲突配置会报错，不会自动覆盖。两两比较需要现有的 openai 兼容多图后端。

消融参数 `--relative-evidence-use-u`、`--relative-evidence-early-stop`、
`--relative-evidence-pairwise`、`--relative-evidence-drop-rejected` 均默认 `true`；
对应环境变量为 `RELATIVE_EVIDENCE_USE_U`、`RELATIVE_EVIDENCE_EARLY_STOP`、
`RELATIVE_EVIDENCE_PAIRWISE`、`RELATIVE_EVIDENCE_DROP_REJECTED`。
关闭 early-stop 时仍在第 860 个策略步兜底。返回预算读取实际剩余原子动作，按 `ceil(8*d+10)` 筛选。

冻结标定文件默认为 `configs/relative_evidence_s6.json`，可用
`--relative-evidence-calibration PATH` / `RELATIVE_EVIDENCE_CALIBRATION` 指定。
已知 val10 场景使用对应折外模型及停止参数，其他场景使用全开发集模型。
文件中的来源明确标记为“完整描述探索”；关闭探索描述后的运行属于标定迁移验证，不预设能复现旧离线增益。
运行头记录文件哈希、参数和子开关，逐 episode 记录实际模型与决策。

重建标定并检查全部 S6 观测的一致性（不调用大模型）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=habitat-lab:GLIP:. \
  python tools/export_relative_evidence.py
```

固定少量任务的真实冒烟（基线/方法分别使用 GPU 0/1，共六条，不是 100 条评测）：

```bash
PYTHONPATH=habitat-lab:GLIP:. python \
  tools/smoke_relative_evidence.py --run-id relative_smoke_YYYYMMDD --gpus 0,1
```

冒烟固定原 val10 JSON 索引 17（chair）、2（plant）、11（tv_monitor）的任务身份，
自动映射到按场景分组后的启动器索引；映射、源码哈希、逐任务日志和结果保存在该 run-id 的 outputs 目录。
同一次冒烟期间不能修改运行代码；需重跑时使用新的 run-id。

### CoIN-Bench（Context-Nav 协议）

启动器 `scripts/run_coin.sh`，配置 `configs/coin_contextnav.yaml`；它不调用也不修改 `run_instance100.sh`。
协议与 Context-Nav 一致：完整 `instruction`、无用户交互、500 原子步，STOP 时距目标视点测地距离 < 0.25 m 为成功；
RGB 512×512、HFOV 42°，深度 512×512、HFOV 79°、0–5 m，相机高 1.31 m，智能体 1.41 m/0.17 m，允许滑动；
智能体在直线距离目标 < 0.75 m 时停止（Context-Nav 的 `pointnav_stop_radius`）。
另记 `distractor_success`：在同类干扰物 0.35 m 内 STOP。导航开关默认值与 `run_instance100.sh` 相同；
相对证据只在 InstanceNav 上标定，CoIN 中禁用。

不带参数即为默认值：`val_unseen` 的固定 100 条子集（seed 77，覆盖全部 47 个目标：每个目标先抽 1 条，再从其余任务中随机补 53 条），GPU 0、1 共享任务队列。
这 100 条写在 `configs/coin_val_unseen_subset100_seed77.json`（受 git 管理，用 `tools/coin_subset.py --write-pool` 生成）。
之后 val_unseen 上的 `--subset N` 都只从这 100 条里抽（N ≤ 100）；`--full` 仍跑整个划分。其他两个划分不受影响，仍从整个划分中抽子集。

```bash
bash scripts/run_coin.sh                              # 默认：val_unseen 固定 100 条，GPU 0,1
bash scripts/run_coin.sh --full                       # val_unseen 整个划分（459 条）
bash scripts/run_coin.sh --split val_seen --subset 20 # 指定划分与子集大小
bash scripts/run_coin.sh --subset 3 --gpu 2           # 从固定 100 条中抽 3 条，单卡顺序运行（冒烟）
bash scripts/run_coin.sh --check-only                 # 只预检
```

默认值也可用环境变量改：`COIN_SPLIT`、`COIN_SUBSET`、`COIN_SEED`；命令行参数优先。
`--subset 0` 与 `--full` 等价，表示跑整个划分。
三个划分：val_seen 831 条、val_seen_synonyms 359 条、val_unseen 459 条（目标数 86/36/47）。
结果写到 `data/results_coin_<split>_<时间>/`，结束时自动生成 `summary.json`（SR、SPL、SoftSPL、DTG、
distractor_success；SR 的 95% 区间按目标重采样）和 `per_episode.csv`，也可手动运行
`tools/summarize_coin.py <结果目录>`。中断后用同一 `--timestamp` 加 `--resume`。

几何适配见 `utils/coin_geometry.py`：障碍图和可通行图用 79° 深度；检测框、掩码和场景图使用重采样到
42° RGB 视野的深度及 RGB 内参；投影高度取相机的 1.31 m，不取智能体高度。评测器自检不调用大模型，需要一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=habitat-lab:GLIP:. python \
  tools/check_coin_protocol.py --targets-per-split 3
```

中断恢复时使用同一 timestamp 并加 `--resume`。运行前先激活环境，按可用 GPU 调整 `CUDA_VISIBLE_DEVICES`，并使用启动器的 `--check-only` 检查本机资源。完整 GPU 导航或 API 验证需要自行准备数据、权重及 API Key；仓库整理本身不执行这些实验。
