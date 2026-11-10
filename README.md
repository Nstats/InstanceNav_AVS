# AVS-Nav：接近—核验—切换的零样本实例导航

**Approach–Verify–Switch (AVS)**

本仓库是 AVS-Nav 论文的研究代码，面向**由完整文本描述指定目标的零样本实例导航**。智能体需要从同类物体中找到描述对应的目标实例，并决定何时停止。AVS 将候选接近、视觉语言核验、目标承诺和候选切换组成一个闭环，无需为该策略额外训练或标定。

核心规则是：**近处单次正面判断触发承诺，单次未通过不永久排除候选，行动结果决定何时释放候选。**

当前实现以 PSG-Nav 的探索与导航模块为基础，保留了部分原有文件名和环境变量名（如 `PSG_Nav.py`、`PSGNAV_PYTHON`）。本项目的研究对象是 AVS 的实例决策闭环，PSG-Nav 是其实现基座。

## 方法概述

类别检测可以发现“一个椅子”，但不能直接决定它是否是描述中的“那把椅子”。AVS 用完整目标描述核验候选，并将当前观察与候选的后续处理分开：

1. **提出候选。** 同类检测直接进入候选集合，主配置不进行额外的大模型准入筛选。
2. **接近（Approach）。** 未锁定目标时，暂停地标导航并接近最近的候选；策略估计距离在 4 m 以内时才进行实例核验。
3. **核验（Verify）。** 将整张 RGB 图像、候选编号框与完整目标描述交给视觉语言模型。`match_strict_no` 提示词区分 `yes`、`no` 和 `unknown`：明确可见的矛盾对应 `no`，遮挡、尺度过小或上下文不可见等证据不足对应 `unknown`。
4. **承诺。** 一次 `yes` 即锁定候选位置，随后导航至停止距离；锁定后不继续核验。InstanceNav 中，策略以预测目标位置的 FMM 地图距离 <1 m 触发 STOP，官方成功则按目标视点的测地距离 <1 m 判定，两者需要区分。
5. **保留后续机会。** `no` 和 `unknown` 都不形成确认票，也不构成永久黑名单；候选仍可在后续观察中再次接受核验。两种回答具有不同的证据含义。
6. **切换（Switch）。** 到达候选附近仍未被认可、出现更近候选或受困时，释放当前候选并切换；没有候选则恢复探索。受困恢复包含转向、前进及必要的随机导航，释放后的候选仍可被再次接近。

AVS 的当前主配置使用整体描述核验，不进行属性因子分解、专门的视角规划或多票承诺。距离门控用于组织接近与核验流程，不预设“走得越近，单帧判断就越准”。

## 实验状态与结果

以下为论文草稿中 `match_strict_no` 主配置的开发集结果，使用 qwen3.5-flash：

- **评测集合：** InstanceNav val10 固定 100 个起点、99 个目标实例，重复运行三次；这是 300 次执行，不是 300 个独立任务。
- **成功率（SR）：** 三次分别为 **36%、37%、36%**，均值 **36.33%**。
- **路径加权成功率（SPL）：** 均值 **0.1424**。
- **开销：** 平均 **500.29 个原子动作**、**221.72 次实际 API 请求**；动作预算为每次执行 1000 个原子动作。

在可标注序列中，100 次执行最终接受了目标，其中 47 次此前出现过 `no` 或 `unknown`。这一观察支持为尚未通过核验的候选保留后续机会，同时不排除持续观察带来的误接受和开销。

这些结果不证明单次 `yes` 是最优承诺门槛，也不证明明确 `no` 没有信息。匹配探索输入与提示词的消融、完整验证集评测和公平的跨方法比较仍需补充。仓库另提供 CoIN-Bench 适配入口，其协议和结果应与上述 InstanceNav 实验分开报告。

本说明依据本地论文草稿 `outputs/paper_AVSNav.md` 整理。该草稿及原始实验报告保留在本机，不随源码上传；论文正式标题、公开链接及引用信息待定稿后补充。

## 环境与资源准备

部署步骤与本地资源约定见 [部署说明](DEPLOYMENT_ZH.md)，依赖记录见 [environment.yml](environment.yml) 和 [运行依赖清单](reproduction/requirements-runtime.txt)。当前环境记录包含 Python 3.9、Habitat-Sim 0.2.4、PyTorch、PyTorch3D 及感知模型依赖；PyTorch/CUDA 组合需适配本机驱动。

运行前准备以下资源：

- **场景与数据集：** HM3D 场景放在 `data/scene_datasets/`；InstanceNav 子集放在 `data/datasets/instancenav/val10/`。评测配置及复现清单位于 [configs/runtime](configs/runtime)。
- **感知模型：** GLIP 权重位于 `GLIP/MODEL/`，SAM 与 GroundingDINO 权重位于 `data/models/`；本地 BERT 和 NLTK 资源分别位于 `data/huggingface/` 与 `data/nltk_data/`。
- **模型接口：** 将 `.env.example` 复制为 `.env`，在本机填写对应提供方的 API Key。主配置通过 DashScope 兼容接口调用 qwen3.5-flash。

```bash
# 激活已配置好依赖的环境
conda activate psgnav-instance
export PSGNAV_PYTHON="$(command -v python)"

# 首次配置时创建本机环境文件；已有 .env 时保留原文件
if [ ! -e .env ]; then cp .env.example .env; fi
chmod 600 .env
# 在本机编辑 .env，填写 API Key
```

## 运行 AVS

### InstanceNav 主配置

使用 [scripts/run_instance100.sh](scripts/run_instance100.sh) 评测固定 100 条任务。建议先用 `--check-only` 检查输入，不启动导航：

```bash
bash scripts/run_instance100.sh --check-only
```

下面显式指定论文主配置的关键开关，避免同名环境变量改变实验设置：

```bash
bash scripts/run_instance100.sh \
  --gpus 0,1 \
  --timestamp avs_instancenav_run1 \
  --vlm qwen3.5-flash \
  --enable-instance-description-for-exploration false \
  --enable-instance-confirmation true \
  --verify-before-candidate false \
  --instance-confirmation-mode positive_match \
  --instance-verification-prompt match_strict_no \
  --instance-confirmation-threshold 1 \
  --enable-post-confirmation-contradiction-check false \
  --preserve-positive-evidence-on-recovery false \
  --instance-collect-only false \
  --enable-relative-evidence false \
  --rag-mode 0
```

探索阶段只使用类别名，实例核验始终使用完整描述。脚本读取 [InstanceNav 配置](configs/runtime/instancenav_unseen10_1000atomic_1m.yaml)，使用 1000 原子动作预算。GPU 编号应按本机资源调整；重复实验使用不同的 `--timestamp`，恢复中断运行则保持原配置、使用相同时间戳并添加 `--resume`。

仓库保留了相对证据决策、后置复验和只采集等研究选项；上述 AVS 主配置关闭这些选项。

### CoIN-Bench 适配

[scripts/run_coin.sh](scripts/run_coin.sh) 提供 CoIN-Bench 的独立评测入口，默认使用 `val_unseen` 的固定 100 条子集。其预算为 500 原子动作，官方成功要求 STOP 时距目标视点的测地距离 <0.25 m；不能沿用 InstanceNav 的距离与预算解释结果。

```bash
bash scripts/run_coin.sh --check-only
bash scripts/run_coin.sh --timestamp avs_coin_run1
```

数据配置见 [configs/coin_contextnav.yaml](configs/coin_contextnav.yaml)，固定子集见 [configs/coin_val_unseen_subset100_seed77.json](configs/coin_val_unseen_subset100_seed77.json)。采集与离线分析说明见 [CoIN 采集文档](docs/coin_collection.md)。

## 代码入口

- [PSG_Nav.py](PSG_Nav.py)：导航主循环与受困恢复。
- [utils/object_detection.py](utils/object_detection.py)：候选检测、接近目标管理、实例核验与锁定。
- [utils/instance_verifier.py](utils/instance_verifier.py)：核验提示词与回答解析。
- [utils/model_client.py](utils/model_client.py)：模型接口、重试与缓存。
- [configs/navigation_config.py](configs/navigation_config.py)：导航和核验开关。
- [graph](graph)、[utils/map_manager.py](utils/map_manager.py)：场景图与地图模块。
- [tests](tests)：配置、数据适配和方法组件的测试。

## 文件管理

仓库跟踪源码、配置、依赖声明、文档和小型测试夹具。数据集、模型权重、API Key、运行日志与实验结果均保留在本机；运行产物放在 `outputs/` 或 `data/` 下。任务索引与复现清单可跟踪，完整数据集不上传。

部分大型文档图片和演示动画也保留本地，见 [可选资源清单](docs/optional_assets.md)。日常提交规则见 [Git 使用说明](docs/git_workflow.md)。

## 致谢与许可证

AVS 当前实现继承了 PSG-Nav / SG-Nav 的探索与导航基础，并使用 Habitat、GLIP、GroundingDINO、Segment Anything 等项目的组件。感谢相关开源项目的作者与维护者。

仓库保留现有 [LICENSE](LICENSE) 及第三方源码中的许可声明；使用相关组件时请同时遵守其许可要求。
