# 相对上游的改动清单

基准提交：181d878ff207845a31e542a19a6592925c1e9bab。安装与复现以 [REPRODUCE_ZH.md](REPRODUCE_ZH.md) 为准。以下是代码适配，不是 FactNav 实现。

| 文件/模块 | 改动 | 行为影响 |
|---|---|---|
| utils/model_client.py、graph/meta_func/flask_func.py、server.py、server_local.py | 云端 Qwen3-VL-Flash、dotenv、缓存、重试、并发；保留可选本地服务 | 模型不同；API 故障明确退出 |
| graph/meta_func/core_func.py | 独立图边视觉请求并发 | 保留结果对应关系 |
| utils/landmark_selector.py | 三轮比较并发、实例描述、只输出 A/B、非法响应报错 | 保留效用公式；提示/回退改变可能影响结果 |
| utils/instance_verifier.py、utils/object_detection.py | InstanceNav 候选整体匹配、格式重试、GLIP 类别和标签边界修复 | 只有 yes 保留；no/unknown 尚未保存为独立导航状态 |
| utils/instancenav_dataset.py、utils/ovon_dataset.py | 任务目标、路径、开放词汇、固定清单适配 | 任务侧新增能力 |
| configs/dataset_registry.py、configs/dataset_categories.py、utils/utils_glip.py | 数据集注册、类别映射、OVON 动态类别/确认阈值 | OVON 小物体分支为空、确认次数阈值 3 |
| configs/instancenav_val20_qwen.yaml | 200 条 / 1000 步 / 1 米成功距离 | 不启用原子动作展开 |
| configs/hm3d_ovon_qwen.yaml | 历史通用 OVON 配置 | 不用于固定 300 条复现 |
| configs/hm3d_ovon_unseen10_500atomic_1m.yaml | 固定 300 条 / 500 原子动作 / 1 米 | 必须使用此配置 |
| PSG_Nav.py | 配置化步数/停止、缓存和线程开关、seed、数据集加载、resume/config/check-agent、默认范围/可视化修复 | InstanceNav RAG 按 timestamp 隔离；原 RAG 标签反馈仍在 |
| utils/atomic_actions.py、habitat-lab/habitat/sims/habitat_simulator/actions.py | 注册并展开 60° 转向 | 原子模式两次 30° 各计一步 |
| habitat-lab/habitat/core/benchmark.py | 续跑前缀校验、环境步数、进度/ETA、身份和结果安全写入 | 重跑未完成 episode，不恢复完整随机状态 |
| habitat-lab/habitat/datasets/pointnav/pointnav_dataset.py | 场景切片边界和全场景默认值 | 不再默认只取前 20 场景 |
| utils/inference_cache.py、utils/utils_fmm/fmm_planner.py | 相同输入检测/特征/SAM/FMM 缓存 | 避免重复计算；API 缓存会复用答案 |
| utils/map_manager.py、graph/meta_func/nav_func.py | 房间类别列表与未知标签处理 | 防止未知类别崩溃 |
| utils/glip_compat.py、patches/glip_torch2.patch | 现代 torch 算子兼容与无梯度推理优化 | 需同时应用独立 GLIP 仓库补丁 |
| GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py、utils/utils_sg_uq/sam_utils.py | 导入/依赖兼容 | 目标机仍需编译匹配的 CUDA 扩展 |
| utils/utils_fmm/control_helper.py、utils/utils_sg_uq/merge_utils.py | math/Counter 导入修复 | 修复运行错误 |
| run_parallel_with_server.py | API 模式启动适配 | 不必启动本地语言模型 |
| scripts/setup_runtime.sh、scripts/build_pytorch3d.sh | 环境与依赖准备 | CUDA 架构需按目标机设置 |
| scripts/run_instancenav_val20_qwen.sh、scripts/run_ovon_val_unseen_qwen.sh、scripts/run_ovon_after_instancenav.sh | 启动、预检查、顺序运行 | 使用仓库内 Habitat 与 GLIP |
| scripts/run_ovon_unseen300_qwen.sh | 发布新增：固定 300 条正确配置和重试默认值 | 不改 agent 算法 |
| tools/download_instancenav.py、tools/prepare_instancenav.py、tools/download_hm3d_hf.py | 下载、固定采样、hash 校验 | 数据本身不提交 Git |
| tools/preflight_instancenav.py、tools/smoke_detectors.py、tools/smoke_navigation.py | 输入/API/检测器/模拟器检查 | smoke 不计正式评测 |
| tools/benchmark_perception.py、tools/profile_navigation.py、tests/ | 性能与回归验证 | 不能替代目标机完整试跑 |
| reproduction/ | 固定子集清单、模型校验和实测环境 | 支持新机器核对实验输入 |
| .env.example、.gitignore、requirements.txt、README.md、INSTANCENAV_QWEN.md、REPRODUCE_ZH.md | 配置模板、密钥忽略、文档 | 真实 .env 不发布 |

## 尚未实现

FactNav 的因子级解析、支持/矛盾/未确定证据表、候选实例绑定、提示生命周期、主动补证据闭环均未加入。现有 InstanceNav 整体验证只是任务适配。

默认 RAG 仍使用评测反馈收集正负例，导航失败不必然意味着检测错误。做无评测反馈的独立 episode 对照前应双方统一协议并重跑。
