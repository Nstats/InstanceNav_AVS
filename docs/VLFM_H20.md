# H20 部署：只为 VLFM 准备的文件

前提：这台机器已有本项目可运行的 `psgnav-instance` 环境（包含为本机 PyTorch/CUDA 编译的 GroundingDINO 扩展）、CoIN 场景/数据和原有配置。AVS 仍需在 H20 本地配置核验接口。**不要复制 `.env`、`.so`、其他机器的 conda 环境或整套 Habitat。** 默认的 PSG-Nav 不读取本页的目录，也不启动这四个服务。

## 1. 源码用 Git，权重单独复制一次

`third_party/vlfm/` 是仓库内的推理源码，不是子模块。`nav_bases/` 是与现有评测框架的适配；不需要同级 `vlfm/`、`ContextNav/`、`.vlfm-deps`、`VLFM_ROOT` 或 `tmux`。上游版本和不同组件的授权见 [源码清单](../third_party/vlfm/README.md)。`data/` 本来就不进 Git，**不要强制添加模型文件**。

从已准备好的机器把 **`data/vlfm/` 目录里的内容**复制到 H20 上**项目根目录的 `data/vlfm/`**，不要多套一层 `vlfm/`。最终目录结构如下（复制文件内容，不要复制原机器的软链接）：

```text
data/vlfm/
├── pointnav_weights_state_dict.pth  # 已导出的离散 PointNav，不是 pointnav_weights.pth
├── groundingdino_swint_ogc.pth      # 与项目原有同名 DINO 权重可共用真实文件
├── mobile_sam.pt                    # TinyViT，不是 PSG-Nav 的 sam_vit_h_*.pth
├── yolov7-e6e.pt                    # 官方 E6E，启动前校验 SHA-256
├── blip2_pretrained.pth             # BLIP-2 图文匹配
├── eva_vit_g.pth                    # BLIP-2 的视觉编码器；前一个文件不包含它
└── bert-base-uncased/               # GroundingDINO 与 BLIP-2 共用的本地模型目录
    ├── config.json
    ├── vocab.txt
    ├── pytorch_model.bin
    ├── tokenizer.json               # 随已整理的快照一起复制
    └── tokenizer_config.json
```

这套文件合计约 **4.29 GB（3.99 GiB）**。`pointnav_weights_state_dict.json` 是导出记录，可以一起复制。不要复制 `traced_model.pt`、原始 PointNav pickle、Spot 权重、整个 Hugging Face 缓存或密钥。确需把模型放到别处，可以在 H20 上设置 `VLFM_ASSET_DIR` 指向同样布局的目录；不影响 PSG-Nav。

若从原始文件重新整理：三个检测/分割模型取 VLFM `data/` 中同名文件的**真实内容**；BLIP-2 与 EVA 常位于 `~/.cache/torch/hub/checkpoints/`；BERT 取 `bert-base-uncased` 快照 `86b5e0934494bd15c9632b12f734a8a67f723594`，把上表中文件解引用后放到同一目录。无需运行 PointNav 导出脚本。

## 2. 验证并使用（H20 只有 GPU 0）

在项目根目录、已激活的 `psgnav-instance` 环境中，先检查文件（不启动模型、不运行导航）：

```bash
bash scripts/launch_vlfm_servers.sh --check-only
```

如需逐字节确认复制的是这批已验证的文件：

```bash
(cd data/vlfm && sha256sum -c ../../third_party/vlfm/weights.sha256)
```

只启动或复用四个本地服务（12181 `/gdino`、12182 `/blip2itm`、12183 `/mobile_sam`、12184 `/yolov7`），不启动导航：

```bash
bash scripts/launch_vlfm_servers.sh --gpu 0
```

启动器使用当前 Python（可用 `VLFM_PYTHON` 或 `PSGNAV_PYTHON` 覆盖），模型用本地文件，不自动下载。日志在 `outputs/vlfm_servers/<会话名>/`。只有当前会话登记的四个进程均通过归属检查，且各端口返回对应 PID，才能复用。新会话只在四个端口全部空闲时启动；遇到别人的服务，请换端口，不要停止它。旧版本启动的服务没有 PID 响应头：等使用它的任务结束后，由原启动者停止对应会话，再重启。停止默认会话：

```bash
bash scripts/launch_vlfm_servers.sh --stop
```

[InstanceNav H20 队列](../scripts/run_continuous_evaluation_vlfm_base_h20.sh) 会自行启动 `instancenav_h20` 会话，不必先启动默认会话。它默认使用 22181–22184；可在运行队列前覆盖 `GROUNDING_DINO_PORT`、`BLIP2ITM_PORT`、`SAM_PORT`、`YOLOV7_PORT`，服务和导航会继承同一组端口。评测结束后，只停止该 H20 会话：

```bash
bash scripts/launch_vlfm_servers.sh --stop --session instancenav_h20
```

**预检 InstanceNav 的提示词④（不启动导航）：**

```bash
bash scripts/run_instance_ablation.sh A0 --nav-base vlfm --instance-verification-prompt match_visible_evidence --gpus 0 --check-only
```

`A0` 是起始预设，不是实验组合白名单。普通 `A0` 默认沿用 `match_strict_no`；上面的显式参数优先，也可以将 `vlfm` 换成 `psgnav`。旧的 `A0VisibleEvidence` 是相同参数的快捷写法。`--dry-run` 只校验配置并打印最终参数；正式运行时用新的 `--timestamp`，恢复同一次运行时 `--resume` 须使用相同的有效参数。InstanceNav 的 VLFM 支持 A0，以及主动接近的 B、带独立准入预筛的 RA、同一候选在不同策略步得到两票才锁定的 E5；未实现的组合仍在预检拒绝。三种接近配置不改变确认前的目标地图，临时目标到达或 PointNav 返回 STOP 后退回前沿探索。

CoIN 同样支持显式覆盖，预检示例：

```bash
bash scripts/run_coin_e0.sh --split val_unseen --subset 100 --nav-base vlfm --instance-verification-prompt match_visible_evidence --gpu 0 --check-only
```

当前 [CoIN H20 队列](../scripts/run_continuous_evaluation_coin_h20.sh) 只运行 PSG-Nav，不会启动 VLFM。需要在 H20 测 VLFM 时，先按上文启动服务，再用 CoIN 启动器显式指定 `--nav-base vlfm`、`--gpu 0` 和新的 `--timestamp`。仅检查部署时，运行 `--check-only` 即可，不需要正式测评。
