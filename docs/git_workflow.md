# Git 使用约定

本仓库从当前工作区建立独立初始版本，默认分支为 `main`。提交身份仅在本仓库配置。

## 提交内容

保留源码、依赖清单、文档、许可证、必要配置和小型测试夹具。`configs/runtime/` 中的 JSON 是复现所需任务索引或清单，不是完整数据集。

不要提交数据集、模型权重、运行结果、日志、`.env`、私钥、本机配置、缓存或构建产物。运行产物放在 `outputs/` 或 `data/`。这些文件仍保存在本机，`.gitignore` 不会删除它们。模型不使用 Git LFS。

启动评测前激活 Python 环境，或设置 `PSGNAV_PYTHON`。可选 FastSAM 路径通过 `FASTSAM_WEIGHT_PATH`、`FASTSAM_CODE_PATH` 设置。

## 日常操作

```bash
git status --short
git add <具体源码或配置路径>
git diff --cached --stat
git diff --cached
# 确认没有密钥、数据和结果后提交
git commit -m "Describe the change"
git push
```

新增文件先检查忽略规则：`git check-ignore -v <路径>`。不要使用 `git add -f` 绕过保护规则。新文件超过 10 MiB 时先判断是否为下载资源或生成物；此类文件应留在 Git 外。

提交前建议运行 Gitleaks（`gitleaks git --redact .`）；扫描工作区可使用 `gitleaks dir --redact <待上传文件目录>`。真实密钥仅写入本机 `.env`，示例文件必须使用占位值。

大体积的可选文档图片和演示动画也不提交，清单见 [optional_assets.md](optional_assets.md)。导航所需小型资源继续保留。
