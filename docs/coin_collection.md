# CoIN 三轮仅采集运行

当前入口显式使用 `match_strict_no`、关闭描述探索，并沿用 CoIN 官方传感器设置。
类别映射、场景图门槛和 FMM 停止配置见 [离线标定说明](coin_calibration.md)。
20261108 的历史采集使用 `match` 和描述探索开启，原记录保持不变。

手动启动（两卡 0、1，每轮固定的 100 个 val_unseen 起点，每个任务 500 个原子动作）：

```bash
cd /path/to/InstanceNav_AVS
bash scripts/run_continuous_coin_collection.sh
```

只检查配置、数据集和依赖，不运行导航或调用模型：

```bash
bash scripts/run_continuous_coin_collection.sh --check-only
```

三轮顺序运行，使用独立运行编号和进程，不共享跨轮模型回答缓存。
任一轮执行失败或采集完整性检查不通过，脚本返回非零状态并停止后续轮次。
现有普通连续评测脚本没有改成采集模式。

## 采集策略

保留基线的检测、接近、4 米内确认，以及 positive_match、一次 yes 的规则。
到本应锁定时记录 Intended Lock，将该导航候选标记为不再主动接近，然后继续探索。
该候选再次被看到时仍然确认。未增加主动复核、多视角采样或负证据导航规则。
规划器意外返回 STOP 时，采集模式改成一次 30 度右转，并记录 Collect Suppressed Stop；
普通导航模式的动作不受影响。500 步指原子动作数，60 度扫描按两个 30 度动作计数。

这意味着某些首次 yes 后的候选没有足够后续视角；离线应标记证据不足，不能当作多票规则失败。
采集结果的 SR/SPL 不适用，不能当作正常导航成绩。

## 文件和索引

- `data/results_coin_val_unseen_collect_<批次>_r1/`（以及 r2、r3）：常规结果、`run_info.json`、`summary.json`、`collection_check.json`。
- `outputs/coin_<运行编号>_gpu0.log`、`gpu1.log`：原始导航日志和结构化事件。
- 同名 `gpu0.collect/`、`gpu1.collect/`：`events.jsonl`、`images/*.png`、`depth/*.npz`。

图片和深度按内容哈希去重保存。RGB 是确认时的完整原图；另存实际发给模型的带编号框图片。
深度是策略使用的深度（CoIN 已对齐 RGB，100 为无效值），不是未经处理的传感器深度。
没有语义真值传感器、真实目标位置参与采集或候选关联。真值标签后续离线回放再添加。

每条事件带 `episode_attempt_id`、scene/episode ID、策略步和动作前原子步数。
重试任务使用新的 attempt ID，防止把中断前后的候选轨迹混合。

- `Collect Episode`：描述、采集策略和关联/确认距离。
- `Collect Observation`：所有目标类别候选（包括后续 no、unknown 和未询问的远处框），框坐标、检测分数、估计位置/距离、RGB/深度路径、GPS/朝向/俯仰，以及 `collect_track_id`。
- `Instance Verification`：实际提示词、JSON schema、模型配置、每次格式重试的输入、原始回答、解析结果、缓存命中与请求哈希。API 错误和格式错误独立记录，不转换成 unknown。
- `Collect Decisions`：每个原始框对应的阶段、verification ID、yes/no/unknown、是否保留及导航候选编号。没有询问的框其 decisions 为空，不冒充 unknown。
- `Collect Intended Lock`：本应锁定的候选与 frame ID、导航编号、票数；结合 Collect Decisions 映射到采集编号。
- `Collect End`：完成的原子步数、预算与锁定状态。

原有 Collect Frame（只含保留框）、Collect Snapshot、Landmark Utility 仍保留，方便旧工具读取。
新工具应使用 Observation/Decisions；原有 `track_id` 是导航编号，`collect_track_id` 是独立的被动观测编号。
关联先用场景图 node ID，再用位置的最大坐标差不超过 0.25 米；它不是实例真值。
没有位置和 node ID 的框无法跨帧可靠关联；歧义、位置抖动或检测路径变化可能产生分裂轨迹。
保存原框和位置允许离线重新关联，但不得用未来信息或真值决定当时的导航选择。

相同策略步、相同 RGB 哈希或相同请求哈希的回答不要直接视为独立视角。
采集器保留原始重复事件，不修改在线计票，离线分析时可分别按步、图片、视角去重。
分数只覆盖检测器实际输出的候选；降低检测阈值需要重新检测原图。距离超过 4 米的框
可能没有模型回答，若要研究更大的确认距离，需要追加离线模型调用。

## 验证与后续分析

每轮结束自动运行 `tools/check_coin_collection.py`，核对 100 个结果、500 动作预算、
无实际 STOP、完成 episode 与采集 attempt 对齐、候选/回答映射、缓存记录、图片和深度文件存在性。
这是结构完整性检查，不验证跟踪身份的语义正确性，也不把离线规则的正确率换算成导航 SR。
磁盘写入失败直接终止，避免继续消耗模型调用却缺失证据。

手工复查已有运行：

```bash
python tools/check_coin_collection.py data/results_coin_<运行编号>
```

离线按时间前缀比较确认票数、负票权重和近期窗口；同一目标的多个起点应分在同一组，
避免调参与验证共享目标。候选选择的结论最终仍需闭环评测。
