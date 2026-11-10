# psg+ 论文提纲

## 摘要（前半部分）

文本目标实例导航要求智能体在未知环境中，从同类候选中定位同时满足类别、属性与空间关系约束的目标实例。然而，如何整合跨观测跨约束的证据，区分约束冲突与证据不足，并据此协调候选筛选、主动补证与停止决策，仍是零样本实例导航的关键挑战。

为此，我们基于 PSG-Nav 提出 psg+，一种约束级证据驱动的零样本实例导航框架，通过持续维护逐因子证据与信念，将验证中的证据缺口转化为探索子目标，并以统一的约束证据状态协调候选筛选、主动补证与停止决策。首先，构建因子化约束感知信念图（Factorized Constraint-Aware Belief Graph），将导航目标分解为类别、属性与关系因子，解耦约束可判定性与满足性判决，通过有效证据的跨观测积累与修正动态更新逐因子信念。其次，设计归因引导探索机制（Attribution-Guided Exploration），依据约束冲突与证据缺口的具体来源，选择候选筛选或针对性补证行动，并利用新观测更新信念。最后，采用证据记忆联合停止机制（Joint Evidence-Memory Stopping），综合当前观测与累计约束证据进行目标确认，将尚未解决的证据需求反馈至探索过程，形成证据维护、行动选择与停止判断的闭环。

## Introduction

### Contributions

本文的主要贡献如下：

1. **提出约束级证据驱动的零样本实例导航框架 psg+。** 将目标因子分解、逐因子信念维护、归因引导探索与联合停止纳入统一闭环，使约束证据持续参与候选筛选、观测目标选择与终止判断，建立从细粒度验证反馈到导航行动的明确联系。
2. **构建因子化约束感知信念图。** 围绕目标类别、物体属性与空间关系，分别维护约束可判定性与满足性判决，并通过有效证据的跨观测积累与修正导出累计信念，为区分可靠约束冲突与尚未解决的证据缺口提供结构化依据。
3. **设计由约束归因连接探索与停止的决策机制。** 通过 Attribution-Guided Exploration，将具体约束的证据缺口映射为候选检查、参考物搜索或关系验证等观测子目标；通过 Joint Evidence-Memory Stopping，结合当前观测与历史约束证据判断目标是否得到充分确认，使补证行动及其终止条件建立在一致的证据状态之上。



## Related Work



### 概率场景表示与导航决策：PSG-Nav

PSG-Nav 使用概率场景图维护对象的语义类别分布，通过 Multiverse Decision 在多个可能的场景解释下评估导航地标，并利用 Evidential Experience Calibrator 检索历史成功与失败经验，校准候选检测及停止判断。其信息获取机制也考虑未知空间覆盖与对象类别不确定性，因此，概率信念、历史经验和不确定性引导探索均属于 psg+ 所继承的研究基础。[PSG-Nav，§3.2–3.4](https://arxiv.org/html/2606.01313v1)

psg+ 在该导航框架上增加面向当前目标描述的约束级证据层：除对象的类别分布外，进一步维护各候选对类别、属性和关系因子的满足信念及证据充分性。这使智能体能够将验证未通过归因到具体约束，并据此生成补证任务。相较于原方法对语义类别不确定性和历史误检经验的建模，psg+ 重点研究多约束实例目标的证据维护，以及这些证据如何共同控制候选筛选、探索与停止。

### 细粒度实例验证与上下文搜索：Context-Nav

Context-Nav 将完整目标描述用于探索价值图，同时分别验证内在属性与外在空间关系。其属性验证已区分 Yes、Unknown 和 No，对未知结果使用后续观测重新查询；当目标所在房间内的参考物尚未发现时，还会触发一次局部探索。因此，该方法已经具备细粒度验证、不确定性处理及上下文搜索机制。[Context-Nav，§3.3–3.4](https://arxiv.org/html/2603.09506v3)

psg+ 的重点是将这些验证需求组织为持续更新的逐因子证据状态，并让该状态在整个搜索过程中直接参与行动选择。对于属性不可辨、参考物未定位和关系证据不足等不同情况，归因引导探索生成相应的观测子目标，再将行动获得的新证据反馈至同一信念图。由此，psg+ 与 Context-Nav 的比较应围绕跨观测证据维护及其对探索和停止的持续作用展开，属性分解或三态判断本身属于两者共享的机制基础。

### 主动观测与目标验证：SAP-Nav

SAP-Nav 面向层次化开放词汇导航，结合在线空间语义表示与 Active Viewpoint Verification。它先从可见性和观察角度评估当前视角是否足以验证候选，证据不足时主动移动，再检查类别与属性约束。其视角选择在可达性等几何条件约束下优化目标可见性，已经实现了观测充分性评估、主动换视角与候选验证的连接。[SAP-Nav，Active Viewpoint Verification](https://arxiv.org/html/2608.12707v1)

psg+ 将补证需求细化到目标描述中的具体因子，并以证据缺口的来源确定观测对象和任务类型。例如，颜色证据不足对应候选检查，参考物缺失对应参考物搜索，空间关系未决对应关系验证。相较于以候选视角充分性和几何可见性为中心的主动验证，psg+ 关注不同约束缺口如何形成不同的导航子目标，以及所得证据如何持续更新信念并影响停止判断。

### 场景记忆与细粒度目标确认：OmniNav

OmniNav 维护可更新的场景记忆，并在细粒度目标确认时，将查询图分别与记忆子图及当前观测子图匹配。它逐项评估属性和关系约束，对不可获得的约束赋予中性分数，对可评估但不一致的约束赋予低分，再聚合并融合两类来源的分数，决定确认目标或继续探索。这表明，区分缺失信息与不匹配、结合记忆和当前观测进行确认，已有直接相关研究。[OmniNav，§V-E](https://arxiv.org/html/2609.08159)

psg+ 进一步将逐因子累计证据及其未决原因保留为行动决策的输入，使归因结果能够指定下一项需要获取的证据。联合停止与归因引导探索共享这一状态：候选尚未得到充分确认时，具体的证据需求被反馈至探索；完成补证后，更新后的约束信念再次参与目标确认。本文关注的差异因此在于逐因子证据如何连接探索与停止，而记忆和当前观测的联合使用属于已有基础。

### 证据累计与探索—停止闭环：AECNav

AECNav 面向零样本开放词汇物体导航，为候选三维对象簇累计目标类别的 log-odds 信念，结合目标支持、混淆类别和预期检测缺失等信息更新证据。在尚不能确认目标时，它依据语义相关性、预期未知空间覆盖和运动成本选择探索方向，并结合累计信念的时间稳定性与空间距离决定停止。[AECNav，§III-B–III-C](https://arxiv.org/html/2608.10817v2)

psg+ 与 AECNav 均通过证据更新连接探索和停止。两者的主要区别在于证据的组织粒度和探索的决策依据：AECNav 维护候选属于目标类别的信念；psg+ 面向由类别、属性及关系共同限定的实例目标，维护逐约束证据，并根据具体约束的未决原因选择补证任务。因此，psg+ 的贡献定位是多约束证据与归因决策的闭环协同，证据累计和闭环导航本身不作为独立的新颖性主张。

### 本节参考文献

1. Rufeng Chen, Yue Chang, Xiaqiang Tang, Hechang Chen, Sihong Xie. **PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making.** ICML 2026（arXiv 页面标注）。[论文与发表信息](https://arxiv.org/abs/2606.01313)
2. Won Shik Jang, Ue-Hwan Kim. **Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation.** CVPR 2026, pp. 9626–9636。[正式发表页面](https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Context-Nav_Context-Driven_Exploration_and_Viewpoint-Aware_3D_Spatial_Reasoning_for_Instance_CVPR_2026_paper.html)
3. Xuetong Pei et al. **SAP-Nav: Spatial Semantic Representation Meets Active Perception for Hierarchical Open-Vocabulary Object Navigation.** arXiv:2608.12707, 2026，预印本。[论文](https://arxiv.org/abs/2608.12707)
4. Yujie Tang et al. **OmniNav: Robust Long-Horizon Target Navigation in Dynamic Environments.** arXiv:2609.08159, 2026，预印本。[论文](https://arxiv.org/abs/2609.08159)
5. Guanlin Liu et al. **AECNav: Active Evidence Consolidation for Efficient Zero-Shot Open-Vocabulary Object Navigation.** arXiv:2608.10817, 2026，预印本。[论文](https://arxiv.org/abs/2608.10817)



## Method



## Experiments



### Settings



### Main Results



## Conclusion

