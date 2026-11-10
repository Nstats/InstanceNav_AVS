#!/usr/bin/env python3
"""Target-grouped five-fold selection and paired bootstrap for offline CoIN replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def bootstrap_interval(values, seed=77, repeats=5000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [0., 0.]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def select_grouped(rows, candidates, baseline, score, tie=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row['target']].append(row)
    targets = sorted(groups)
    means = {t: {k: float(np.mean([score(row, k) for row in groups[t]])) for k in candidates} for t in targets}
    def best(selected):
        if not selected:
            return baseline
        def ranking(k):
            secondary = ()
            if tie is not None:
                secondary = tuple(np.mean([np.mean([tie(r, k) for r in groups[t]], axis=0)
                                           for t in selected], axis=0))
            return (np.mean([means[t][k] for t in selected]), *secondary, -abs(k-baseline), -k)
        return max(candidates, key=ranking)
    rng = np.random.default_rng(77)
    shuffled = rng.permutation(targets).tolist()
    folds = [shuffled[i::5] for i in range(5)]
    deltas, selections, heldout = [], [], []
    for fold, test in enumerate(folds):
        train = [t for t in targets if t not in test]
        chosen = best(train)
        selections.append(chosen)
        for target in test:
            deltas.append(means[target][chosen] - means[target][baseline])
            heldout.append({'target': target, 'fold': fold, 'choice': chosen})
    interval = bootstrap_interval(deltas)
    candidate = best(targets)
    accepted = len(targets) >= 5 and interval[0] > 0
    return {'targets': len(targets), 'candidate': candidate, 'selected': candidate if accepted else baseline,
            'accepted': accepted, 'oof_delta_mean': float(np.mean(deltas)) if deltas else 0.,
            'oof_delta_ci95': interval, 'fold_choices': selections, 'fold_assignment': heldout,
            'target_macro_score': {str(k): float(np.mean([means[t][k] for t in targets])) if targets else None
                                   for k in candidates}}


def summarize(root):
    def load(stage):
        return [json.loads(p.read_text()) for p in sorted((root / stage).glob('*.json'))]
    audits = load('audit')
    valid = {row['key'] for row in audits if row['valid']}
    nodes = [r for r in load('nodes') if r['valid'] and r['key'] in valid and r.get('node_schema') == 6]
    stops = [r for r in load('stops') if r['valid'] and r['key'] in valid and r.get('stop_schema') == 2]
    # Branch restoration can introduce tiny renderer differences within the
    # audit tolerance. Keep those runs for inspection, but do not fit a stop
    # threshold on them: the plan requires consistent observations for fitting.
    exact_stops = [r for r in stops if r['rgb_nonidentical'] == 0 and r['max_depth_p99_error_m'] == 0]
    observation_excluded = [r for r in stops if r not in exact_stops]
    complete = len(audits) == 200 and len(nodes) == len(stops) == len(valid)
    node_result = select_grouped(nodes, [1, 2, 3, 4, 5], 3, lambda r, k: r['node_metrics'][str(k)]['f1'])
    book_rows = [r for r in nodes if r['category'] == 'book']
    book_diagnostics = {'executions': len(book_rows), 'targets': len({r['target'] for r in book_rows}),
        'target_available_executions': {str(k): sum(r['node_metrics'][str(k)]['target_first_atomic_step'] is not None
                                                     for r in book_rows) for k in range(1, 6)}}
    node_totals = {}
    for k in map(str, range(1, 6)):
        grouped = defaultdict(list)
        for row in nodes:
            grouped[row['target']].append(row['node_metrics'][k])
        firsts = [r['node_metrics'][k]['target_first_atomic_step'] for r in nodes]
        firsts = [x for x in firsts if x is not None]
        total = {field: sum(r['node_metrics'][k][field] for r in nodes)
                 for field in ('true_nodes', 'false_nodes', 'unknown_nodes', 'book_nodes_rescued')}
        total.update({f'target_macro_{field}': float(np.mean([
            np.mean([r[field] for r in group]) for group in grouped.values()])) if grouped else None
            for field in ('precision', 'recall', 'f1')})
        total.update(target_available_executions=len(firsts),
                     target_first_step_median=float(np.median(firsts)) if firsts else None,
                     false_node_fraction=total['false_nodes'] / max(1, total['true_nodes'] + total['false_nodes']))
        node_totals[k] = total
    overrides = {}
    category_results = {}
    default = node_result['selected']
    for category in sorted({r['category'] for r in nodes}):
        rows = [r for r in nodes if r['category'] == category]
        result = select_grouped(rows, [1, 2, 3, 4, 5], default, lambda r, k: r['node_metrics'][str(k)]['f1'])
        candidate = result['selected']
        round_gains = []
        for run in sorted({r['run'] for r in rows}):
            by_target = defaultdict(list)
            for row in rows:
                if row['run'] == run:
                    by_target[row['target']].append(row['node_metrics'][str(candidate)]['f1'] - row['node_metrics'][str(default)]['f1'])
            round_gains.append(float(np.mean([np.mean(v) for v in by_target.values()])))
        result['round_gains'] = round_gains
        if result['targets'] >= 10 and result['accepted'] and len(round_gains) == 2 and min(round_gains) > 0:
            from configs.dataset_categories import dataset_to_unified_name
            overrides[dataset_to_unified_name(category, 'coin')] = candidate
        category_results[category] = result
    branch_groups = []
    all_branches = []
    for row in exact_stops:
        by_lock = defaultdict(dict)
        for b in row['branches']:
            all_branches.append({**b, 'target': row['target'], 'run': row['run']})
            if b['metric'] == 'fmm':
                by_lock[b['frame_id'], b['navigation_track_id']][b['radius_m']] = b
        for lock, branches in by_lock.items():
            if set(branches) == {.25, .5, .75, 1.}:
                branch_groups.append({'target': row['target'], 'run': row['run'], 'branches': branches})
    stop_result = select_grouped(branch_groups, [.25, .5, .75, 1.], .75,
                                 lambda r, k: float(r['branches'][k]['success']),
                                 tie=lambda r, k: (-int(r['branches'][k]['reason'] == 'stop' and
                                                       not r['branches'][k]['success']),
                                                   -r['branches'][k]['extra_actions']))
    stop_totals = {}
    for metric, radius in [('fmm', .25), ('fmm', .5), ('fmm', .75), ('fmm', 1.), ('euclidean', .75)]:
        rows = [b for b in all_branches if b['metric'] == metric and b['radius_m'] == radius]
        strata = {}
        for label in ('target', 'same_category', 'other', 'unlabeled'):
            matches = [b for b in rows if b['target_label']['label'] == label]
            if matches:
                distances = [b['stop_dtg_m'] for b in matches if b['stop_dtg_m'] is not None]
                strata[label] = {'n': len(matches), 'success': sum(b['success'] for b in matches),
                    'wrong_stop': sum(b['reason'] == 'stop' and not b['success'] for b in matches),
                    'stop_dtg_median_m': float(np.median(distances)) if distances else None,
                    'mean_extra_actions': float(np.mean([b['extra_actions'] for b in matches])),
                    'collisions': sum(b['collisions'] for b in matches)}
        distances = [b['stop_dtg_m'] for b in rows if b['stop_dtg_m'] is not None]
        stop_totals[f'{metric}_{radius:g}'] = {'branches': len(rows), 'success': sum(b['success'] for b in rows),
            'wrong_stop': sum(b['reason'] == 'stop' and not b['success'] for b in rows),
            'unlock': sum(b['reason'].startswith('unlock_') for b in rows),
            'budget': sum(b['reason'] == 'budget' for b in rows),
            'exploration_boundary': sum(b['reason'] == 'exploration_boundary' for b in rows),
            'mean_extra_actions': float(np.mean([b['extra_actions'] for b in rows])) if rows else None,
            'collisions': sum(b['collisions'] for b in rows),
            'stop_dtg_median_m': float(np.median(distances)) if distances else None,
            'stop_dtg_p90_m': float(np.quantile(distances, .9)) if distances else None,
            'target_identity_strata': strata}
    # An incomplete replay can be inspected but never promotes a tuned setting.
    recommended = {'node_min_detections': default if complete else 3,
                   'node_overrides': overrides if complete else {},
                   'stop_metric': 'fmm', 'stop_distance_m': stop_result['selected'] if complete else .75}
    result = {'complete': complete, 'coverage': {'audited': len(audits), 'valid': len(valid),
              'nodes': len(nodes), 'stops': len(stops), 'conditional_locks': len(branch_groups),
              'continuation_executions': sum(bool(r['branches']) for r in stops),
              'all_recoverable_locks': sum(len(r['branches']) // 5 for r in stops),
              'stop_fit_executions': len(exact_stops),
              'stop_fit_continuation_executions': sum(bool(r['branches']) for r in exact_stops),
              'observation_excluded_executions': len(observation_excluded),
              'observation_excluded_locks': sum(len(r['branches']) // 5 for r in observation_excluded),
              'excluded_locks': sum(len(r['excluded_locks']) for r in stops)},
              'recommended': recommended, 'node_selection': node_result,
              'node_totals': node_totals,
              'book_diagnostics': book_diagnostics,
              'node_category_selection': category_results, 'stop_selection': stop_result,
              'stop_totals': stop_totals,
              'stop_observation_exclusions': [{'key': r['key'], 'rgb_nonidentical': r['rgb_nonidentical'],
                    'max_rgb_mae': r['max_rgb_mae'], 'max_depth_p99_error_m': r['max_depth_p99_error_m'],
                    'reason': 'nonexact_observations_after_branch_restore'} for r in observation_excluded],
              'limitations': ['Fixed trajectories were collected with prompt=match and description exploration=true.',
                              'Point-cloud RNG state was not saved; nodes are regenerated with fixed per-attempt seeds.',
                              'Local continuations condition on historical locks with recoverable controller state.',
                              'These are offline proxies, not full-navigation success rates.']}
    (root / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = ['# CoIN 配置对齐与离线标定结果', '',
             f"状态：{'完成' if complete else '计算中，参数尚未定稿'}。", '',
             '输入：20261108 collect-only r1/r2，200 次执行、100 个起点、47 个目标实例。',
             f"轨迹核对 {len(audits)} 次，通过 {len(valid)} 次；节点重建 {len(nodes)} 次，局部续跑阶段核查 {len(stops)} 次。",
             f"其中 {result['coverage']['continuation_executions']} 次执行存在可恢复的确认事件，"
             f"共 {result['coverage']['all_recoverable_locks']} 个；控制状态不可恢复的 {result['coverage']['excluded_locks']} 个事件排除。",
             f"停止阈值只使用 RGB/深度逐值一致的 {len(exact_stops)} 次执行，其中 "
             f"{result['coverage']['stop_fit_continuation_executions']} 次有 {len(branch_groups)} 个事件、涉及 {stop_result['targets']} 个目标。",
             f"另 {len(observation_excluded)} 次执行（{result['coverage']['observation_excluded_locks']} 个事件）"
             '存在极小观测差异，单独保留记录，不用于选择阈值。', '',
             '## 当前配置', '',
             '- 描述探索关闭；提示词 match_strict_no；GLIP 原始词表与官方传感器设置保留。',
             '- 小物体：plant、tv_monitor、book；只有 CoIN book 豁免 0.4 米跨度过滤。',
             f"- 节点默认次数：{recommended['node_min_detections']}；类别覆盖：{recommended['node_overrides']}。",
             f"- 内部停止：FMM {recommended['stop_distance_m']} 米；官方成功距离仍为 0.25 米。", '',
             '## 节点门槛', '',
             'F1 先按执行计算，再按目标实例平均；真值要求物体身份与预测类别相容。', '']
    for k, f1 in node_result['target_macro_score'].items():
        if f1 is not None:
            totals = node_totals[k]
            lines.append(f"- {k} 次：宏平均 P={totals['target_macro_precision']:.4f}、"
                         f"R={totals['target_macro_recall']:.4f}、F1={f1:.4f}；"
                         f"伪节点比例 {totals['false_node_fraction']:.1%}，"
                         f"可用目标节点 {totals['target_available_executions']}/{len(nodes)} 次执行，"
                         f"首次可用步数中位数 {totals['target_first_step_median']}；"
                         f"书籍尺寸例外挽回 {totals['book_nodes_rescued']} 个预测节点。")
    if book_rows:
        lines += ['', f"书籍专项只有 {book_diagnostics['targets']} 个目标、{len(book_rows)} 次执行。"
                  f"1 次门槛下出现可用真实目标节点的执行数为 {book_diagnostics['target_available_executions']['1']}/{len(book_rows)}；"
                  '因此尺寸例外挽回的预测节点数不能解释为目标书籍召回或导航成功的提升。']
    lines += [f"五折留出改善均值 {node_result['oof_delta_mean']:.4f}，95% 区间 {node_result['oof_delta_ci95']}。", '',
              '逐类别门槛仍以统一值为基准；样本数按不同目标实例计算：', '']
    if complete:
        chosen, base = node_totals[str(default)], node_totals['3']
        lines[-2:-2] = [f"五折候选为 {node_result['fold_choices']}；最终统一值为 {default} 次。"
                       f"与 3 次基准相比，宏平均精确率 {base['target_macro_precision']:.4f} → {chosen['target_macro_precision']:.4f}，"
                       f"召回 {base['target_macro_recall']:.4f} → {chosen['target_macro_recall']:.4f}。"
                       '节点 F1 的改善不等于完整导航成功率改善。', '']
    for category, selected in category_results.items():
        if selected['targets'] < 10:
            reason = '不足 10 个目标，不单独拟合'
        elif not selected['accepted']:
            reason = '分组验证未证明优于统一值'
        elif len(selected['round_gains']) != 2 or min(selected['round_gains']) <= 0:
            reason = '两轮未同时改善'
        else:
            reason = f"支持覆盖为 {selected['selected']} 次" if complete else '等待全部结果'
        lines.append(f"- {category}：{selected['targets']} 个目标，{reason}。")
    lines += ['', '## 停止距离', '', '以下成功数仅指历史锁定后的局部续跑，不是完整导航成功率。', '']
    for name, totals in stop_totals.items():
        if totals['branches']:
            lines.append(f"- {name}：成功 {totals['success']}/{totals['branches']}，错误 STOP {totals['wrong_stop']}，"
                         f"解锁 {totals['unlock']}，预算耗尽 {totals['budget']}，重新探索边界 {totals['exploration_boundary']}，"
                         f"平均追加 {totals['mean_extra_actions']:.1f} 动作，碰撞 {totals['collisions']} 次。")
            strata = totals['target_identity_strata']
            target = strata.get('target', {'n': 0, 'success': 0, 'wrong_stop': 0})
            wrong = [v for label, v in strata.items() if label != 'target']
            lines.append(f"  确认正确组：成功 {target['success']}/{target['n']}、错误 STOP {target['wrong_stop']}；"
                         f"确认错误组：成功 {sum(v['success'] for v in wrong)}/{sum(v['n'] for v in wrong)}、"
                         f"错误 STOP {sum(v['wrong_stop'] for v in wrong)}。")
    lines += [f"五折留出改善均值 {stop_result['oof_delta_mean']:.4f}，95% 区间 {stop_result['oof_delta_ci95']}。",
              f"训练排序候选为 {stop_result['candidate']} 米；"
              + (f"分组验证支持选用 {stop_result['selected']} 米。" if stop_result['accepted'] else
                 '分组验证未证明成功比例提升，因此保留 FMM 0.75 米，待完整测评验证。'), '',
              '## 验证边界', '',
              '- 分组单位是目标实例，重复运行和同目标的多个起点不能跨折。bootstrap 5000 次，seed 77。',
              '- 旧轨迹使用 match 和描述探索开启；新提示词与新探索方式的完整 SR 仍需另行测评。',
              '- 原点云随机状态未保存；节点使用原处理方法和固定离线种子重新生成，不声称逐值恢复历史节点。',
              '- 节点重建采用对齐后的内部类别和书籍尺寸规则，动作轨迹仍固定为旧采集轨迹。',
              '- 局部续跑只纳入能恢复控制计数的历史锁定，不能恢复的事件没有用猜测状态补齐。',
              '- 续跑观测即使在预定容差内，只要 RGB/深度未逐值一致，也排除在停止阈值选择之外。',
              '- 原始日志和图片保持不变；语义真值只用于离线评分，没有送入策略。',
              '- 节点 P/R/F1 是几何与类别代理指标；未知标签不计入精确率，真实同类干扰物不算伪节点。',
              '- 召回分母包括全部观察帧中至少 25 像素的非结构实例，并非仅限检测器输出的对象。',
              '- 确认正确/错误分组按历史框中占比最高的非结构语义实例判断；原始像素占比保存在逐分支记录中。',
              '- 成功按官方停止位置距离评分；候选身份判断错误也可能满足官方距离条件。',
              '- 书籍挽回数按每次执行的预测节点累加，不等同于独立书籍真值实例数。',
              '- 置信区间不支持改善时保留基准；未完成的汇总不会促成参数调整。', '',
              '细分指标、目标分组和逐类判断见同目录 summary.json；重放方法见 docs/coin_calibration.md。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path('outputs/coin_calibration_20261109'))
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({k: result[k] for k in ('complete', 'coverage', 'recommended')}, indent=2))
