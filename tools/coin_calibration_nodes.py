"""Replay GroundedSAM and real 3D object merging once for all five count gates."""
from collections import Counter, OrderedDict
import copy
import hashlib
import os
from pathlib import Path
import numpy as np

from configs.dataset_categories import dataset_to_unified_name

STRUCTURAL = {'', 'unknown', 'unlabeled', 'wall', 'floor', 'ceiling', 'objects', 'misc'}


def sg_name(raw):
    canonical = dataset_to_unified_name(raw, 'coin')
    return {'chest_of_drawers': 'drawers', 'tv_monitor': 'tv'}.get(canonical, canonical)


def caption_matches(caption, annotation):
    annotation = {'couch': 'sofa', 'potted plant': 'plant', 'chest of drawers': 'drawers',
                  'tv monitor': 'tv'}.get(annotation, annotation)
    caption, annotation = sg_name(caption), sg_name(annotation)
    return caption == annotation or annotation in caption.split() or caption in annotation.split()


class NodeReplay:
    def __init__(self, config, geometry, cache_dir=None):
        import torch
        from graph.graph_class import SceneGraph
        from utils.inference_cache import cache_eval_module, cache_sam_images, install_sam_position_cache
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.graph = SceneGraph(5, 4000, 800, geometry.rgb_camera_matrix(), dataset='coin')
        self.base_node_space = self.graph.node_space
        dino, sam = self.graph.mask_generator
        cache_eval_module(dino.bert)
        cache_sam_images(sam)
        install_sam_position_cache()
        self.cache = OrderedDict()
        cache_dir = Path(cache_dir) if cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        # Perception only: the cache contains no navigation decisions or GT.
        sources = ['graph/meta_func/seg_func.py', 'utils/utils_sg_uq/sam_utils.py',
                   'data/models/groundingdino_swint_ogc.pth', 'data/models/sam_vit_h_4b8939.pth']
        version = hashlib.sha256()
        for path in sources:
            p = Path(path)
            version.update((str(p) + str(p.stat().st_size) + str(p.stat().st_mtime_ns)).encode())
        cache_version = version.hexdigest()
        original = self.graph.get_sam_segmentation_dense

        def cached(variant, model, image):
            key = (self.graph.node_space, hashlib.sha256(image.tobytes()).digest())
            if key not in self.cache:
                disk_key = hashlib.sha256(cache_version.encode() + key[0].encode() + key[1]).hexdigest()
                path = cache_dir / (disk_key + '.npz') if cache_dir else None
                if path and path.exists():
                    with np.load(path, allow_pickle=False) as data:
                        if not int(data['present']):
                            value = (None, None, None, None)
                        else:
                            shape = tuple(data['shape'])
                            masks = np.unpackbits(data['bits'], count=int(np.prod(shape))).reshape(shape).astype(bool)
                            value = (masks, data['boxes'].copy(), data['conf'].copy(), data['captions'].tolist())
                else:
                    value = original(variant, model, image)
                    if path:
                        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
                        payload = {'present': np.array(int(value[0] is not None))}
                        if value[0] is not None:
                            payload.update(shape=value[0].shape, bits=np.packbits(value[0]),
                                           boxes=value[1], conf=value[2], captions=np.asarray(value[3]))
                        with tmp.open('wb') as stream:
                            np.savez_compressed(stream, **payload)
                        tmp.replace(path)
                self.cache[key] = value
                while len(self.cache) > 24:
                    self.cache.popitem(last=False)
            return copy.deepcopy(self.cache[key])
        self.graph.get_sam_segmentation_dense = cached
        # Attach evaluation labels to each actual 3D detection before merging.
        # image_idx can repeat when production mapping reuses the last masks;
        # indexing GT by image_idx would incorrectly relabel those observations.
        import graph.meta_func.core_func as core
        original_detections = core.gobs_to_detection_list

        def labelled_detections(*args, **kwargs):
            foreground, background = original_detections(*args, **kwargs)
            gobs = kwargs['gobs']
            for obj in [*foreground, *background]:
                j = obj['mask_idx'][-1]
                ids, counts = np.unique(self.current_semantic[gobs['mask'][j]], return_counts=True)
                label = None
                if len(ids):
                    dominant = int(ids[np.argmax(counts)])
                    name = self.renderer.info.get(dominant, {}).get('name', '').lower().strip()
                    if name not in {'', 'unknown', 'unlabeled', 'misc'}:
                        label = dominant if (name not in STRUCTURAL and max(counts) / sum(counts) >= .5 and
                                             caption_matches(gobs['caption'][j], name)) else -1
                obj['offline_gt_instances'] = [label]
                # A reused segmentation index can create a new spatial object
                # at a later pose. Identify actual mapping detections, not just
                # the source segmentation, so these nodes are not deduplicated.
                obj['offline_detection_origins'] = [(self.current_step, j)]
            return foreground, background
        core.gobs_to_detection_list = labelled_detections

    def begin(self, item, ep, renderer):
        self.item, self.ep, self.renderer = item, ep, renderer
        # Production point-cloud jitter and overlap subsampling use NumPy RNG.
        # A per-attempt seed makes offline reconstruction independent of worker
        # assignment and resume order; it does not change the production policy.
        seed_key = f"{item['run']}:{ep.scene_id}:{ep.episode_id}"
        self.replay_seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
        np.random.seed(self.replay_seed)
        self.graph.reset()
        self.graph.node_space = self.base_node_space
        for raw in [*item['preceding_goals'], ep.object_category]:
            self.graph.set_obj_goal(dataset_to_unified_name(raw, 'coin'), sg_name(raw))
        self.visible, self.labels = set(), {}
        self.eligible = {k: {} for k in range(1, 6)}
        self.target_first = {k: None for k in range(1, 6)}
        self.book_rescued = {k: set() for k in range(1, 6)}
        self.history = []

    def frame(self, step, obs, atomic, sim):
        import torch
        graph = self.graph
        self.current_step = step['policy_step']
        self.current_semantic = np.asarray(obs['semantic']).squeeze()
        graph.set_observations(obs)
        with torch.no_grad():
            graph.segment2d()
            if graph.segment2d_results:
                graph.mapping3d()
        # The recall denominator must include visible objects even when the
        # detector produced no masks. Conditioning visibility on a detection
        # would hide misses from the offline score.
        ids, counts = np.unique(self.current_semantic, return_counts=True)
        for gid, count in zip(ids, counts):
            name = self.renderer.info.get(int(gid), {}).get('name', '').lower().strip()
            if count >= 25 and name not in STRUCTURAL:
                self.visible.add(int(gid))
        current = []
        for obj in graph.objects:
            key = tuple(obj['offline_detection_origins'][0])
            counts = Counter(obj['offline_gt_instances'])
            known = Counter({g: n for g, n in counts.items() if g is not None})
            identity = None
            if known:
                identity, votes = known.most_common(1)[0]
                if votes < sum(known.values()) / 2:
                    identity = -1  # merging several real instances is also a bad node
            points = np.asarray(obj['pcd'].points)
            if not len(points):
                continue
            extent = float(np.ptp(points, axis=0).max())
            captions = obj.get('detection_captions', [])
            caption = Counter(captions).most_common(1)[0][0] if captions else ''
            exempt = caption == 'book'
            size_pass = extent > .4 or exempt
            detections = int(obj['num_detections'])
            current.append({'origin': key, 'detections': detections, 'extent_m': extent,
                            'caption': caption, 'gt_instance': identity, 'size_pass': size_pass})
            if not size_pass or len(points) < graph.cfg.obj_min_points:
                continue
            for k in range(1, 6):
                if detections < k:
                    continue
                # Label at first eligibility; later views cannot rewrite this decision.
                self.eligible[k].setdefault(key, identity)
                if exempt and extent <= .4:
                    self.book_rescued[k].add(key)
                if (identity == int(self.ep.goal_object_id) and
                    sg_name(caption) == sg_name(self.ep.object_category) and self.target_first[k] is None):
                    self.target_first[k] = atomic
        self.history.append({'step': step['policy_step'], 'atomic_before': atomic, 'objects': current})
        if step['policy_step'] % 100 == 0:
            print(f"NODE_PROGRESS {self.item['run']} {self.ep.episode_id} {step['policy_step']}/482", flush=True)

    def finish(self):
        metrics = {}
        for k, objects in self.eligible.items():
            values = list(objects.values())
            correct = [g for g in values if g is not None and g >= 0]
            false = values.count(-1)
            precision = len(correct) / max(1, len(correct) + false)
            recall = len(set(correct) & self.visible) / max(1, len(self.visible))
            metrics[str(k)] = {'precision': precision, 'recall': recall,
                'f1': 2 * precision * recall / max(1e-12, precision + recall),
                'true_nodes': len(correct), 'false_nodes': false, 'unknown_nodes': values.count(None),
                'visible_instances': len(self.visible), 'target_first_atomic_step': self.target_first[k],
                'book_nodes_rescued': len(self.book_rescued[k])}
        return {'node_metrics': metrics, 'node_history': self.history,
                'node_space': self.graph.node_space, 'node_schema': 6,
                'replay_seed': self.replay_seed,
                'gt_label_rule': 'semantic-majority-0.5 AND caption-compatible; structural=false; unlabeled=unknown'}
