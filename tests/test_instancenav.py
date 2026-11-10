import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tools.prepare_instancenav import prepare
from utils.instance_verifier import verify_candidates, parse_matches
from utils.model_client import ModelRequestError


def verification_records(capsys):
    prefix = '[Instance Verification] '
    return [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)]


def test_real_dataset_subset_reproducible_and_instance_specific(tmp_path):
    attributes=Path('data/datasets/instancenav/val/val_text.json.gz')
    content=Path('data/datasets/instance_imagenav_hm3d_v3/val/content')
    if not attributes.exists() or not content.exists():
        pytest.skip('Public dataset not downloaded')
    a=prepare(attributes,content,tmp_path/'a.json.gz')
    b=prepare(attributes,content,tmp_path/'b.json.gz')
    assert (a['source_count'],a['selected_count'])==(1000,200)
    assert a['episodes']==b['episodes']
    assert a['prepared_sha256']==b['prepared_sha256']
    import habitat
    from utils.instancenav_dataset import InstanceNavDataset
    config=habitat.get_config('configs/instancenav_val20_qwen.yaml')
    assert config.ENVIRONMENT.MAX_EPISODE_STEPS==1000
    assert config.TASK.SUCCESS.SUCCESS_DISTANCE==1.0
    dataset=InstanceNavDataset(config.DATASET)
    source=json.load(gzip.open(tmp_path/'a.json.gz','rt'))
    assert len(dataset.episodes)==200
    for ep in dataset.episodes:
        assert len(ep.goals)==1
        assert str(ep.goals[0].object_id)==str(source['goals'][ep.goals_key]['object_id'])
        assert ep.goal_attributes


def test_missing_attributes_fail_instead_of_changing_subset_population(tmp_path):
    attrs=tmp_path/'attr.gz'
    with gzip.open(attrs,'wt') as f: json.dump({'attribute_data':{}},f)
    content=tmp_path/'content';content.mkdir()
    with gzip.open(content/'scene.json.gz','wt') as f:
        json.dump({'episodes':[{'episode_id':'0','scene_id':'scene.basis.glb','goal_object_id':'1'}]},f)
    with pytest.raises(ValueError,match='lack PSL descriptions'):
        prepare(attrs,content,tmp_path/'subset.gz')


def test_instance_verification_uses_description_and_never_oracle_path(capsys):
    prompts=[]
    def complete(prompt,image,**options):
        prompts.append(prompt)
        schema = options['response_format']['json_schema']['schema']
        assert schema['properties']['matches']['minItems'] == 3
        assert schema['properties']['matches']['maxItems'] == 3
        answer = '```json\n{"matches":["no","yes","unknown"]}\n```'
        options['validator'](answer)
        return answer
    client=SimpleNamespace(complete=complete)
    indices=verify_candidates(client,np.zeros((50,50,3),dtype=np.uint8),
                              [[1,1,10,10],[11,11,20,20],[21,21,30,30]],'chair',
                              {'intrinsic_attributes':'red striped chair','image':'secret_target_image.png'})
    assert indices==[1]
    assert 'red striped chair' in prompts[0]
    assert 'secret_target_image' not in prompts[0]
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['raw_response', 'result']
    assert records[0]['answer'] == '```json\n{"matches":["no","yes","unknown"]}\n```'
    assert records[1]['matches'] == ['no', 'yes', 'unknown']
    assert records[1]['kept_candidate_ids'] == [1]
    assert all(r['mode'] == 'instance_match' and r['category'] == 'chair' for r in records)
    assert 'secret_target_image' not in json.dumps(records)
    client.complete=lambda *args, **kwargs:'{"matches":["yes"]}'
    with pytest.raises(ModelRequestError):
        verify_candidates(client,np.zeros((50,50,3),dtype=np.uint8),[[1,1,5,5],[8,8,10,10]],
                          'chair',{'intrinsic_attributes':'red striped chair'})


def test_unseen_subset_has_no_training_scene_overlap():
    import zipfile
    archive=Path('data/downloads/instance_imagenav_hm3d_v3.zip')
    manifest=Path('data/datasets/instancenav/val20/val20.manifest.json')
    if not archive.exists() or not manifest.exists():pytest.skip('Public dataset not downloaded')
    selected={Path(s).name.split('.')[0] for s in json.loads(manifest.read_text())['scenes']}
    with zipfile.ZipFile(archive) as z:
        train={Path(n).name.removesuffix('.json.gz') for n in z.namelist() if '/train/content/' in n and n.endswith('.json.gz')}
        val={Path(n).name.removesuffix('.json.gz') for n in z.namelist() if '/val/content/' in n and n.endswith('.json.gz')}
    assert len(train)==145 and len(val)==36
    assert selected<=val
    assert selected.isdisjoint(train)


def test_instance_format_retry_preserves_evidence_and_avoids_cached_bad_reply(capsys):
    calls = []
    def complete(prompt, image, **options):
        calls.append((prompt, image.tobytes()))
        answer = '{"matches":["yes"]}' if len(calls) == 1 else '{"matches":["NO", " yes "]}'
        options['validator'](answer)
        return answer
    indices = verify_candidates(SimpleNamespace(complete=complete), np.zeros((20,20,3),dtype=np.uint8),
                                [[1,1,5,5],[8,8,12,12]], 'chair', {'intrinsic_attributes':'red chair'})
    assert indices == [1] and len(calls) == 2
    assert calls[0][0] != calls[1][0] and calls[0][1] == calls[1][1]
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['raw_response', 'format_error', 'raw_response', 'result']
    assert [r['attempt'] for r in records] == [1, 1, 2, 2]
    assert len({r['verification_id'] for r in records}) == 1
    assert records[0]['answer'] == '{"matches":["yes"]}'
    assert records[2]['answer'] == '{"matches":["NO", " yes "]}'
    assert records[3]['matches'] == ['no', 'yes']


def test_invalid_verification_saves_replay_and_never_returns_no(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('PSGNAV_API_DIAGNOSTICS_DIR', str(tmp_path))
    def complete(prompt, image, **options):
        options['validator']('{"matches":[true]}')
    with pytest.raises(ModelRequestError, match='3 format attempts'):
        verify_candidates(SimpleNamespace(complete=complete), np.zeros((20,20,3),dtype=np.uint8),
                          [[1,1,5,5]], 'chair', {'intrinsic_attributes':'red chair'})
    records = [json.loads(p.read_text()) for p in tmp_path.glob('instance_*.json')]
    assert len(records) == 3
    assert all(record['answer'] == '{"matches":[true]}' for record in records)
    assert all((tmp_path / record['image']).is_file() for record in records)
    assert all(record['error'] == 'ValueError: non-string decision' for record in records)
    logs = verification_records(capsys)
    assert [r['event'] for r in logs] == ['raw_response', 'format_error'] * 3 + ['failed']
    assert all(r['answer'] == '{"matches":[true]}' for r in logs if r['event'] == 'raw_response')
    assert logs[-1]['attempts'] == 3


def test_singleton_object_wrapper_preserves_decisions_and_order():
    assert parse_matches('[{"matches":["no","unknown","yes"]}]', 3) == ['no','unknown','yes']


@pytest.mark.parametrize('answer', [
    '[{"matches":["yes"]},{"matches":["no"]}]',
    '["yes","no"]',
    '[{"matches":["yes"]}]',
    '[{"matches":["yes","no","unknown"]}]',
    '[{"matches":[true,false]}]',
    '[{"matches":["yes","maybe"]}]',
    '[{"matches":["yes","no"],"alternative":["no","yes"]}]',
])
def test_ambiguous_or_invalid_wrappers_are_rejected(answer):
    with pytest.raises((ValueError, TypeError)):
        parse_matches(answer, 2)


def test_singleton_wrapper_does_not_spend_an_extra_api_call():
    calls = []
    def complete(prompt, image, **options):
        calls.append(prompt)
        answer = '[{"matches":["unknown","yes"]}]'
        options['validator'](answer)
        return answer
    indices = verify_candidates(SimpleNamespace(complete=complete), np.zeros((20,20,3),dtype=np.uint8),
                                [[1,1,5,5],[8,8,12,12]], 'chair', {'intrinsic_attributes':'red chair'})
    assert indices == [1] and len(calls) == 1


# The match prompt must stay byte-identical to the one used by all earlier runs: the exact
# API cache keys and the comparison with those runs depend on it.
MATCH_PROMPT_SHA256 = '185cc02577349fb2c1710d822fee3349efd8a2281dcd91f134ce75567f107ca6'


def _positive_match_prompt(monkeypatch, name):
    if name is None:
        monkeypatch.delenv('INSTANCE_VERIFICATION_PROMPT', raising=False)
    else:
        monkeypatch.setenv('INSTANCE_VERIFICATION_PROMPT', name)
    prompts = []

    def complete(prompt, image, **options):
        prompts.append(prompt)
        answer = '{"matches":["no","yes","unknown"]}'
        options['validator'](answer)
        return answer

    kept = verify_candidates(SimpleNamespace(complete=complete), np.zeros((50, 50, 3), dtype=np.uint8),
                             [[1, 1, 10, 10], [11, 11, 20, 20], [21, 21, 30, 30]], 'chair',
                             {'intrinsic_attributes': 'red striped chair', 'extrinsic_attributes': 'next to a desk'},
                             log_context={'stage': 'goal_confirmation'})
    assert kept == [1]  # only yes is kept under every prompt version
    return prompts[0]


def test_default_verification_prompt_is_the_unchanged_match_prompt(monkeypatch):
    import hashlib
    default = _positive_match_prompt(monkeypatch, None)
    assert default == _positive_match_prompt(monkeypatch, 'match')
    assert 'If evidence is inadequate, answer unknown.' in default
    assert hashlib.sha256(default.encode()).hexdigest() == MATCH_PROMPT_SHA256


def test_strict_no_prompt_changes_only_the_rule_sentence(monkeypatch):
    match = _positive_match_prompt(monkeypatch, 'match')
    strict = _positive_match_prompt(monkeypatch, 'MATCH_STRICT_NO')
    assert 'If evidence is inadequate' not in strict
    assert 'clearly contradicts the description' in strict and 'answer "unknown", not "no"' in strict
    old_rule = 'If evidence is inadequate, answer unknown. '
    head, tail = match.split(old_rule)
    assert strict.startswith(head) and strict.endswith(tail)  # yes wording, schema text and stage prefix shared


def test_unknown_verification_prompt_is_rejected(monkeypatch):
    from configs.navigation_config import instance_verification_prompt
    monkeypatch.setenv('INSTANCE_VERIFICATION_PROMPT', 'strict')
    with pytest.raises(ValueError, match='match_strict_no'):
        instance_verification_prompt()


def test_contradiction_mode_ignores_verification_prompt(monkeypatch):
    prompts = []

    def complete(prompt, image, **options):
        prompts.append(prompt)
        answer = '{"matches":["no"]}'
        options['validator'](answer)
        return answer

    for name in ('match', 'match_strict_no'):
        monkeypatch.setenv('INSTANCE_VERIFICATION_PROMPT', name)
        verify_candidates(SimpleNamespace(complete=complete), np.zeros((20, 20, 3), dtype=np.uint8),
                          [[1, 1, 5, 5]], 'chair', {'intrinsic_attributes': 'red chair'}, conflict=True)
    assert prompts[0] == prompts[1]
