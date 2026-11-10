from pathlib import Path
from unittest.mock import patch


def test_parallel_api_launcher_starts_only_navigation(tmp_path, monkeypatch):
    import run_parallel_with_server as runner
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND','openai')
    monkeypatch.setenv('PSGNAV_PYTHON','/test/python')
    job=runner.ParallelRunner([0],dataset='instancenav',total_scenes=35)
    job.log_dir=tmp_path
    with patch.object(runner.subprocess,'Popen') as popen, patch.object(runner,'get_model_client'), \
         patch.object(runner.time,'sleep') as sleep:
        result=job.start_job(0,0,35)
    assert popen.call_count==1
    assert 'PSG_Nav.py' in popen.call_args.args[0]
    assert 'server.py' not in popen.call_args.args[0]
    assert result[1] is None
    sleep.assert_not_called()
