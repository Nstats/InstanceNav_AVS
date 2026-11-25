"""Exercise the real launcher only in a temporary checkout, on ephemeral loopback ports."""
import argparse
from contextlib import contextmanager, ExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from tools import vlfm_servers as launcher


FAKE_CHECK = """\
import json, os, sys
from pathlib import Path
with Path('checks.jsonl').open('a') as log:
    print(json.dumps([sys.argv[1:], os.getcwd(), os.getenv('CUDA_VISIBLE_DEVICES'),
                      os.getenv('PYTHONPATH'), os.getenv('FAKE_PYTHON')]), file=log)
sys.exit(int(os.getenv('FAIL_CHECK', '') in ('any', 'navigation' if '--group' in sys.argv else 'full')))
"""
FAKE_SERVER = """\
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, os, sys, time
from pathlib import Path
model, flag, port = sys.argv[1:]
assert flag == '--port'
route = {'grounding_dino': '/gdino', 'sam': '/mobile_sam', 'blip2itm': '/blip2itm', 'yolov7': '/yolov7'}[model]
print('dummy', model, flush=True)
Path('ran-' + model).write_text(json.dumps([os.getpid(), os.getcwd(), os.getenv('CUDA_VISIBLE_DEVICES'),
                                            os.getenv('PYTHONPATH'), sys.executable]))
if model == os.getenv('FAIL_MODEL'):
    print('dummy load failed', flush=True)
    sys.exit(23)
if model == os.getenv('STALL_MODEL'):
    time.sleep(40)
    sys.exit(1)
class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200 if self.path == route else 404)
        if self.path == route:
            self.send_header('Allow', 'POST, OPTIONS')
            override = Path('pid-header-' + model)
            pid = override.read_text() if override.exists() else str(os.getpid())
            if pid:
                self.send_header('X-VLFM-Pid', pid)
        self.end_headers()
    def log_message(self, *args): pass
with HTTPServer(('127.0.0.1', int(port)), Handler) as server:
    server.timeout = 0.2
    deadline = time.monotonic() + 40  # never leak a persistent test service
    while time.monotonic() < deadline:
        server.handle_request()
"""


class Checkout:
    def __init__(self, tmp):
        self.root = tmp / 'project with spaces'
        self.script = self.root / 'scripts' / 'launch_vlfm_servers.sh'
        self.base = self.root / 'outputs' / 'vlfm_servers'
        for sub in ('scripts', 'tools', 'nav_bases'):
            (self.root / sub).mkdir(parents=True)
        shutil.copyfile(launcher.ROOT / 'scripts' / self.script.name, self.script)
        shutil.copyfile(launcher.__file__, self.root / 'tools' / 'vlfm_servers.py')
        (self.root / 'nav_bases' / '__init__.py').touch()
        (self.root / 'nav_bases' / 'vlfm_runtime.py').write_text(FAKE_CHECK)
        (self.root / 'nav_bases' / 'vlfm_server.py').write_text(FAKE_SERVER)
        self.env = dict(os.environ, PSGNAV_PYTHON=sys.executable, VLFM_PYTHON='', VLFM_SERVER_GPU='0',
                        CUDA_VISIBLE_DEVICES='99', PYTHONPATH=str(tmp / 'unrelated sibling'),
                        http_proxy='http://127.0.0.1:1', HTTP_PROXY='http://127.0.0.1:1',
                        ALL_PROXY='http://127.0.0.1:1', NO_PROXY='', no_proxy='',
                        FAIL_CHECK='', FAIL_MODEL='', STALL_MODEL='')
        # Reserve all four at once so releasing one cannot give its port to a second service.
        with ExitStack() as stack:
            for key, _, _ in launcher.SERVICES.values():
                sock = stack.enter_context(socket.socket())
                sock.bind(('127.0.0.1', 0))
                self.env[key] = str(sock.getsockname()[1])

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.script), *args], cwd=self.root.parent,
                              env={**self.env, **env}, capture_output=True, text=True, timeout=12)

    def checks(self):
        return [json.loads(line) for line in (self.root / 'checks.jsonl').read_text().splitlines()]

    def ports(self):
        return {m: int(self.env[key]) for m, (key, _, _) in launcher.SERVICES.items()}


@pytest.fixture
def checkout(tmp_path):
    project = Checkout(tmp_path)
    yield project
    for file in project.base.glob('*/state.json'):
        project.invoke('--stop', '--session', file.parent.name)


@contextmanager
def external(route, allow='OPTIONS, post'):
    """Simulate a legacy (or unrelated) server; it never belongs to the launcher."""
    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            matches = route in (self.path, '*')
            self.send_response(204 if matches else 404)
            if matches:
                self.send_header('Allow', allow)
            self.end_headers()

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize('choice', ['VLFM_PYTHON', 'PSGNAV_PYTHON', 'python'])
def test_check_only_interpreter_order_cwd_gpu_and_clean_env(checkout, choice):
    bindir = checkout.root.parent / 'bin'
    bindir.mkdir()
    for name in ('VLFM_PYTHON', 'PSGNAV_PYTHON', 'python'):
        interpreter = bindir / name
        interpreter.write_text(f'#!/bin/sh\nexport FAKE_PYTHON={name}\nexec {shlex.quote(sys.executable)} "$@"\n')
        interpreter.chmod(0o700)
    env = dict(VLFM_PYTHON=str(bindir / 'VLFM_PYTHON') if choice == 'VLFM_PYTHON' else '',
               PSGNAV_PYTHON=str(bindir / 'PSGNAV_PYTHON') if choice != 'python' else '',
               PATH=f'{bindir}:{os.environ["PATH"]}', VLFM_SERVER_GPU='3')
    result = checkout.invoke('--check-only', **env)
    assert result.returncode == 0, result.stderr
    assert checkout.checks() == [[['check'], str(checkout.root), '3', None, choice]]
    assert not checkout.base.exists()


def test_complete_owned_session_reuse_and_stop(checkout):
    launch = ('--session', 'mine', '--gpu', '7')
    result = checkout.invoke(*launch)
    assert result.returncode == 0, result.stderr
    path = checkout.base / 'mine' / 'state.json'
    state = json.loads(path.read_text())
    assert path.stat().st_mode & 0o077 == path.parent.stat().st_mode & 0o077 == 0
    assert set(state['pids']) == set(launcher.SERVICES)
    assert checkout.checks()[-1][0] == ['check']
    for model, pid in state['pids'].items():
        assert json.loads((checkout.root / ('ran-' + model)).read_text()) == [
            pid, str(checkout.root), '7', None, sys.executable]
        assert f'dummy {model}' in (path.parent / f'{model}.log').read_text()
    assert checkout.invoke(*launch, FAIL_CHECK='full').returncode == 0
    assert checkout.checks()[-1][0] == ['check', '--group', 'navigation']
    assert checkout.invoke(*launch, FAIL_CHECK='navigation').returncode != 0
    assert json.loads(path.read_text()) == state
    assert 'different GPU/ports' in checkout.invoke('--session', 'mine', '--gpu', '2').stderr
    assert 'different GPU/ports' in checkout.invoke(*launch, SAM_PORT='1').stderr
    # Another session cannot reuse even our exact routes, and cannot stop them.
    assert 'already in use' in checkout.invoke('--session', 'another').stderr
    assert checkout.invoke('--stop', '--session', 'another').returncode == 0
    assert all(launcher.probe(m, p, expected_pid=state['pids'][m]) for m, p in checkout.ports().items())
    result = checkout.invoke('--stop', '--session', 'mine', SAM_PORT='invalid',
                             VLFM_SERVER_GPU='invalid', FAIL_CHECK='any')
    assert result.returncode == 0, result.stderr
    assert not path.exists() and not any(launcher.probe(m, p) for m, p in checkout.ports().items())


@pytest.mark.parametrize('route,allow', [('/mobile_sam', 'OPTIONS, POST'), ('/sam', 'POST'),
                                         ('/mobile_sam', 'GET, OPTIONS'), ('*', 'POST')])
def test_occupied_port_refused_even_with_matching_external_route(checkout, route, allow):
    with external(route, allow) as port:
        result = checkout.invoke(SAM_PORT=str(port))
        assert result.returncode != 0
        assert f'127.0.0.1:{port} is already in use' in result.stderr and 'SAM_PORT' in result.stderr
        assert not list(checkout.root.glob('ran-*')) and not (checkout.root / 'checks.jsonl').exists()
        assert not list(checkout.base.glob('*/state.json'))
        assert checkout.invoke('--stop').returncode == 0
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            pass  # The external listener was left in place.


@pytest.mark.parametrize('failure', ['partial', 'wrong_token', 'missing_header', 'different_pid'])
def test_reuse_fails_closed_and_preserves_state(checkout, failure):
    assert checkout.invoke('--session', 'mine').returncode == 0
    path = checkout.base / 'mine' / 'state.json'
    original = path.read_text()
    header = checkout.root / 'pid-header-sam'
    try:
        state = json.loads(original)
        if failure == 'partial':
            del state['pids']['sam']
        elif failure == 'wrong_token':
            state['token'] = '0' * 32
        else:
            header.write_text('' if failure == 'missing_header' else str(os.getpid()))
        launcher.save_state(path, state)
        result = checkout.invoke('--session', 'mine')
        assert result.returncode != 0 and '--stop --session mine' in result.stderr
        assert ('incomplete/stale' if failure in ('partial', 'wrong_token') else 'X-VLFM-Pid') in result.stderr
        assert json.loads(path.read_text()) == state
        assert len(checkout.checks()) == 1
    finally:
        path.write_text(original)  # fixture stops all four original children


def test_failed_preflight_starts_nothing(checkout):
    result = checkout.invoke(FAIL_CHECK='full')
    assert result.returncode != 0 and 'asset preflight' in result.stderr
    assert not list(checkout.root.glob('ran-*')) and not list(checkout.base.glob('*/state.json'))


def test_child_failure_cleans_up_only_new_processes_and_keeps_logs(checkout):
    result = checkout.invoke(FAIL_MODEL='sam')
    assert result.returncode != 0 and 'sam exited with status 23' in result.stderr
    log = checkout.base / 'vlfm_servers' / 'sam.log'
    assert str(log) in result.stderr and 'dummy load failed' in log.read_text()
    assert not list(checkout.base.glob('*/state.json'))
    assert not any(launcher.probe(m, p) for m, p in checkout.ports().items())


@pytest.mark.parametrize('header', ['', '1'])
def test_startup_rejects_ready_route_without_expected_child_pid(checkout, header):
    (checkout.root / 'pid-header-sam').write_text(header)
    result = checkout.invoke()
    assert result.returncode != 0 and 'expected X-VLFM-Pid' in result.stderr
    assert not list(checkout.base.glob('*/state.json'))
    assert not any(launcher.probe(m, p) for m, p in checkout.ports().items())


def test_timeout_rolls_back(checkout, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', checkout.root)
    monkeypatch.setattr(launcher, 'START_TIMEOUT', 0.5)
    for key, value in checkout.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('STALL_MODEL', 'sam')
    args = argparse.Namespace(session='timeout', gpu='1', stop=False, check_only=False)
    with pytest.raises(launcher.LauncherError, match=r'Timed out waiting for .*sam'):
        launcher.run(args)
    assert not list(checkout.base.glob('*/state.json'))
    assert not any(launcher.probe(m, p) for m, p in checkout.ports().items())


def test_sigterm_during_startup_rolls_back(checkout):
    child = subprocess.Popen(['bash', str(checkout.script)], env=dict(checkout.env, STALL_MODEL='sam'),
                             cwd=checkout.root.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 6
        while not (checkout.root / 'ran-sam').exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (checkout.root / 'ran-sam').exists()
        competing = checkout.invoke('--session', 'different-name')
        assert competing.returncode != 0 and 'Another VLFM launcher' in competing.stderr
        child.send_signal(signal.SIGTERM)
        _, err = child.communicate(timeout=8)
        assert child.returncode != 0 and 'interrupted by SIGTERM' in err
        assert not list(checkout.base.glob('*/state.json'))
        assert not any(launcher.probe(m, p) for m, p in checkout.ports().items())
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=8)


@pytest.mark.parametrize('wrong', ['launch_token', 'command'])
def test_recycled_pid_is_not_signalled(checkout, wrong):
    # Even another launcher-style Python process on the expected port is not ours.
    launcher.private_dir(checkout.base)
    path = checkout.base / 'stale' / 'state.json'
    launcher.private_dir(path.parent)
    token = 'a' * 32
    ports = checkout.ports()
    cmd = [sys.executable, '-m', 'nav_bases.vlfm_server', 'sam', '--port', str(ports['sam'])]
    owner = f'{path}:{token}'
    if wrong == 'launch_token':
        owner = f'{path}:another-generation'
    else:
        cmd = [sys.executable, '-c', 'import time; time.sleep(30)']
    child = subprocess.Popen(cmd, cwd=checkout.root, env=dict(checkout.env, **{launcher.OWNER_ENV: owner}),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        launcher.save_state(path, dict(host=socket.gethostname(), gpu=0, ports=ports,
                                       token=token, pids={'sam': child.pid}))
        result = checkout.invoke('--stop', '--session', 'stale')
        assert result.returncode == 0 and child.poll() is None and not path.exists()
        assert f'Skipping sam PID {child.pid}' in result.stdout
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_real_flask_options_identifies_serving_process(monkeypatch):
    # Exercise Flask's automatic OPTIONS handler, not an inference POST or a real model.
    import importlib.util
    wrapper = launcher.ROOT / 'third_party/vlfm/vlfm/vlm/server_wrapper.py'
    spec = importlib.util.spec_from_file_location('vlfm_test_wrapper', wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    apps = []
    monkeypatch.setattr(module.Flask, 'run', lambda app, **kw: apps.append(app))
    module.host_model(None, 'mobile_sam')
    reply = apps[0].test_client().options('/mobile_sam')
    assert reply.status_code == 200 and 'POST' in reply.allow
    assert reply.headers['X-VLFM-Pid'] == str(os.getpid())


@pytest.mark.parametrize('override,launch_status', [(False, 0), (True, 0), (False, 1)])
def test_h20_queue_shares_ports_and_aborts_on_conflict(tmp_path, override, launch_status):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    name = 'run_continuous_evaluation_vlfm_base_h20.sh'
    shutil.copyfile(launcher.ROOT / 'scripts' / name, scripts / name)
    keys = [setting for setting, _, _ in launcher.SERVICES.values()]
    record = 'printf "%s\\n" ' + ' '.join('"$' + k + '"' for k in keys)
    (scripts / 'launch_vlfm_servers.sh').write_text(record + ' "$@" > launch\nexit "$RESULT"\n')
    (scripts / 'run_instance_ablation.sh').write_text(record + ' > evaluate\n')
    env = {k: v for k, v in os.environ.items() if k not in keys}
    ports = [str((23181 if override else 22181) + i) for i in range(4)]
    if override:
        env.update(zip(keys, ports))
    result = subprocess.run(['bash', str(scripts / name)], cwd=tmp_path, env=dict(env, RESULT=str(launch_status)))
    assert result.returncode == launch_status
    assert (tmp_path / 'launch').read_text().splitlines() == ports + ['--gpu', '0', '--session', 'instancenav_h20']
    if not launch_status:
        assert (tmp_path / 'evaluate').read_text().splitlines() == ports
    else:
        assert not (tmp_path / 'evaluate').exists()


def test_refuses_unsafe_session_and_duplicate_ports(checkout):
    assert checkout.invoke('--session', '../other', '--stop').returncode == 2
    result = checkout.invoke('--check-only', SAM_PORT=checkout.env['BLIP2ITM_PORT'])
    assert result.returncode != 0 and 'distinct ports' in result.stderr
    assert not checkout.base.exists()
