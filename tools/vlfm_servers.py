"""Start the local VLFM model services or stop only the children of a named session."""
import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
# Module argument, port setting, and actual upstream Flask route (not always the module name).
SERVICES = {
    'grounding_dino': ('GROUNDING_DINO_PORT', 12181, '/gdino'),
    'blip2itm': ('BLIP2ITM_PORT', 12182, '/blip2itm'),
    'sam': ('SAM_PORT', 12183, '/mobile_sam'),
    'yolov7': ('YOLOV7_PORT', 12184, '/yolov7'),
}
MODULE = 'nav_bases.vlfm_server'
OWNER_ENV = 'VLFM_SERVER_OWNER'
START_TIMEOUT = 300


class LauncherError(RuntimeError):
    pass


def configured_ports():
    ports = {}
    for model, (key, default, _) in SERVICES.items():
        value = os.environ.get(key) or str(default)
        if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
            raise LauncherError(f'{key} must be a TCP port (1–65535), got {value!r}')
        ports[model] = int(value)
    if len(set(ports.values())) != len(ports):
        raise LauncherError(f'The four VLFM models need distinct ports: {ports}')
    return ports


def _options(port, route):
    # HTTPConnection connects directly: no environment proxy, redirect, or inference POST.
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
    try:
        conn.request('OPTIONS', route)
        response = conn.getresponse()
        allow = response.getheader('Allow', '')
        return response.status, allow, (200 <= response.status < 300 and
                                        'POST' in {m.strip().upper() for m in allow.split(',')})
    finally:
        conn.close()


def probe(model, port):
    """True for the model endpoint; False for no listener; refuse an unidentified listener."""
    key, _, route = SERVICES[model]
    try:
        status, allow, matches = _options(port, route)
        if matches:
            # A catch-all HTTP/CORS handler advertising POST for every path is not evidence.
            if not _options(port, '/__vlfm_unknown_route__')[2]:
                return True
            reason = 'also advertises POST on an unknown route'
        else:
            reason = f'HTTP {status}, Allow={allow!r}'
    except ConnectionRefusedError:
        return False
    except (OSError, http.client.HTTPException) as exc:
        reason = str(exc)
    raise LauncherError(f'Cannot identify {model} on 127.0.0.1:{port}: OPTIONS {route} ({reason}). '
                        f'Check the listener or choose another {key}; it will not be stopped.')


def preflight(env, *, navigation_only=False):
    cmd = [sys.executable, '-m', 'nav_bases.vlfm_runtime', 'check']
    if navigation_only:
        cmd += ['--group', 'navigation']
    if subprocess.run(cmd, cwd=ROOT, env=env).returncode:
        raise LauncherError('VLFM asset preflight failed; fix the files reported above and retry.')


def private_dir(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise LauncherError(f'{path} must be a private directory (mode 700) owned by this user.')


def save_state(path, state):
    # NamedTemporaryFile is mode 600; replacement keeps the last complete PID list on disk.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as tmp:
        try:
            json.dump(state, tmp)
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def load_state(path):
    try:
        state = json.loads(path.read_text())
        ports, pids = state['ports'], state['pids']
        valid = (isinstance(state['host'], str) and type(state['gpu']) is int and
                 re.fullmatch('[a-f0-9]{32}', state['token']) and set(ports) == set(SERVICES) and
                 all(type(p) is int and 1 <= p <= 65535 for p in ports.values()) and
                 isinstance(pids, dict) and pids and set(pids) <= set(SERVICES) and
                 all(type(p) is int and p > 1 for p in pids.values()))
        if not valid:
            raise ValueError('invalid session fields')
    except FileNotFoundError:
        return None
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise LauncherError(f'Invalid VLFM session state {path}; inspect it manually. '
                            'No processes were signalled.') from exc
    if state['host'] != socket.gethostname():
        raise LauncherError(f'{path} belongs to host {state["host"]}; use that host or another --session.')
    return state


def _args(model, port):
    return ['-m', MODULE, model, '--port', str(port)]


def _signal_fd(fd, sig):
    """Signal the process pinned by this /proc/PID fd, never a recycled numeric PID."""
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    # Conda Pythons built with older Linux headers omit the Python binding even on
    # recent GPU hosts. Linux x86-64 and arm64 both use syscall 424 (since 5.1).
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, 'pidfd_send_signal'):
        result = libc.pidfd_send_signal(fd, sig, None, 0)
    elif os.uname().machine in ('x86_64', 'aarch64') and ctypes.sizeof(ctypes.c_void_p) == 8:
        result = libc.syscall(424, fd, sig, None, 0)
    else:
        raise LauncherError('Safe VLFM shutdown needs a Linux Python with signal.pidfd_send_signal.')
    if result == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def owned_signal(path, state, model, sig=0):
    """Both cmd and private launch token must match, using the same handle as the signal."""
    try:
        fd = os.open(f'/proc/{state["pids"][model]}', os.O_DIRECTORY)
        try:
            if os.fstat(fd).st_uid != os.getuid():
                return False
            proc = Path(f'/proc/self/fd/{fd}')
            argv = (proc / 'cmdline').read_bytes().split(b'\0')[1:]
            expected = [os.fsencode(arg) for arg in _args(model, state['ports'][model])] + [b'']
            tag = os.fsencode(f'{OWNER_ENV}={path}:{state["token"]}')
            if argv != expected or tag not in (proc / 'environ').read_bytes().split(b'\0'):
                return False
            if sig:
                _signal_fd(fd, sig)
            return True
        finally:
            os.close(fd)
    except (FileNotFoundError, ProcessLookupError):
        return False


def terminate(models, send):
    """Grace period shared by all children. send returns False for a gone/unowned PID."""
    live = list(models)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        live = [m for m in live if send(m, sig)]
        deadline = time.monotonic() + 5
        while live and time.monotonic() < deadline:
            live = [m for m in live if send(m, 0)]
            if live:
                time.sleep(0.1)
        if not live:
            return
    raise LauncherError(f'Could not stop {", ".join(live)}; session state retained. Retry --stop.')


def start(path, gpu, ports, missing, env):
    # Refuse to create detached processes on platforms where we could not stop them safely.
    try:
        fd = os.open('/proc/self', os.O_DIRECTORY)
        try:
            _signal_fd(fd, 0)
        finally:
            os.close(fd)
    except OSError as exc:
        raise LauncherError(f'Safe VLFM shutdown requires Linux procfs and pidfd signalling: {exc}') from exc

    state = dict(host=socket.gethostname(), gpu=gpu, ports=ports, token=secrets.token_hex(16), pids={})
    env = dict(env, PYTHONUNBUFFERED='1', **{OWNER_ENV: f'{path}:{state["token"]}'})
    children = {}
    # Defer cancellation to a checkpoint so a signal can't land between Popen and PID bookkeeping.
    interrupted = []
    old = {sig: signal.signal(sig, lambda s, f: interrupted.append(s)) for sig in (signal.SIGTERM, signal.SIGINT)}

    def check_children():
        if interrupted:
            raise LauncherError(f'VLFM startup interrupted by {signal.Signals(interrupted[0]).name}.')
        for model, child in children.items():
            if child.poll() is not None:
                raise LauncherError(f'{model} exited with status {child.returncode} on port {ports[model]}; '
                                    f'inspect {path.parent / (model + ".log")}')

    def signal_child(model, sig):
        child = children[model]  # direct, unreaped child; never a PID from a file
        if child.poll() is not None:
            return False
        if sig:
            child.send_signal(sig)
        return True

    try:
        for model in missing:
            check_children()
            with (path.parent / f'{model}.log').open('w') as log:
                children[model] = subprocess.Popen([sys.executable, *_args(model, ports[model])], cwd=ROOT,
                                                   env=env, start_new_session=True, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT)
            state['pids'][model] = children[model].pid
            save_state(path, state)
        deadline = time.monotonic() + START_TIMEOUT
        waiting = missing
        while waiting:
            check_children()
            waiting = [m for m in waiting if not probe(m, ports[m])]
            if waiting and time.monotonic() >= deadline:
                raise LauncherError(f'Timed out waiting for {", ".join(waiting)}; inspect logs in {path.parent}')
            if waiting:
                time.sleep(0.3)
        # Reconfirm once, rather than repeatedly probing a busy legacy server during model loading.
        for model, port in ports.items():
            if not probe(model, port):
                raise LauncherError(f'{model} disappeared from port {port} during startup; inspect {path.parent}')
        check_children()
    except BaseException:
        print(f'Cleaning up newly started VLFM processes; logs: {path.parent}', file=sys.stderr)
        terminate(children, signal_child)
        path.unlink(missing_ok=True)
        raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def run(args):
    base = ROOT / 'outputs' / 'vlfm_servers'
    path = base / args.session / 'state.json'
    if not args.stop:
        if not args.gpu.isascii() or not args.gpu.isdecimal():
            raise LauncherError('--gpu (or VLFM_SERVER_GPU) must be a non-negative physical GPU index.')
        gpu = int(args.gpu)
        ports = configured_ports()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        env.pop('PYTHONPATH', None)
        if args.check_only:
            preflight(env)
            return
    if args.stop and not base.exists():
        print(f'No owned session {args.session}.')
        return
    private_dir(base)
    # Serialize even different session names: two launches must not race for the same ports.
    with (base / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LauncherError('Another VLFM launcher is starting/stopping services; retry when it finishes.') from exc
        if args.stop and not path.parent.exists():
            print(f'No owned session {args.session}.')
            return
        private_dir(path.parent)
        state = load_state(path)
        if args.stop:
            if state is None:
                print(f'No owned session {args.session}; external servers were left alone.')
            else:
                def stop_owned(model, sig):
                    matched = owned_signal(path, state, model, sig)
                    if sig == signal.SIGTERM and not matched:
                        print(f'Skipping {model} PID {state["pids"][model]} (gone or no longer owned).')
                    return matched

                terminate(state['pids'], stop_owned)
                path.unlink()
                print(f'Cleared session {args.session}; unowned processes were left alone.')
            return
        stop_hint = f'--stop --session {args.session}'
        if state and (gpu != state['gpu'] or ports != state['ports']):
            raise LauncherError(f'Session {args.session} has different GPU/ports; use {stop_hint} '
                                'first or choose another session and ports.')
        missing = [m for m, port in ports.items() if not probe(m, port)]
        if state and (missing or not all(owned_signal(path, state, m) for m in state['pids'])):
            raise LauncherError(f'Session {args.session} is incomplete/stale; inspect {path.parent} and '
                                f'use {stop_hint} before restarting.')
        preflight(env, navigation_only=not missing)
        if missing:
            print(f'Starting {", ".join(missing)} on physical GPU {gpu}; logs: {path.parent}', flush=True)
            start(path, gpu, ports, missing, env)
        print(f'VLFM endpoints ready: {ports}. Only this session\'s own children can be stopped.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default=os.environ.get('VLFM_SERVER_GPU') or '0', help='physical GPU index')
    parser.add_argument('--session', default='vlfm_servers', help='name of private launcher state')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--stop', action='store_true', help='stop only the verified children of this session')
    action.add_argument('--check-only', action='store_true', help='check model/navigation assets without starting')
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}', args.session):
        parser.error('--session must be a simple name (letters, digits, dots, underscores or dashes)')
    try:
        run(args)
    except (LauncherError, OSError) as exc:
        parser.exit(1, f'VLFM: {exc}\n')


if __name__ == '__main__':
    main()
