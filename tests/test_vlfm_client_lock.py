"""Linux client locks only: no real model servers, ports, or GPUs are used."""
import fcntl
import multiprocessing
from types import SimpleNamespace

import pytest

from third_party.vlfm.vlfm.vlm import server_wrapper as client


URL = 'http://localhost:12183/mobile_sam'
OTHER = 'http://localhost:12182/blip2itm'
OK = SimpleNamespace(status_code=200, json=lambda: {'ok': True})


def test_uncontended_calls_do_not_sleep_and_reuse_inode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(client.requests, 'post', lambda *a, **kw: OK)
    monkeypatch.setattr(client.time, 'sleep', lambda _: pytest.fail('uncontended client slept'))
    assert client.send_request(URL) == {'ok': True}
    lock, = (tmp_path / 'lockfiles').glob('*.flock')
    inode = lock.stat().st_ino
    assert client.send_request(URL) == {'ok': True}
    assert lock.stat().st_ino == inode


def test_error_releases_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(client.requests, 'post', lambda *a, **kw: SimpleNamespace(status_code=503))
    with pytest.raises(Exception, match='Request failed'):
        client._send_request(URL)
    lock, = (tmp_path / 'lockfiles').glob('*.flock')
    with lock.open('a') as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _request(url, pipe, hold):
    # Instrument only this child. Tell the parent when we actually reach flock
    # (rather than inferring contention from how fast a process starts).
    acquire = fcntl.flock

    def lock(fd, flags):
        pipe.send('locking')
        return acquire(fd, flags)

    def post(*args, **kwargs):
        pipe.send('posted')
        if hold:
            assert pipe.recv() == 'release'
        return OK

    client.fcntl.flock = lock
    client.requests.post = post
    assert client._send_request(url) == {'ok': True}
    pipe.close()


@pytest.mark.parametrize('kill_holder', [False, True])
def test_same_endpoint_waits_but_other_endpoint_proceeds(tmp_path, monkeypatch, kill_holder):
    monkeypatch.chdir(tmp_path)
    ctx = multiprocessing.get_context('fork')
    children = []

    def start(url, hold=False):
        rx, tx = ctx.Pipe()
        proc = ctx.Process(target=_request, args=(url, tx, hold))
        proc.start(); tx.close()
        children.append((proc, rx))
        assert rx.poll(5) and rx.recv() == 'locking'
        return proc, rx

    try:
        holder, first = start(URL, hold=True)
        assert first.poll(5) and first.recv() == 'posted'
        waiter, second = start(URL)
        independent, other = start(OTHER)
        assert other.poll(5) and other.recv() == 'posted'
        independent.join(5)
        assert independent.exitcode == 0
        assert not second.poll(0.05)  # Same URL cannot POST as long as first holds it.
        if kill_holder:
            holder.kill()
        else:
            first.send('release')
        holder.join(5)
        assert holder.exitcode is not None
        assert second.poll(5) and second.recv() == 'posted'
        waiter.join(5)
        assert waiter.exitcode == 0
    finally:
        for proc, pipe in children:
            if proc.is_alive():
                proc.kill()
            proc.join(5)
            pipe.close()
