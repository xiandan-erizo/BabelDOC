import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


def test_worker_command_runs_the_worker_module():
    from web.server.main import build_worker_command

    assert build_worker_command("job-123") == [
        sys.executable,
        "-m",
        "web.server.worker",
        "job-123",
    ]


def test_launch_translate_worker_starts_detached_process(monkeypatch):
    from web.server import main

    calls = []

    class FakeProcess:
        pid = 4242

    def fake_popen(command, cwd, start_new_session):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "start_new_session": start_new_session,
            }
        )
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    process = main.launch_translate_worker("job-123")

    assert process.pid == 4242
    assert calls == [
        {
            "command": [
                sys.executable,
                "-m",
                "web.server.worker",
                "job-123",
            ],
            "cwd": main.PROJECT_DIR,
            "start_new_session": True,
        }
    ]
