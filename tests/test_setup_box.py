"""Exercise the actual Bash installer using a local Python/pip simulator.

The metadata probes execute with real importlib.metadata and temporary Debian-
style dist-info. No pip download, model import, GPU call or system install occurs.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "phase0" / "setup_box.sh"


def bash_executable():
    if os.name == "nt":
        # Windows' bash.exe is commonly the WSL dispatcher, not an installed
        # shell. Prefer Git Bash, which can run without a Linux distro.
        git = shutil.which("git")
        candidates = ([Path(git).resolve().parents[1] / "bin" / "bash.exe"] if git else [])
        candidates += [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        pytest.skip("Git Bash unavailable; shell test cannot use the WSL dispatcher")
    value = shutil.which("bash")
    if value is None:
        pytest.skip("Bash unavailable")
    return value


SIMULATOR = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ["SETUP_FIXTURE"])
with (root / "calls.jsonl").open("a", encoding="utf-8") as out:
    out.write(json.dumps({"args": args, "interpreter": os.environ.get("SETUP_INTERPRETER", "base")}) + "\n")
if args[:2] == ["-m", "pip"]:
    if "rich" in args:
        if os.environ.get("FAIL_OVERLAY") == "1":
            sys.exit(7)
        assert "--ignore-installed" in args and "--no-deps" in args
        assert args[-1] == "rich"
        dist = root / ("unselected-target/rich-13.7.1.dist-info" if os.environ.get("UNSELECTED_OVERLAY") == "1"
                       else "metadata/rich-13.7.1.dist-info")
        dist.mkdir(parents=True, exist_ok=True)
        if not (dist / "METADATA").exists():
            (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: rich\nVersion: 13.7.1\n")
        (dist / "RECORD").write_text("pip-managed-overlay\n")
    elif "huggingface_hub" in args and os.environ.get("SETUP_MODE") == "system":
        dist = root / "metadata/rich-13.7.1.dist-info"
        if dist.exists() and not (dist / "RECORD").exists():
            print("Cannot uninstall Debian rich: RECORD missing", file=sys.stderr)
            sys.exit(8)
    sys.exit(0)
if args[:2] == ["-m", "venv"]:
    destination = pathlib.Path(args[2])
    (destination / "bin").mkdir(parents=True)
    (destination / "pyvenv.cfg").write_text("home = fixture\n")
    wrapper = pathlib.Path(os.environ["FAKE_PYTHON"]).read_text()
    # Both interpreters share the simulator, but the isolated one's context is
    # observable in every pip/import probe.
    wrapper = wrapper.replace("exec ", "SETUP_INTERPRETER=venv exec ", 1)
    target = destination / "bin/python"
    target.write_text(wrapper); target.chmod(0o755)
    sys.exit(0)
if args == ["-"]:
    source = sys.stdin.read()
    if "import torch, transformers, kernels" in source:
        print("mock installed-version report")
        sys.exit(0)
    sys.base_prefix = "fixture-base"
    sys.prefix = ("fixture-venv" if os.environ.get("SETUP_MODE") == "venv" or
                  os.environ.get("SETUP_INTERPRETER") == "venv" else "fixture-base")
    sys.path.insert(0, str(root / "metadata"))
    exec(compile(source, "setup-probe", "exec"))
    sys.exit(0)
raise AssertionError("unexpected simulated Python invocation")
'''


@pytest.fixture
def installer(tmp_path):
    shell = bash_executable()
    simulator = tmp_path / "simulate_python.py"
    simulator.write_text(SIMULATOR, encoding="utf-8")
    wrapper = tmp_path / "fixture python"
    wrapper.write_text(f'#!/bin/bash\nexec "{Path(sys.executable).as_posix()}" "{simulator.as_posix()}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    env = {**os.environ, "SHARD_SETUP_PYTHON": wrapper.as_posix(), "SETUP_FIXTURE": str(tmp_path),
           "FAKE_PYTHON": str(wrapper), "SETUP_MODE": "system"}
    env.pop("SHARD_SETUP_VENV", None)
    env.pop("SETUP_INTERPRETER", None)
    def install(*args, mode="system", rich="record", fail_overlay=False, overlay_selected=True):
        metadata = tmp_path / "metadata/rich-13.7.1.dist-info"
        if rich != "absent" and not metadata.exists():
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: rich\nVersion: 13.7.1\n")
            (metadata / "INSTALLER").write_text("debian\n" if rich == "debian" else "pip\n")
            if rich == "record":
                (metadata / "RECORD").write_text("rich/__init__.py,,\n")
        result = subprocess.run([shell, SCRIPT.as_posix(), *map(str, args)], env={**env, "SETUP_MODE": mode,
                                "FAIL_OVERLAY": "1" if fail_overlay else "0",
                                "UNSELECTED_OVERLAY": "0" if overlay_selected else "1"}, text=True,
                                capture_output=True, timeout=30)
        log = tmp_path / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls
    return install


def pip_calls(calls):
    return [row for row in calls if row["args"][:2] == ["-m", "pip"]]


def test_debian_rich_without_record_gets_only_targeted_overlay_before_dependencies(installer):
    result, calls = installer(rich="debian")
    assert result.returncode == 0, result.stderr
    installs = pip_calls(calls)
    assert len(installs) == 3
    overlay = installs[0]["args"]
    assert overlay[-3:] == ["--ignore-installed", "--no-deps", "rich"]
    assert "--break-system-packages" in overlay
    assert installs[1]["args"][-1] == "torch==2.11.0"
    assert "huggingface_hub" in installs[2]["args"]
    assert all("--ignore-installed" not in row["args"] for row in installs[1:])


@pytest.mark.parametrize("rich", ["record", "absent"])
def test_normal_system_install_does_not_bypass_uninstallation(installer, rich):
    result, calls = installer(rich=rich)
    assert result.returncode == 0, result.stderr
    installs = pip_calls(calls)
    assert len(installs) == 2
    assert all("--break-system-packages" in row["args"] and "--ignore-installed" not in row["args"] for row in installs)


def test_active_venv_uses_same_interpreter_without_system_flags(installer):
    result, calls = installer(mode="venv", rich="debian")
    assert result.returncode == 0, result.stderr
    assert len(pip_calls(calls)) == 2
    assert all("--break-system-packages" not in row["args"] and "--ignore-installed" not in row["args"] for row in pip_calls(calls))


def test_explicit_venv_with_spaces_is_isolated_created_once_and_reported(installer, tmp_path):
    directory = tmp_path / "isolated runtime"
    result, calls = installer("--venv", directory.as_posix())
    assert result.returncode == 0, result.stderr
    assert sum(row["args"][:2] == ["-m", "venv"] for row in calls) == 1
    assert all(row["interpreter"] == "venv" for row in pip_calls(calls))
    assert all("--break-system-packages" not in row["args"] for row in pip_calls(calls))
    assert directory.as_posix() + "/bin/python" in result.stdout
    again, calls = installer("--venv", directory.as_posix())
    assert again.returncode == 0, again.stderr
    assert sum(row["args"][:2] == ["-m", "venv"] for row in calls) == 1


def test_overlay_failure_aborts_before_installing_torch_or_other_dependencies(installer):
    result, calls = installer(rich="debian", fail_overlay=True)
    assert result.returncode != 0
    assert len(pip_calls(calls)) == 1
    assert "rich" in pip_calls(calls)[0]["args"]


def test_successful_overlay_outside_metadata_path_fails_before_other_installs(installer, tmp_path):
    result, calls = installer(rich="debian", overlay_selected=False)
    assert result.returncode == 1
    assert "overlay is not selected" in result.stderr and "--venv DIRECTORY" in result.stderr
    assert len(pip_calls(calls)) == 1
    assert (tmp_path / "unselected-target/rich-13.7.1.dist-info/RECORD").is_file()
    assert not (tmp_path / "metadata/rich-13.7.1.dist-info/RECORD").exists()


def test_repeated_setup_skips_overlay_once_pip_record_exists(installer):
    result, _ = installer(rich="debian")
    assert result.returncode == 0, result.stderr
    again, calls = installer(rich="debian")
    assert again.returncode == 0, again.stderr
    assert sum("--ignore-installed" in row["args"] for row in pip_calls(calls)) == 1


def test_explicit_venv_refuses_nonvenv_directory_without_installs(installer, tmp_path):
    directory = tmp_path / "existing unrelated"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("untouched")
    result, calls = installer("--venv", directory.as_posix())
    assert result.returncode == 2
    assert calls == []
    assert marker.read_text() == "untouched"


def test_config_file_alone_cannot_pretend_a_system_interpreter_is_isolated(installer, tmp_path):
    directory = tmp_path / "broken venv"
    (directory / "bin").mkdir(parents=True)
    (directory / "pyvenv.cfg").write_text("home = stale\n")
    target = directory / "bin/python"
    target.write_text((tmp_path / "fixture python").read_text())
    target.chmod(0o755)
    result, calls = installer("--venv", directory.as_posix())
    assert result.returncode == 1
    assert "not isolated" in result.stderr
    assert not pip_calls(calls)


@pytest.mark.parametrize("args", [("--venv",), ("--unknown",), ("--venv", "--help")])
def test_invalid_options_fail_without_touching_python(installer, args):
    result, calls = installer(*args)
    assert result.returncode == 2
    assert calls == []


def test_help_does_not_install_or_probe(installer):
    result, calls = installer("--help")
    assert result.returncode == 0
    assert "--venv DIRECTORY" in result.stdout
    assert calls == []
