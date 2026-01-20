#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-shot runner for Ollama + Gemma (Gemma/Gemma2/Gemma3).

What it does (Linux):
1) Check if ollama exists; if not, install via official install.sh
2) Ensure ollama service is running (systemd if available; else start server)
3) Pull the model (ollama pull)
4) Warm up once (optional)
5) Call Ollama HTTP API /api/generate and print output

Usage:
  python3 run_gemma_ollama.py
Optional env:
  MODEL=gemma2:9b PROMPT="Hello" HOST=http://127.0.0.1:11434 python3 run_gemma_ollama.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error


DEFAULT_MODEL = os.environ.get("MODEL", "gemma2:9b")
DEFAULT_PROMPT = os.environ.get("PROMPT", "请用三点总结注意力机制的核心思想。")
OLLAMA_HOST = os.environ.get("HOST", "http://127.0.0.1:11434")
INSTALL_SH_URL = "https://ollama.com/install.sh"


def run(cmd, check=True, capture=False, sudo=False):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    if capture:
        return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return subprocess.run(cmd, check=check)


def which_ollama():
    return shutil.which("ollama")


def is_linux():
    return platform.system().lower() == "linux"


def has_systemd():
    if not is_linux():
        return False
    return shutil.which("systemctl") is not None and os.path.isdir("/run/systemd/system")


def http_post_json(url, payload, timeout=30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_ollama_ready(host, timeout_sec=30):
    # Ollama usually exposes a root page / or health-ish endpoint.
    # We'll probe /api/tags since it's stable for checking readiness.
    deadline = time.time() + timeout_sec
    url = host.rstrip("/") + "/api/tags"
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Ollama is not ready at {url}. Last error: {last_err}")


def install_ollama_linux():
    print("[1/5] Ollama not found. Installing on Linux via install.sh ...")

    try:
        # Download install.sh to a temp file
        script_path = "/tmp/ollama_install.sh"
        with urllib.request.urlopen(INSTALL_SH_URL, timeout=30) as resp:
            content = resp.read()
        with open(script_path, "wb") as f:
            f.write(content)

        os.chmod(script_path, 0o755)

        # Run install script (needs root)
        run(["bash", script_path], sudo=True)
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Failed to download install.sh. If you previously saw HTTP/2 PROTOCOL_ERROR, "
            "try re-running on a more stable network, or download the script with curl --http1.1.\n"
            f"Error: {e}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Install script failed: {e}")


def ensure_ollama_running():
    print("[2/5] Ensuring Ollama service/server is running ...")

    if has_systemd():
        # Start and enable service
        try:
            run(["systemctl", "enable", "--now", "ollama"], sudo=True)
        except subprocess.CalledProcessError:
            # Some environments might already have it running or have a different setup
            pass
    else:
        # No systemd: run 'ollama serve' in background (best-effort).
        # Note: this is not a perfect daemonization method but works for typical shells/containers.
        if os.environ.get("NO_SERVE") == "1":
            print("NO_SERVE=1 set; skipping 'ollama serve'. Assuming Ollama is already running.")
        else:
            # If already running, serve will fail quickly; we ignore failure.
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception:
                pass

    wait_ollama_ready(OLLAMA_HOST, timeout_sec=30)
    print(f"    Ollama is reachable at {OLLAMA_HOST}")


def pull_model(model):
    print(f"[3/5] Pulling model: {model} ...")
    # This will download if missing, and is idempotent.
    res = run(["ollama", "pull", model], capture=True)
    print(res.stdout.strip())


def warmup(model):
    # Optional warmup: small prompt to trigger first-time load
    print("[4/5] Warming up model (first load may take a while) ...")
    try:
        res = run(["ollama", "run", model, "ping"], capture=True, check=False)
        # Don't fail the whole run if warmup fails; API call will still be attempted.
        if res.stdout:
            print("    Warmup output:", res.stdout.strip()[:200])
    except Exception:
        pass


def generate(model, prompt):
    print("[5/5] Generating via HTTP API ...")
    url = OLLAMA_HOST.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    out = http_post_json(url, payload, timeout=120)
    obj = json.loads(out)
    return obj.get("response", "")


def main():
    model = DEFAULT_MODEL
    prompt = DEFAULT_PROMPT

    # 0) Basic platform guidance
    if not is_linux():
        # On Windows/macOS, require user to install Ollama app first
        if which_ollama() is None:
            print("Ollama CLI not found. On Windows/macOS, please install Ollama first, then re-run this script.")
            sys.exit(2)

    # 1) Ensure ollama installed
    if which_ollama() is None:
        if is_linux():
            install_ollama_linux()
        else:
            print("Ollama not found and auto-install is only implemented for Linux in this script.")
            sys.exit(2)

    # 2) Ensure server running
    ensure_ollama_running()

    # 3) Pull model
    pull_model(model)

    # 4) Warmup
    warmup(model)

    # 5) Generate
    resp = generate(model, prompt)
    print("\n===== MODEL RESPONSE =====\n")
    print(resp.strip())
    print("\n==========================\n")


if __name__ == "__main__":
    main()
