#!/usr/bin/env python3
"""Thin bootstrap for the Boltz-2 server image.

The image only contains Python + this file + the repo source. On start it:
  1. answers HTTP on $PORT immediately (/health -> 503 until ready, /bootstrap -> status),
  2. in parallel installs PyTorch (CUDA wheel matched to the node's driver) + boltz
     into a venv, and downloads the Boltz-2 weights / molecule data,
  3. stops its own listener and starts server.py on the same port,
  4. optionally exits after BOLTZ_MAX_LIFETIME_SECONDS so a one-hour session
     ends (and stops billing) by itself when restart_policy is "never".
Only the standard library is used here.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("PORT", "8080"))
VENV = Path(os.environ.get("BOLTZ_VENV", "/opt/venv"))
CACHE = Path(os.environ.get("BOLTZ_CACHE", "/opt/boltz-cache"))
MAX_LIFETIME = int(os.environ.get("BOLTZ_MAX_LIFETIME_SECONDS", "0"))  # 0 = unlimited
APP = Path(__file__).resolve().parent

# Keep in sync with src/boltz/main.py
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
CKPTS = {
    "boltz2_conf.ckpt": [
        "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
    ],
    "boltz2_aff.ckpt": [
        "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
    ],
}

START = time.time()
STATE = {"phase": "starting", "torch": "pending", "weights": "pending", "error": None, "cuda_wheel": None}
LOCK = threading.Lock()


def log(msg):
    print(f"[launcher {time.time() - START:6.0f}s] {msg}", flush=True)


def set_state(**kw):
    with LOCK:
        STATE.update(kw)


class V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6  # dual-stack on Linux: IPv6 gateway and IPv4 both work


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with LOCK:
            snap = dict(STATE, elapsed_s=int(time.time() - START))
        if self.path.startswith("/bootstrap"):
            self._send(200, snap)
        else:  # /health and everything else: not ready yet
            self._send(503, {"status": "starting", **snap})

    do_POST = do_GET

    def log_message(self, *a):
        pass


def cuda_wheel():
    """Pick a PyTorch CUDA wheel index from the driver's max supported CUDA version."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            v = (int(m[1]), int(m[2]))
            if v >= (12, 8):
                return "cu128"
            if v >= (12, 6):
                return "cu126"
            return "cu124"
    except Exception as e:  # noqa: BLE001
        log(f"nvidia-smi failed: {e}")
    return "cu128"


def run(cmd, **kw):
    log("$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def install_python():
    set_state(torch="installing")
    wheel = cuda_wheel()
    set_state(cuda_wheel=wheel)
    if not (VENV / "bin" / "python").exists():
        run([sys.executable, "-m", "venv", str(VENV)])
    pip = [str(VENV / "bin" / "python"), "-m", "pip", "install", "--no-cache-dir",
           "--default-timeout=300", "--retries", "10"]
    run(pip + ["torch", "--index-url", f"https://download.pytorch.org/whl/{wheel}"])
    run(pip + [f"{APP}[server]"])
    set_state(torch="done")


def fetch(urls, dest, tries=3):
    dest = Path(dest)
    last = None
    for url in urls:
        for attempt in range(tries):
            try:
                log(f"download {url} (try {attempt + 1})")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
                    shutil.copyfileobj(r, f, 1 << 20)
                if tmp.stat().st_size == 0:
                    raise RuntimeError("empty download")
                tmp.rename(dest)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                log(f"  failed: {e}")
    raise RuntimeError(f"all sources failed for {dest.name}: {last}")


def download_weights():
    set_state(weights="downloading")
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, urls in CKPTS.items():
        if not (CACHE / name).exists():
            fetch(urls, CACHE / name)
    if not (CACHE / "mols").exists():
        tar = CACHE / "mols.tar"
        fetch([MOL_URL], tar)
        with tarfile.open(tar) as t:
            t.extractall(CACHE)  # noqa: S202
        tar.unlink()
    set_state(weights="done")


def bootstrap():
    errors = []

    def guard(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
            log(f"ERROR {fn.__name__}: {e}")

    threads = [threading.Thread(target=guard, args=(f,)) for f in (install_python, download_weights)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    if errors:
        set_state(phase="failed", error="; ".join(errors))
        return False
    set_state(phase="ready")
    return True


def main():
    httpd = V6Server(("::", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log(f"listening on [::]:{PORT}; installing…")
    set_state(phase="installing")

    if not bootstrap():
        log("bootstrap failed; keeping /bootstrap up for 120 s then exiting")
        time.sleep(120)
        sys.exit(1)

    httpd.shutdown()
    httpd.server_close()
    env = dict(os.environ, BOLTZ_CACHE=str(CACHE), PATH=f"{VENV}/bin:" + os.environ["PATH"])
    log("starting server.py")
    proc = subprocess.Popen([str(VENV / "bin" / "python"), "server.py", "--host", "::", "--port", str(PORT)],
                            cwd=APP, env=env)
    try:
        if MAX_LIFETIME:
            remaining = max(MAX_LIFETIME - (time.time() - START), 0)
            log(f"session limit: exiting in {int(remaining)} s")
            try:
                sys.exit(proc.wait(timeout=remaining))
            except subprocess.TimeoutExpired:
                log("lifetime reached; stopping server")
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                sys.exit(0)
        sys.exit(proc.wait())
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
