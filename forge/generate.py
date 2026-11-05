"""Generates a run: every shot in a run file, at every seed, for every model. Finished images are
skipped, so rerunning the same command resumes. Submit it through the GPU queue:

    gpu submit --vram 30G --name sprite-derek -- .venv/bin/python -m forge.generate characters/derek/runs/keys.toml [model ...]
"""
import hashlib, json, os, shutil, signal, subprocess, sys, time, tomllib, urllib.error, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .pixelize import black_alpha
from .workflows import MODELS

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FORGE_DATA", "/mnt/data/sprite-forge"))
API = "http://127.0.0.1:8188"
KEY = (255, 0, 255)


def rev():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return {"sha": sha or None, "dirty": dirty, "comfy": (ROOT / "comfy/COMFY_REF").read_text().strip()}


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def get(path):
    return json.load(urllib.request.urlopen(API + path, timeout=30))


def wait_up(proc, secs=300):
    for _ in range(secs):
        if proc.poll() is not None:
            sys.exit(f"ComfyUI exited with {proc.returncode}")
        try:
            return get("/system_stats")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    sys.exit("ComfyUI did not come up")


def prepare_ref(path, box):
    """Places a reference on the key colour at the working size. Copies are named by content."""
    img = Image.open(ROOT / path).convert("RGB")
    a = np.asarray(img)
    if a[0, 0].max() < 14:
        keep = black_alpha(a)
        a = np.where(keep[..., None], a, np.array(KEY, np.uint8))
        img = Image.fromarray(a.astype(np.uint8))
    img = img.resize(box, Image.LANCZOS)
    name = hashlib.sha256(img.tobytes()).hexdigest()[:16] + ".png"
    (DATA / "in").mkdir(parents=True, exist_ok=True)
    img.save(DATA / "in" / name)
    return name


def publish(src, out):
    """Moves a finished image into place only once it is durable, so `out` existing means done."""
    tmp = out.with_suffix(".tmp")
    shutil.move(src, tmp)
    Image.open(tmp).verify()
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, out)


def run_prompt(graph, out):
    pid = post("/prompt", {"prompt": graph})["prompt_id"]
    while True:
        h = get(f"/history/{pid}").get(pid)
        if h:
            if h["status"].get("status_str") == "error":
                raise RuntimeError(json.dumps(h["status"].get("messages"))[-1500:])
            f = h["outputs"]["save"]["images"][0]
            publish(DATA / "out" / f["subfolder"] / f["filename"], out)
            return
        time.sleep(0.5)


def main():
    spec_path = Path(sys.argv[1])
    spec = tomllib.loads(spec_path.read_text())
    char = tomllib.loads((spec_path.parent.parent / "character.toml").read_text())
    run_dir = DATA / "runs" / spec_path.parent.parent.name / spec_path.stem
    w, h = spec["width"], spec["height"]
    todo = []
    for model in sys.argv[2:] or spec["models"]:
        for shot, s in spec["shots"].items():
            for seed in spec["seeds"]:
                out = run_dir / model / shot / f"{seed}.png"
                if not out.exists():
                    todo.append((model, shot, s, seed, out))
    print(f"{len(todo)} images to make", flush=True)
    if not todo:
        return
    proc = subprocess.Popen([ROOT / "comfy/run.sh", *spec.get("comfy_args", [])], stdout=open(DATA / "comfy.log", "a"), stderr=subprocess.STDOUT)
    stop = lambda *_: (subprocess.run(["docker", "stop", "-t", "5", "sprite-forge-comfy"]), sys.exit(143))
    signal.signal(signal.SIGTERM, stop)
    try:
        print(wait_up(proc)["devices"], flush=True)
        meta = rev()
        for n, (model, shot, s, seed, out) in enumerate(todo, 1):
            out.parent.mkdir(parents=True, exist_ok=True)
            refs = [prepare_ref(r, (w, h)) for r in s.get("refs", spec.get("refs", []))]
            prompt = spec["template"].format(description=char["description"], action=s["prompt"])
            t = time.time()
            graph = MODELS[model](prompt, refs, seed, w, h, "forge")
            run_prompt(graph, out)
            secs = round(time.time() - t, 1)
            out.with_suffix(".json").write_text(json.dumps({"model": model, "shot": shot, "seed": seed, "prompt": prompt, "refs": s.get("refs", spec.get("refs", [])), "size": [w, h], "secs": secs, "rev": meta}, indent=1))
            print(f"chunk {n}/{len(todo)} {model} {shot} {seed} {secs}s", flush=True)
    finally:
        subprocess.run(["docker", "stop", "-t", "5", "sprite-forge-comfy"], capture_output=True)


if __name__ == "__main__":
    main()
