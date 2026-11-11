"""Pose-guided frames for any character and motion, then game-ready sprites and a review sheet.

    python -m forge.poseguide gen    CHARACTER MOTION TRACK [--seed N] [--frames 0,1,2]   (GPU: submit through the queue)
    python -m forge.poseguide sprite CHARACTER MOTION TRACK [--seed N]                   (CPU: key, palette, outline, sheet)

TRACK names a recipe in `TRACKS`: a list of stages, each a ComfyUI workflow builder plus the images it reads
(the character master, the pose render, the previous stage's output) and a prompt. Everything comes from
characters/CHARACTER/{character.toml,pose.toml,references/master.png} and forge/motions/MOTION.toml.

Outputs: DATA/pose-options/CHARACTER/MOTION/TRACK/{pose,gen}... ; the sprite step runs the same keyout,
palette snap and outline as forge.compose, placing each frame by the pose canvas (fixed transform, no per-frame
re-centring), so frames are comparable with game art.
"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tomllib, time
from pathlib import Path

import numpy as np
from PIL import Image

from .color import load_palette
from .compose import keyout, thicken
from .generate import DATA, ROOT, KEY, get, post, publish, rev, wait_up
from . import generate as G
from .pixelize import despeckle, to_sprite, upscale
from . import lineart
from .posekit import Figure
from .workflows import MODELS

PORT = int(os.environ.get("FORGE_PORT", "8288"))
CONTAINER = os.environ.get("FORGE_CONTAINER", "sprite-forge-comfy-pose")
G.API = f"http://127.0.0.1:{PORT}"
SIZE = 1024
BG = "flat solid magenta background"
COMMON = "Exactly two arms and two legs, side view facing right, full body visible from head to feet, no text, letters or logos."

# stage: name, workflow, refs (names: master, pose:MODE, prev), prompt template ({desc})
TRACKS = {
    # Track 1: character master + pose render in one Qwen edit
    "qwen-pose": dict(pose="openpose", stages=[dict(model="qwen-edit", refs=["master", "pose"], prompt=(
        "Image 1 is the character. Image 2 is a pose skeleton for a side view. Redraw the exact same character from image 1, same colours, "
        "proportions, thick dark outline and chunky pixel-art game style, in the pose of the skeleton in image 2: put the head, arms and legs "
        "exactly where the skeleton's limbs go, both legs from the same hip, with the crossing and bending the skeleton shows. " + COMMON + " " + BG + "."))]),
    # human bodies: the skeleton reads as a body, so say what the limbs do
    "qwen-pose-run": dict(pose="openpose", stages=[dict(model="qwen-edit", refs=["master", "pose"], prompt=(
        "Image 1 is the character. Image 2 is a pose skeleton for a side view. Redraw the exact same character from image 1, same clothes, colours, "
        "proportions, thick dark outline and chunky pixel-art game style, {verb} in the pose of the skeleton in image 2: head, arms and legs exactly where the skeleton's "
        "limbs go. {hands} Both legs from the same hip, crossing and bending as the skeleton shows. {note} "
        + COMMON + " " + BG + ", no ground shadow."))]),
    "qwen-mannequin": dict(pose="mannequin", stages=[dict(model="qwen-edit", refs=["master", "pose"], prompt=(
        "Image 1 is the character. Image 2 shows black limb shapes: the arms and legs of the pose wanted. Redraw the exact same character from image 1, "
        "same colours, proportions, thick dark outline and chunky pixel-art game style, with its arms and legs exactly on the black limb shapes of "
        "image 2 and its body above the legs. " + COMMON + " " + BG + "."))]),
    # pose image first: the first reference is the one Qwen follows hardest
    "qwen-pose-first": dict(pose="mannequin_body", stages=[dict(model="qwen-edit", refs=["pose", "master"], prompt=(
        "Image 1 is a pose drawing: a black figure with its arms and legs in a running pose. Image 2 is the character. Redraw image 1 as the character from image 2: "
        "keep every arm and leg of image 1 exactly where it is, with the same bends, the crossing legs and the lifted foot, and give them the character's own "
        "sneakers, gloves, colours, thick dark outline and chunky pixel-art game style. The body sits on top of the legs where image 1 shows it. "
        + COMMON + " " + BG + "."))]),
    "outline-paint2": dict(pose="lines", stages=[
        dict(model="qwen-edit", refs=["pose", "master"], name="lineart", prompt=(
            "Image 1 is a rough outline drawing of a running pose. Redraw image 1 as a clean black-and-white line drawing of the character from image 2: "
            "keep every limb of image 1 where it is with the same bends and crossing, but with the character's body, face, gloves and sneakers drawn "
            "in the thick black outline of a coloring book. Plain white background, no colour, no shading. " + COMMON)),
        dict(model="qwen-edit", refs=["prev", "master"], prompt=(
            "Image 1 is a black line drawing of a character. Image 2 is the finished character. Colour image 1 to match image 2: keep every line of "
            "image 1 exactly where it is, same pose, same legs and arms, and paint it in the colours, shading and chunky pixel-art style of image 2, "
            "with a thick dark outline. " + COMMON + " " + BG + "."))]),
    # Track 3: Qwen-Image-InstantX union ControlNet (pose) on Qwen-Image-Edit, character from the master
    "qwen-cn-pose": dict(pose="openpose", stages=[dict(model="qwen-edit", refs=["master"], control=True, kw=dict(cn_strength=0.8), prompt=(
        "Redraw this exact same character, same colours, proportions, thick dark outline and chunky pixel-art game style, running in the pose given. "
        + COMMON + " " + BG + "."))]),
    "qwen-cn-pose-hi": dict(pose="openpose", stages=[dict(model="qwen-edit", refs=["master"], control=True, kw=dict(cn_strength=1.2), prompt=(
        "Redraw this exact same character, same colours, proportions, thick dark outline and chunky pixel-art game style, running in the pose given. "
        + COMMON + " " + BG + "."))]),
    "qwen-cn-pose-max": dict(pose="openpose", stages=[dict(model="qwen-edit", refs=["master"], control=True, kw=dict(cn_strength=2.0), prompt=(
        "Redraw this exact same character, same colours, proportions, thick dark outline and chunky pixel-art game style, running in the pose given. "
        + COMMON + " " + BG + "."))]),
    # ControlNet draws the pose from the character's description alone (no master to anchor the stance), then an edit
    # stage re-skins that figure as the exact character, keeping its limbs
    "cn-then-edit": dict(pose="openpose", stages=[
        dict(model="qwen-edit", refs=[], control=True, name="body", kw=dict(cn_strength=1.0), prompt=(
            "{desc}. Side view facing right, running, in the pose given. Chunky pixel-art game style with a thick dark outline. " + COMMON + " " + BG + ".")),
        dict(model="qwen-edit", refs=["prev", "master"], prompt=(
            "Image 1 is a character in a running pose. Image 2 is the exact character it must be. Redraw image 1 as the character from image 2, keeping every arm and leg "
            "of image 1 exactly where it is, with the same bends and crossing, and the same body position, but with the face, colours, gloves, sneakers and style of image 2. "
            + COMMON + " " + BG + "."))]),
    # as cn-then-edit, but the re-skin gets no master (which drags the stance back to the master's): the description carries identity
    "cn-restyle": dict(pose="openpose", stages=[
        dict(model="qwen-edit", refs=[], control=True, name="body", kw=dict(cn_strength=1.0), prompt=(
            "{desc}. Side view facing right, running, in the pose given. Chunky pixel-art game style with a thick dark outline. " + COMMON + " " + BG + ".")),
        dict(model="qwen-edit", refs=["prev"], prompt=(
            "Redraw this running cartoon character in exactly the same pose, keeping every arm and leg where it is, as: {desc}. Painted in the style of a glossy "
            "pixel-art game sprite with a thick dark outline, rich saturated colours, shaded cheese. " + COMMON + " " + BG + "."))]),
    # a code rig (forge/runrig.py) fixed the pose; deterministic edge lines carry it, a model only paints
    "rig-paint": dict(pose="rigline", stages=[dict(model="qwen-edit", refs=["pose", "master"], prompt=(
        "Image 1 is a black line drawing of a character. Image 2 is the finished character. Colour image 1 to match image 2: keep every line of "
        "image 1 exactly where it is, same pose, same legs and arms, and paint it in the colours, shading and chunky pixel-art style of image 2, "
        "with a thick dark outline. {note} " + COMMON + " " + BG + ", no ground shadow."))]),
    "qwen-cn-lines": dict(pose="lines", stages=[dict(model="qwen-edit", refs=["master"], control=True, kw=dict(cn_strength=0.9), prompt=(
        "Redraw this exact same character, same colours, proportions, thick dark outline and chunky pixel-art game style, following the line drawing for the pose. "
        + COMMON + " " + BG + "."))]),
    # FLUX.2 Klein takes reference images instead of an IP-Adapter: master + pose as two references
    "klein-pose": dict(pose="openpose", stages=[dict(model="klein", refs=["master", "pose"], prompt=(
        "Redraw the exact same cartoon character from the first image, same colours, proportions, thick dark outline and chunky pixel-art game style, "
        "in the pose of the skeleton in the second image: head, arms and legs exactly where the skeleton's limbs are. " + COMMON + " " + BG + "."))]),
    # Track 2: line drawing first, then paint it
    "outline-paint": dict(pose="lines", stages=[
        dict(model="qwen-edit", refs=["master", "pose"], name="lineart", prompt=(
            "Image 2 is a rough outline of a side-view pose. Draw the character from image 1 as a clean black-and-white line drawing in exactly that "
            "pose: black outlines only on a plain white background, no colour, no shading, same proportions and features as image 1, limbs exactly where "
            "image 2 puts them. " + COMMON)),
        dict(model="qwen-edit", refs=["prev", "master"], prompt=(
            "Image 1 is a black line drawing of a character. Image 2 is the finished character. Colour image 1 to match image 2: keep every line of "
            "image 1 exactly where it is, same pose, same legs and arms, and paint it in the colours, shading and chunky pixel-art style of image 2, "
            "with a thick dark outline. " + COMMON + " " + BG + ".")),
    ]),
}


def pose_cfg(char):
    return tomllib.loads((ROOT / "characters" / char / "pose.toml").read_text())


def master_path(char):
    return ROOT / "characters" / char / pose_cfg(char).get("master", "references/master.png")


def stage_name(track, i):
    return TRACKS[track]["stages"][i].get("name", f"s{i}")


TAG = ""


def paths(char, motion, track):
    return DATA / "pose-options" / char / motion / (track + TAG)


def to_input(img):
    """Save a PIL image into ComfyUI's input dir, named by content; returns the name."""
    img = img.convert("RGB")
    name = hashlib.sha256(img.tobytes()).hexdigest()[:16] + ".png"
    (DATA / "in").mkdir(parents=True, exist_ok=True)
    img.save(DATA / "in" / name)
    return name


def master_input(char):
    img = Image.open(master_path(char)).convert("RGB")
    return to_input(img.resize((SIZE, SIZE), Image.LANCZOS))


def rig_source(char, anim):
    """Frames of a finished rig animation under characters/CHAR/sprites/ANIM: (list of RGBA arrays, atlas)."""
    d = ROOT / "characters" / char / "sprites" / anim
    atlas = json.loads((d / "atlas.json").read_text())
    sheet = np.asarray(Image.open(d / "sheet.png").convert("RGBA"))
    w = atlas["frame"]["w"]
    return [sheet[:, i * w:(i + 1) * w] for i in range(atlas["count"])], atlas


def gen(args):
    char, motion, track = args.char, args.motion, args.track
    tr = TRACKS[track]
    rigline = tr["pose"] == "rigline"
    if rigline:
        rframes, ratlas = rig_source(char, args.src)
        fig = type("F", (), {"n": len(rframes), "motion": {"verb": "running", "frame_ms": ratlas["frame_ms"]}})()
    else:
        fig = Figure(char, motion)
    desc = tomllib.loads((ROOT / "characters" / char / "character.toml").read_text())["description"]
    pc = pose_cfg(char)
    fill = dict(verb=fig.motion.get("verb", "running"), hands=pc.get("hands_" + args.hands, pc.get("hands", "Both arms swing.")) if args.hands != "free" else pc.get("hands", "Both arms swing."),
                note=" ".join(x for x in (pc.get("note", ""), args.note) if x))
    out = paths(char, motion, track)
    out.mkdir(parents=True, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",")] if args.frames else list(range(fig.n))
    for t in frames:   # the pose images, saved for the review sheet
        if rigline:
            lineart.render(rframes[t], SIZE, *lineart.layout(rframes, SIZE)).save(out / f"pose_{t}.png")
        else:
            getattr(fig, tr["pose"])(t, SIZE).save(out / f"pose_{t}.png")
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
    todo = [(t, i, sd) for sd in seeds for t in frames for i in range(len(tr["stages"])) if not (out / f"{stage_name(track, i)}_{t}_{sd}.png").exists()]
    print(f"{len(todo)} images to make", flush=True)
    if not todo:
        return
    env = dict(os.environ, FORGE_PORT=str(PORT), FORGE_CONTAINER=CONTAINER)
    proc = subprocess.Popen([ROOT / "comfy/run.sh", "--reserve-vram", "1.5"], env=env, stdout=open(DATA / "comfy-pose.log", "a"), stderr=subprocess.STDOUT)

    def stop(*_):
        subprocess.run(["docker", "stop", "-t", "5", CONTAINER], capture_output=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, stop)
    try:
        wait_up(proc)
        mname = master_input(char)
        meta = rev()
        for n, (t, i, sd) in enumerate(todo, 1):
            st = tr["stages"][i]
            dest = out / f"{stage_name(track, i)}_{t}_{sd}.png"
            refs = []
            for r in st["refs"]:
                if r == "master":
                    refs.append(mname)
                elif r == "pose":
                    refs.append(to_input(Image.open(out / f"pose_{t}.png")))
                else:  # previous stage output
                    refs.append(to_input(Image.open(out / f"{stage_name(track, i - 1)}_{t}_{sd}.png")))
            t0 = time.time()
            kw = dict(st.get("kw", {}))
            if st.get("control"):
                kw["control"] = to_input(Image.open(out / f"pose_{t}.png"))
            prompt = st["prompt"].replace("{desc}", desc.rstrip("."))
            for k, v in fill.items():
                prompt = prompt.replace("{" + k + "}", v)
            graph = MODELS[st["model"]](prompt, refs, sd, SIZE, SIZE, "pg", **kw)
            G.run_prompt(graph, dest)
            secs = round(time.time() - t0, 1)
            dest.with_suffix(".json").write_text(json.dumps({"track": track, "stage": i, "frame": t, "seed": sd, "prompt": prompt, "secs": secs, "rev": meta}, indent=1))
            print(f"chunk {n}/{len(todo)} {track} f{t} s{i} {secs}s", flush=True)
    finally:
        subprocess.run(["docker", "stop", "-t", "5", CONTAINER], capture_output=True)


def deshadow(rgb, alpha):
    """Drop the soft ground shadow some generations add under the feet: a flat, wide, bottom-lying blob that is a darker
    version of the key background (same hue, similar chroma, lower lightness). Black outlines have no chroma and stay."""
    from scipy import ndimage
    from .color import srgb_to_oklab
    lab = srgb_to_oklab(rgb)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, 0)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    bh = np.degrees(np.arctan2(bg[2], bg[1]))
    dh = np.abs((hue - bh + 180) % 360 - 180)
    cand = alpha & (dh < 22) & (chroma > 0.5 * np.hypot(bg[1], bg[2])) & (lab[..., 0] < bg[0] - 0.04)
    cand = ndimage.binary_opening(cand, iterations=2)
    lb, n = ndimage.label(cand)
    ys_all = np.nonzero(alpha)[0]
    top, bot = ys_all.min(), ys_all.max()
    out = alpha.copy()
    for i, sl in enumerate(ndimage.find_objects(lb), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = (lb[sl] == i).sum()
        if area > 150 and w > 2.2 * h and sl[0].start > top + 0.7 * (bot - top):
            out &= ~(lb == i)
    return out


def sprite(args):
    char, motion, track = args.char, args.motion, args.track
    cfg = tomllib.loads((ROOT / "characters" / char / "character.toml").read_text())
    pose = pose_cfg(char)
    pal = load_palette(ROOT / "characters" / char / "palette.json")
    rigline = TRACKS[track]["pose"] == "rigline"
    d = paths(char, motion, track)
    if rigline:
        rframes, ratlas = rig_source(char, args.src)
        fig = type("F", (), {"n": len(rframes), "motion": {"frame_ms": ratlas["frame_ms"]}})()
        args.place = "grid"
    else:
        fig = Figure(char, motion)
    last = stage_name(track, len(TRACKS[track]["stages"]) - 1)
    if rigline:
        pw, ph = ratlas["frame"]["w"], ratlas["frame"]["h"]
    else:
        pw, ph = pose["canvas"]
    W, H = [int(x) for x in args.canvas.split(",")] if args.canvas else (pw, ph)
    s = min(0.9 * SIZE / pw, 0.9 * SIZE / ph)          # source px per sprite px of the pose render
    ox, oy = (SIZE - pw * s) / 2, (SIZE - ph * s) / 2
    ow = 0 if rigline else cfg["sprite"].get("outline", 0)    # a rig frame's outline is already in its lines
    infl = 1 + 2 * ow / (H * 0.95)               # the outline is added on top, so paint the figure smaller
    imgs = {}
    for t in range(fig.n):
        f = d / f"{last}_{t}_{args.seed}.png"
        if f.exists():
            rgb = np.asarray(Image.open(f).convert("RGB"))
            imgs[t] = (rgb, deshadow(rgb, keyout(rgb)))
    hs = [np.ptp(np.nonzero(a.any(1))[0]) + 1 for _, a in imgs.values()]
    grid_k, grid_o = (lineart.layout(rframes, SIZE) if rigline else (None, None))
    # the ground row is where the stance frames' feet land (their lowest pixel, 97th percentile over frames); a frame whose feet
    # sit higher than that is in the air, and stays there
    ground_src = float(np.percentile([np.nonzero(a.any(1))[0].max() + 1 for _, a in imgs.values()], 97))
    cell = grid_k if rigline else (float(np.median(hs)) / (H * 0.93) * infl if args.place != "fixed" else s * infl)
    frames, names = [], []
    for t, (rgb, alpha) in imgs.items():
        ys, xs = np.nonzero(alpha)
        if rigline:                  # the line art sat on a whole-number grid: sample it back exactly
            origin = grid_o
        elif args.place == "fit":      # ground contact: every frame sits on the floor
            origin = (xs.mean() - W / 2 * cell, ys.max() + 1 - (H - 2 - ow) * cell)
        elif args.place == "skel":   # the skeleton's own ground row: flight frames leave the ground
            origin = (xs.mean() - W / 2 * cell, ground_src - (H - 2 - ow) * cell)
        else:
            origin = (ox + (pw * s) * (1 - 1 / infl) / 2, oy + (ph * s) * (1 - 1 / infl) / 2)
        sp = despeckle(to_sprite(rgb, alpha, pal[0], pal[1], cell, origin, (W, H), cfg["sprite"]["accent"]))
        if ow:
            sp = thicken(sp, ow, pal[0][np.argmin(pal[1][:, 0])])
        frames.append(sp)
        names.append(t)
    if not frames:
        sys.exit("no frames")
    sheet = np.concatenate(frames, axis=1)
    Image.fromarray(sheet, "RGBA").save(d / f"sheet_{args.seed}.png")
    (d / f"atlas_{args.seed}.json").write_text(json.dumps({
        "frame": {"w": W, "h": H}, "count": len(frames), "frame_ms": fig.motion.get("frame_ms", 70), "loop": True, "rev": rev(),
        "source": {"poseguide": track + TAG, "motion": motion, "seed": args.seed, "place": args.place, "frames": names}}, indent=1))
    review(char, motion, track + TAG, args.seed, frames, names, fig, d, last)


def make(args):
    """gen, then one review sheet per seed. Run the whole thing through the queue:
    ws submit --cores 2 --mem 34G --vram 30G --expect 20m --name pg-CHAR -- .venv/bin/python -m forge.poseguide make CHAR MOTION qwen-pose-run --seeds 1,2 --name GAME/ANIM"""
    gen(args)
    for sd in [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]:
        args.seed = sd
        sprite(args)


def land(args):
    """Copy a finished sheet into the forge character's sprites/pg-MOTION and import it into the game with scripts/import-sprites.py."""
    char, motion, track = args.char, args.motion, args.track
    d = paths(char, motion, track)
    out = ROOT / "characters" / char / "sprites" / f"pg-{motion}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(d / f"sheet_{args.seed}.png", out / "sheet.png")
    shutil.copy(d / f"atlas_{args.seed}.json", out / "atlas.json")
    name, anim = args.name.split("/")
    importer = ROOT / project()["importer"]
    spec = importlib.util.spec_from_file_location("import_sprites", importer)
    imp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imp)
    imp.FORGE = ROOT / "characters"
    imp.main(char, f"pg-{motion}", name, anim)


CUR = ""


# Where a project keeps its game: forge.toml beside its characters, with importer and assets paths.
def project():
    return tomllib.loads((ROOT / "forge.toml").read_text())


def review(char, motion, track, seed, frames, names, fig, d, last):
    out = DATA / "review/pose-options"
    out.mkdir(parents=True, exist_ok=True)
    z = 3
    track = track
    cw, chh = frames[0].shape[1] * z, frames[0].shape[0] * z
    ims = [upscale(f, z, (40, 34, 52)).convert("RGB") for f in frames]
    pose_row = [Image.open(d / f"pose_{t}.png").convert("RGB").resize((cw, chh), Image.LANCZOS) for t in names]
    raw_row = [Image.open(d / f"{last}_{t}_{seed}.png").convert("RGB").resize((cw, chh), Image.LANCZOS) for t in names]
    cur = None
    if CUR and Path(CUR).exists():     # the version in the game now, same size, for the side-by-side
        c = Image.open(CUR).convert("RGBA")
        n = len(names)
        fw = c.width // n
        cur = upscale(np.asarray(c), z, (40, 34, 52)).convert("RGB")
        fw3 = fw * z
    sheet = Image.new("RGB", (max(cw * len(names), cur.width if cur else 0), chh * (4 if cur else 3)), (40, 34, 52))
    if cur:
        sheet.paste(cur, (0, 3 * chh))
    for k in range(len(names)):
        sheet.paste(pose_row[k], (k * cw, 0))
        sheet.paste(raw_row[k], (k * cw, chh))
        sheet.paste(ims[k], (k * cw, 2 * chh))
    base = f"{char}_{motion}_{track}_s{seed}"
    sheet.save(out / f"{base}.png")
    ims[0].save(out / f"{base}.gif", save_all=True, append_images=ims[1:], duration=90, loop=0)
    print("wrote", out / f"{base}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "sprite", "make", "land"])
    ap.add_argument("char"); ap.add_argument("motion"); ap.add_argument("track")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", default="")
    ap.add_argument("--place", default="skel", choices=["skel", "fit", "fixed"])
    ap.add_argument("--hands", default="free", help="pose.toml hands_<name> text, e.g. phone")
    ap.add_argument("--note", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--src", default="run", help="rig-paint: the finished rig animation under characters/CHAR/sprites/")
    ap.add_argument("--seeds", default="", help="gen: comma list of seeds in one ComfyUI session")
    ap.add_argument("--canvas", default="", help="W,H of the sprite frame, default pose.toml canvas")
    ap.add_argument("--name", default="", help="land: GAME_NAME GAME_ANIM as NAME/ANIM")
    a = ap.parse_args()
    TAG = ("-" + a.tag) if a.tag else ""
    if a.name:
        CUR = str(ROOT / project()["assets"] / "sprites" / a.name / "sheet.png")
    {"gen": gen, "sprite": sprite, "make": make, "land": land}[a.cmd](a)
