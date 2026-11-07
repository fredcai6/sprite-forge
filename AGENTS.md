# sprite-forge

A tool for any project, not a home for one project's art. Characters, backgrounds, style rules and build scripts
belong to the project that uses them, in a folder of its own with `characters/` and a `forge.toml` naming its
`importer` and `assets` paths. Pizza Runner's are in `art/` of its repo. `characters/derek` stays here as the worked example.
Run from that project folder with this repo on the path, or set `FORGE_PROJECT`:
`cd PROJECT/art && PYTHONPATH=~/projects/sprite-forge ~/projects/sprite-forge/.venv/bin/python -m forge.runrig characters/NAME/rig/run.toml`

Sketches and references in, consistent animated pixel sprites out. Each sprite's `atlas.json` records
the code revision and the key pose it was built from.

## Setup

`uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python`, then `comfy/fetch-models.sh`,
clone ComfyUI at `comfy/COMFY_REF` into `/mnt/data/sprite-forge/ComfyUI` and build `comfy/Dockerfile` from that checkout
as `sprite-forge-comfy:latest`.

## The loop

1. `characters/NAME/character.toml` holds the description, sprite height and palette families;
   put references in `references/` and run `python -m forge.palette_build NAME`.
2. Write key poses in `characters/NAME/runs/X.toml`, then generate through the GPU queue:
   `ws submit --cores 2 --mem 34G --vram 30G --name sprite-NAME -- .venv/bin/python -m forge.generate characters/NAME/runs/X.toml qwen-edit`
   (`--mem` and `--cores` are measured: ComfyUI's container peaked at 30.7 GB for generate runs and 31.7 GB
   for pose runs, model files included, on 1.4-2 cores, across 145 runs from 2026-11-03 to 11-07.)
3. Look at `python -m forge.contact RUN_DIR OUT.png` and choose keys.
4. Write `characters/NAME/anims/Y.toml` and run `python -m forge.animate` on it. Output lands in
   `characters/NAME/sprites/Y/`: sheet, atlas, preview.

## What we have measured

- Qwen-Image-Edit 2511 draws key poses: correct limb count and it follows the pose. It reached
  29.8 GB of card use at 1248px, so it takes the whole card. 12 s per image.
- FLUX.2 Klein 4B fits beside Dory (`--vram 17G`) at 6.5 s per image, and often keeps a third arm.
- Neither model nods on request. Head motion comes from `forge/animate.py`.
- Derek's native scale is about 6.1 source pixels per sprite pixel; at 185px tall he keeps his detail.
- Outputs and comfy.log live in `/mnt/data/sprite-forge`, weights in `/mnt/data/models/comfy`.

## Pose-guided runs (humans)

For a character whose body reads as a human skeleton, `forge/poseguide.py` draws a run or walk from a skeleton:
`characters/NAME/pose.toml` (body plan, master, prompt notes) plus `forge/motions/MOTION.toml` (gait table) feed
`forge/posekit.py`, which renders the OpenPose skeleton that Qwen-Image-Edit paints the master onto (track `qwen-pose-run`).
One command through the queue: `ws submit --cores 2 --mem 34G --vram 30G --name pg-NAME -- .venv/bin/python -m forge.poseguide make NAME MOTION qwen-pose-run --seeds 1,2 --name GAME/ANIM`
writes review sheets to `/mnt/data/sprite-forge/review/pose-options/`; `poseguide land NAME MOTION qwen-pose-run --seed N --name GAME/ANIM` imports the pick.
It does not work on bodies a skeleton does not describe (pizza slices, turtles): the master's stance wins. Keep `forge.runrig` for those.

## Dressing a sprite at runtime

When something on a character changes during play (toppings, armor, a logo), paint it out of the finished sheets and let the game put it back. Mark it once in `characters/NAME/decals.toml` on one reference frame, then:

`FORGE_PROJECT=PROJECT/art .venv/bin/python -m forge.decals characters/NAME/decals.toml`

Each frame is registered to the reference, so the marks land correctly on redrawn poses too. It writes `sprites/ANIM-plain/` with the cleaned sheet, a mask of the surface decals may show on, and `seats` in the atlas: where each decal sits on each frame. Drawn through the mask, decals follow the body and pass behind hands. It takes minutes per animation at 330 px wide, because it searches rotation and scale. Check the result by eye: over-wide marks eat faces.

`forge.blink` adds a closed-eye frame to a held idle: `python -m forge.blink characters/NAME/sprites/idle-plain cx,cy,rx,ry ...`, with the ellipses large enough to cover the whole eye and clear of the brows.
