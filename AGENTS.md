# sprite-forge

Sketches and references in, consistent animated pixel sprites out. Each sprite's `atlas.json` records
the code revision and the key pose it was built from.

## Setup

`uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python`, then `comfy/fetch-models.sh`,
clone ComfyUI at `comfy/COMFY_REF` into `/mnt/data/sprite-forge/ComfyUI` and build `comfy/Dockerfile` from that checkout
as `sprite-forge-comfy:latest`.

## The loop

1. `characters/NAME/character.toml` holds the description, sprite height and palette families;
   put references in `references/` and run `PYTHONPATH=. .venv/bin/python characters/palette_build.py NAME`.
2. Write key poses in `characters/NAME/runs/X.toml`, then generate through the GPU queue:
   `~/projects/gpu-manager/gpu submit --vram 30G --name sprite-NAME -- .venv/bin/python -m forge.generate characters/NAME/runs/X.toml qwen-edit`
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
