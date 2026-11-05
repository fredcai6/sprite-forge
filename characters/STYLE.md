# Pizza Runner art style

One rule keeps every asset in one game: **one art pixel is one pixel of the 1280x720 stage.** Sprites and
backgrounds are delivered at that scale, hard-edged, with no smoothing and no semi-transparent pixels.
The game draws a sprite frame at its native size when the frame height equals the height it asks for
(hero 124, old timer 130, teen 118, turtle 92), so those are the canvas heights.

## Pixel density
- Sprites: the key pose is painted large, then sampled onto the stage grid. About 7 painted pixels per
  stage pixel for the hero (`cell` in each anim file). A 124 px hero carries 35-65 colours per animation.
- Backgrounds: painted at 1.5x or larger, reduced to the stage grid, then snapped to a 40-64 colour palette
  (`forge/layers.py: quantize`). Never resample a finished layer.
- Cutscene zooms must use whole-number scales so pixels stay square.

## Outline
Every character and prop has a 2 px outline in near-black violet (about #1e1030), never pure black. Backgrounds
carry the same outline on buildings and furniture; distant layers (`far`) soften it by drawing it lighter.

## Palette
Each character has its own `palette.json`, built from its key poses in hue families (neutral, red, orange,
yellow, green, blue, purple). The shared ground rules:
- Shadows go violet, not grey or black: deep purple-brown (#2a1830 to #4a2a48).
- Highlights go warm cream-yellow, never white, except gloves, eyes, and the cold slice's frost.
- The hero is the most saturated thing on screen: cheese yellow-orange (#f5b82a), pepperoni red (#c01c28),
  sneakers red with white soles. Everything else is a step duller so he reads at once.
- Enemies and the old timer are desaturated and earthy (olive, tan, grey, orange hoodie as the one accent).
- Cold slice: grey-blue (#8fb0c4 base) with dull dark pepperoni.

## Lighting
- Interiors: one warm practical light (the hanging bulb) as key, orange pool on the wall, falloff to deep violet
  at the corners and floor edges. Characters take the same warm light from upper left.
- Street: dusk, purple sky into orange at the horizon; warm lamp pools; everything else cool and dusty.
- Foreground layers are near-black violet silhouettes with no interior detail.

## Depth
- HQ: `back` (room and blank corkboard), `mid` (props along the wall, feet at y 540), `front` (dark edge
  silhouettes). Floor line y 480.
- Street: `far` (opaque skyline), `mid` (storefronts, feet on the y 600 ground line), `near` (dark litter and
  wires, mostly transparent). `far`, `mid` and `near` are 2560 wide and wrap.

## Characters
Cartoon mascots with big expressive eyes, a clear line of action, chunky limbs, white gloves for the pizzas.
Pizzas have a face on the cheese, thin arms and legs, sneakers or boots. Humans and animals share the same
outline weight and shading but are not pizzas, and carry no masks or weapons.

## Records
Each `atlas.json` records the sprite-forge revision, the generating run and model, and the shot/seed of every
frame. Each background folder has a `sources.json` with the same for its layers.
