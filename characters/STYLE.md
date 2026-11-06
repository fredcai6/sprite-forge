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
- Player and enemies (`slice-ny`, `old-timer`, `teen`, `turtle`) carry a heavy outer silhouette of about 4 px in near-black
  violet (the palette's darkest colour), added in `forge.compose` (`outline = 2` in `character.toml` grows the
  generated 2 px edge to 4) on every frame, with the figure painted slightly smaller so the frame height is unchanged.
  Interior lines stay as generated.
- Backgrounds use a thin soft outline, never heavier than the characters, so the characters always win the eye.
- Interactive props (platform, barricade, barrel, door) keep a bold outline and full saturation so they stand clear.

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

## No fake text
Image models write lookalike letters. None may reach a game asset.
- Every generation prompt states that there is no text, letters, logos, numbers, labels, stickers or graffiti
  lettering, and that signs are blank or abstract. This reduces gibberish but does not remove it.
- Before an asset is kept, inspect it at 2x region by region (about 640x380 source regions) for letter-like marks on
  signs, awnings, windows, boxes, tins, sacks and walls. Remove what is found: pick another seed, or fill the box with
  the sign's own median colour (`ERASE_MID` in `backgrounds/newyork/build.py`).
- The only text allowed is lettering a person reads as correct, such as the door's DELIVER.

## Background weight
Backgrounds sit behind the characters: lighter values, lower contrast, little fine detail, a touch of atmosphere.
`forge.layers.soften` does it (lifted values, flatter contrast, an atmosphere wash, a 3x3 median to drop cracks).
Current settings: street `far` contrast 0.9, `mid` 0.72 with a 22% haze, `near` flat dark silhouettes; HQ 0.92 with
a small lift, kept dim but readable. Mid layers carry the most weight, so they are softened the most. Characters and
interactive props are never softened.

## Keying sprites
`forge.compose.keyout` takes everything key-coloured (including between limbs), plus the key-tinted gradient and ground
shadow only where it connects to the image border, then fills every enclosed hole. A dark shell or boot is never
mistaken for background. Check every sprite sheet over bright magenta before it ships.

## Cities
Each city has its own hue family so none is mistaken for another: New York dusk lavender and brick, Chicago cold blue-grey steel
and red brick, DC cream marble and sage green, Naples terracotta, ochre and teal shutters at warm dusk, California sand gold and
turquoise. Barrels take the city's hue (blue, teal, rust, crimson); flames stay orange so the heat pickup reads everywhere.
Platforms are hand-built 64x28 tiles (`backgrounds/props/city_props.py`), never generated, because generated girders were noisy.
Per-city build parameters live in `backgrounds/cities.py` (ground row, seeds, haze) and every pick is logged in `sources.json`.
