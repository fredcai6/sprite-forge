# Pizza Runner: next art round

Tommy has cleared a big image push. All of `characters/STYLE.md` still holds, with these changes from him:
**living cities, no post-apocalypse.** No overgrown ruins, no rubble, no boarded windows, no flaming trash
barrels as a street-war look (a warm brazier/food-cart style heat barrel is fine). Lit windows, awnings, trees in
good health. Backgrounds light and behind the action, player and enemies heavy outlines, no gibberish lettering
anywhere (2x check, region by region).

Game repo: `~/projects/probable-octo-funicular`. Put finished files in its `assets/` as below and do not
touch `js/` or commit there; the main session is rewriting the code at the same time and commits assets. Commit art
sources on the sprite-forge branch as you go. The game picks up files by name; anything missing falls back, so
land things one at a time and in priority order. GPU: `~/projects/gpu-manager/gpu submit` as usual.
FLUX.2 Klein beside Dory (`--vram 17G`), Qwen edit with the whole card (`--vram 30G`).

## Sizes: art height = height drawn, so pixels stay square
| Sprite | Height | Animations (game name) |
|---|---|---|
| slice-ny | 124 | run, idle, jump, fall, cold |
| slice-deep | 170 | run, idle, jump, fall, cold, special |
| slice-jumbo | 240 | run, idle, jump, fall, cold, special |
| slice-naples | 104 | run, idle, jump, fall, cold, special |
| teen | 118 | walk, idle, film (phone up), flee (crying, running) |
| turtle | 92 | walk, munch, stun (tucked, dizzy) |
| ninja | 118 | walk, throw, leap, crouch (perched up high), hurt |
| mutant | 120 | run, idle, bite |
| customer-CITY-AREA | 150 | idle (the door scene draws at exactly 2x) |
| alewife | 170 | idle, wave (2x at her door too) |

## Motion
Tommy agreed the motion is stiff. Run cycles need at least 8 frames at about 70 ms with real contact / down / pass /
up poses, not one pose bobbing. Idles 4-6 frames that loop cleanly. `jump` and `fall` are distinct poses.
- **Deep Dish** must look thick and enormously heavy: a tall pan slice, deep crust wall, lumbering, knees bending.
  Redo it at 170.
- **Jumbo** must look *silly*: an absurd, oversized, droopy NY-style slice far too big for its tiny legs, tip sagging,
  a bit embarrassed. Redo it at 240. The game adds a wobble and tip-flop warp on top, so leave the tip free.
- **Margherita** is small and quick, 104, basil and fresh mozzarella. Its `special` is now a forward dash pose.

## Mutant: full redesign, top priority
Not a floating creature. An original underground sewer-scavenger human mutant: hunched, ragged patched clothes
scavenged from junk, mismatched eyes, an odd extra feature or two, sad and hungry, comic rather than scary.
Runs fast on the ground. It climbs out of storm drains; the game does the rising/sinking by clipping at the ground.

## DJ Alewife
`characters/alewife/references/channel-avatar.png` is her channel avatar, the starting point for how she looks
(Katelyn's DJ persona, this is the gift's final reveal, so take the most care here). Long wavy pink-red hair, light blue
headband, warm tan skin, brown eyes, freckle-blush, tiny gold nose stud, pearl earring, gold heart necklace, navy tank
top, holding up a frothy beer stein, winking. Full body, in our sprite style at 170, same heavy outline. Use Qwen
edit from the reference. Also deliver `assets/backgrounds/hq/alewife-polaroid.png`, 300x300, her head and stein, for
the corkboard photo.

## Neighborhoods: each city has three areas
Layers go to `assets/backgrounds/CITY/AREA/`: `far.png` and `mid.png` 2560x720 (wrap seamlessly), `near.png` 2560x720
transparent, `ground.png` a 256x120 wrapping tile that draws at y 600 downward (sidewalk, grass path, sand, cobbles),
`door.png` the delivery door/address prop with feet at its bottom, about 170 tall. `far` is quiet distant
forms with one slow recognizable landmark, `mid` is street-scale architecture on the y 600 line, `near` is light foreground
framing. Avoid equally dominant oversized buildings at every depth. Each area must be told apart from the others at a
glance, in silhouette and palette, while staying in the city's hue family.

| City | AREA folder | Direction |
|---|---|---|
| newyork | wallstreet | Stone canyon, tall narrow facades, flags with no marks, One World-ish spire far off, cool morning-to-dusk light |
| newyork | centralpark | Lawns, stone bridges, elms, park benches and lamp posts, Midtown skyline low beyond the trees |
| newyork | brooklyn | Brownstones with stoops, water towers, street trees, the Brooklyn Bridge far off, warm brick |
| chicago | loop | Elevated tracks overhead in mid, steel bridges, deep masonry towers, big dark tower far off, cold blue |
| chicago | lincolnpark | Lakefront green, conservatory glass, the lake and a skyline line beyond |
| chicago | wickerpark | Two-flat brick apartments, corner taverns, back-porch stairs, water tanks |
| dc | georgetown | Warm brick row houses, storefronts, canal towpath, close streets |
| dc | mall | Open sky, lawns, elm rows, reflecting water, the Monument and Capitol dome small and far |
| dc | federal | Formal repeating limestone columns and arches, imposing, cream and grey |
| naples | harbor | Waterfront, fishing boats, castle mass, bay with Vesuvius clear and readable in far |
| naples | center | Narrow shop-lined lanes, arches, laundry, tiled domes, ochre and terracotta |
| naples | hillside | Stepped streets, balconies, pastel houses, the bay spread below in far |
| california | downtown | Palm-lined low-rise downtown, Queen Mary-like liner and harbor cranes far off, sunset gold |
| california | beach | Sand, lifeguard tower, pier, turquoise water, oil-island palms far off |
| california | bungalows | Craftsman bungalows, porches, palms, telephone wires, dusk; `door.png` here is the final house porch, big enough to read as a house front, 300 tall. Generic for now: Tommy will give references later |

## Shared props, `assets/backgrounds/props/` (bold outline, full saturation)
`box.png` breakable wooden crate 84x84, clearly smashable, pizza-box-brown; `box-pile.png` a pile 180 wide 120 tall that
blocks an alley; `box-broken.png` flat splintered remains 90x24; `drain.png` street storm drain 120x26; `sign.png` a blank
street sign on a post, about 200 tall, blank arm area, the game writes real text on it; `ledge.png` a 64x28 tile of black
fire-escape walkway; `pickup-TOPPING.png` for pepperoni, jalapeno, garlic, mushroom, basil, pineapple, sausage, a 52x52
glowing topping token; `shot-TOPPING.png` 28x28 thrown topping; `star.png` 24x24 throwing star.
`assets/backgrounds/maps/CITY.png`: 1280x720 aged paper street-map texture for each city, tinted to the city, blank of
any lettering and any roads, with coast or river shapes only (the game draws the routes on top).

## HQ pies
`assets/backgrounds/hq/pie-ny.png`, `pie-deep.png`, `pie-jumbo.png`, `pie-naples.png`: whole top-down plain cold cheese pies,
clearly different from each other (NY thin 104 wide, deep dish tall-walled 96, jumbo enormous 150, Neapolitan puffy charred
rim 96), hard edges and heavy outline. The game cuts missing slices and adds toppings.

## Priority
1. mutant; slice-deep; slice-jumbo; DJ Alewife
2. shared props and maps; slice-ny and slice-naples motion; turtle, teen, ninja new animations
3. New York's three areas, then customers for them, then the other cities in route order
4. HQ pies

## Customers: one each, uber-stereotypes of the place, `customer-CITY-AREA`
newyork: wallstreet banker in a pinstripe suit yelling into a phone; centralpark old man on a bench-side with pigeons on him;
brooklyn bearded man with a tote bag and tiny dog. chicago: loop mustached guy in a big parka and sports cap with no logo;
lincolnpark jogger mom with a stroller; wickerpark tattooed bartender. dc: georgetown professor in a bow tie and tweed;
mall sunburned tourist with fanny pack and camera; federal agent in a black suit, earpiece and sunglasses.
naples: harbor fisherman in a cap holding a fish; center tiny stern nonna in black with a wooden spoon; hillside gesturing man
with a tiny espresso. california: downtown roller-skating woman with giant sunglasses; beach surfer dude with a board.
Broad, cartoony and affectionate, no logos.

Write a short status line per finished item to `characters/NEXT_ROUND_STATUS.md` so the main session can see what landed.
Old `slice-thin` is retired; leave it unused.
