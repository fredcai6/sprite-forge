# Next round status

- shared props (box, box-pile, box-broken, drain, sign, ledge, star, 7 pickup-*, 7 shot-*): landed in assets/backgrounds/props/ (sources.json beside). sign board is cream-blank; drain is a thin angled grille.
- maps/{newyork,chicago,dc,naples,california}.png (1280x720 blank paper + coast/river shapes, no roads/text): landed in assets/backgrounds/maps/.
- HQ pies pie-ny (104), pie-deep (96), pie-jumbo (150), pie-naples (96): landed in assets/backgrounds/hq/. Plain cold cheese, outlined discs; deep dish reads as a dark pan rim, not tall-walled (top-down limit).
- slice-deep redone: 230x170 frames (art height 170), run 8 frames @70ms, idle 6, jump, fall, cold, special. Heavier wedge, but the run poses differ only modestly (model keeps similar side poses).
- slice-jumbo redone: 330x240 frames (art height 240), run 8f @70ms, idle 6, jump, fall, cold, special (belly flop). Tip is left free; floppy sagging slice with tiny legs.
- mutant redesigned (sewer scavenger, patchwork pizza-box vest, boots): 170x120 frames (height 120), run 8f @70ms, idle 6, bite 6. Old walk removed from manifest. Ground-level; the game clips for drains.
- alewife: 150x170 frames (height 170), idle 6, wave 6 (wave has a few stray hair-wisp pixels). hq/alewife-polaroid.png 300x300 (head + stein, from the same sprite key pose, hard-edged).
- slice props/maps/pies as above. NY areas wallstreet, centralpark, brooklyn landed in assets/backgrounds/newyork/<area>/ (far, mid, near, ground 256x120, door ~170 tall). Near layers cleaned (forge.neardeclutter): only ground-line furniture; also applied to old per-city near.png files.
- chicago areas loop, lincolnpark, wickerpark landed (assets/backgrounds/chicago/<area>/). All near layers re-cleaned: only ground-line furniture, no floating trees/banners/wires (forge.neardeclutter; applied to every near.png incl. the older per-city ones).
- dc areas georgetown, mall, federal; naples areas harbor, center, hillside landed in assets/backgrounds/<city>/<area>/. Weak: naples/center ground cobbles read dark; naples/hillside mid has faint pink wash.
- california areas downtown, beach, bungalows landed (bungalows door.png is 200x300, a porch/house front). Lettering check done at 1x over 2 passes x 15 areas (mid layers, 1280x250 strips): no lettering found; california/downtown has a clock-face glyph on one tower (a shape, left).
- slice-ny run: new 8-frame cycle (contact/down/pass/up, both legs, arms in opposition) @70ms, frame 140x124. Other slice-ny anims (idle/jump/fall/cold) still old until keys3 finishes (a GPU memory fault killed the first keys job; resubmitted).
- slice-naples redone at 104: frames 130x104, run 8f @70ms, idle 6, jump, fall, cold, special = forward dash pose (8 frames).
- teen film (120x118, 4f) and flee (150x118, 4f @80ms, crying run) landed; walk/idle unchanged.
- turtle munch (4f) and stun (4f) landed at 150x92. WEAK: stun does not tuck head/legs (model refused), it is a dazed face; stars clipped by canvas top. The new pose is more upright than walk.
- ninja leap (7f), crouch (6f, perched: a single crouch pose with a 1px breath; ledge removed), hurt (8f) landed at 128x118. WEAK: crouch has no second pose (generator drew a ledge, one version cleaned).
