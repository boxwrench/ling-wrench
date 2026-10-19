You are building a demo for a public showcase of what a code-generating model can do. Write ONE self-contained HTML file that runs by opening it. The full requirements follow; everything below is part of your task.

THE EXPERIENCE

A stylized pixel-art pagoda on a stepped island hill at dusk: a three-tier tower with curved eaves, glowing paper lanterns along the stone steps, cherry trees in bloom, a still pond at the foot, stars coming out. The visitor orbits slowly (it may auto-drift until touched), zooms, and taps three small buttons — day, sunset, night — that relight the whole scene: lanterns take over after dark, the pond mirrors the sunset, petals drift through the air throughout.

WHAT MAKES IT DELIGHTFUL

- It must read as PIXELS, not as a low-poly 3D scene: big chunky pixels with hard edges, flat poster-like shading, a tiny harmonious palette. A screenshot at any moment should look like game art.
- The time-of-day switch is the magic trick: one tap and the sun color, sky, fog, lantern glow, and pond tint all swing together to a new mood, with the pixel look intact in every mode.
- Small life sells it: drifting petals, flickering lantern halos, glassy pond, blinking stars. Every animated thing must stay chunky — motion never smears the pixels.

HOW TO BUILD IT (these techniques are known to work)

- Build everything from boxes: the pagoda is stacked shrinking tiers, each a body box plus roof slabs that overhang the walls and kick upward at the corners — fake the classic curve with two or three stepped slabs plus lifted corner blocks. Trunks are jittered stacks of brown boxes; canopies are clusters of pink boxes; the hill is a stepped voxel mound from a small height grid (around 64×64 cells is plenty); the pond is one flat plane. Let the blockiness show — it IS the style.
- The pixel look: render the 3D scene into a small offscreen target at roughly one-third to one-quarter of screen resolution (keep the aspect ratio), then draw that image to the screen with nearest-neighbour filtering and no antialiasing anywhere on that path. Shade with flat materials and quantize the lighting to two or three hard bands instead of smooth gradients. Hold the whole scene to a small fixed palette — on the order of a dozen colors — and grade every material toward it.
- Lanterns: emissive boxes plus soft additive glow sprites that intensify toward night; carry the night scene with one moon-blue ambient plus the lantern lights (a few real lights at most — fake the rest with emissive and a warm ambient lift near the steps).
- Petals: a single instanced set of small quads looping on the wind; stars: points that fade in at night. The pond: a flat shader with banded sky-color reflection plus the pagoda's silhouette suggested as a darkened smudge — a hint, not a true mirror.
- Interface: title and day/sunset/night buttons as a crisp HTML/CSS overlay. Never render text or buttons inside the pixel buffer — they would turn to mush.

HARD CONSTRAINTS (all must hold)

1. Exactly one HTML file. All code inline.
2. three.js ONLY via an import map pinned to `https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js` (plus `.../examples/jsm/` only if you must). No other CDN content, no fonts, no textures, no voxel downloads, no build step — all geometry generated in code.
3. Classic WebGL rendering (`THREE.WebGLRenderer`). No WebGPU.
4. It must run by opening the file in a current desktop or mobile browser — no server, and no network beyond the three.js CDN.
5. 60 fps on an ordinary laptop: merge static boxes by color or use instancing (never thousands of separate meshes), cap the final pixel ratio (~2 max; the inner pixel target stays tiny), zero per-frame allocations in the hot loop.
6. Mouse AND touch: drag orbits, wheel or pinch zooms, buttons are thumb-sized. Honor reduced-motion settings by freezing petals, drift, and flicker.

KNOWN PITFALLS — do not repeat them

- ANY smoothing on the upscale path — linear filtering, antialiasing, mipmaps — destroys the pixel look. Nearest filtering everywhere on that path, antialiasing off on the inner render.
- UI drawn inside the 3D view pixelates into unreadability. DOM overlay for all text and controls, always.
- Thousands of individual boxes mean thousands of draw calls and a slideshow. Merge or instance aggressively; the inner resolution being small does not save you from draw-call cost.
- Roof slabs with no overhang read as a wedding cake. Every roof must overhang its body and lift at the corners, or it isn't a pagoda.
- Night scenes go muddy fast. Floor the ambient so silhouettes stay readable, and let lanterns plus moonlight carry the mood.
- Palette sprawl kills the style. Define the palette first — roofs, wood, stone, leaves, blossom, water, sky per time-of-day — and forbid any color outside it.

SUCCESS means: it opens from the file, reads as pixel art in the very first frame, day/sunset/night each look intentional with lanterns ruling the night, petals drift in chunky pixels, and the console is clean.
