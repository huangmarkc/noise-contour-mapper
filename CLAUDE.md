# CLAUDE.md — Noise Contour Mapper

## What this project is
A tool for occupational noise assessment, owned by Mark Huang: it turns sound level
meter (SLM) readings into a noise contour map over a floor plan. It is used for hearing
conservation planning and for explaining noise risk to clients. Every calculation runs
in the user's browser, and floor plans and measurements are never uploaded anywhere.

## Architecture
- **`ui/index.html` is the entire application.** It is one file with inline CSS/JS:
  IDW and multiquadric RBF interpolation, a source model (inverse square law with
  energy addition) with IDW residual correction, marching-squares contours, and
  PNG/CSV/JSON export. Its only dependency is pdf.js, bundled in `ui/vendor/` for PDF
  floor-plan import.
- `src-tauri/` holds the Tauri v2 shell that wraps `ui/` as a Windows desktop app
  (`Noise Contour Mapper.exe`). The built `.exe` files are gitignored, so rebuild them
  with `npm install` then `npx tauri build` (see README for the output location).
- `noise_model.py` is a Python reference implementation of the source-model math,
  including walls (section 9: `WallModel`, `WalledSourceModel`). It matches the JS to
  ~1e-12 dB on the demo facility (compare with JS sample values, coordinates divided by
  px/m, `vertex_offset_m = 0.75/k`, `join_tol_m = 0.5/k`). If the math in
  `ui/index.html` changes, keep the two in sync.
- The calculation engine is reviewed and correct. Do not change the acoustics math
  unless Mark asks for it.
- **Walls (added Oct 2026):** wall segments in image px with a type preset (`WALL_TYPES`),
  `tl` (dB), `alpha`, and `height` (m; null = full height). Full-height walls block
  (sum of tl), sound bends around free wall ends (visibility-graph shortest path,
  Maekawa/Kurze–Anderson screening at 500 Hz, capped 20 dB, lit-side fade near edges),
  partial barriers screen over the top (source/ear heights default 1.0/1.5 m), and each
  wall reflects once via image sources that are treated like real sources (blocked,
  routed around walls, faded at wall ends). With "Measurements don't pass through
  walls", IDW/RBF/residual distances go around full-height walls; walled-off areas with
  no reachable reading are blank. The physics grid is cached (`srcGrid`) and only
  recomputed when sources, walls, scale or wall settings change.
- **User guide:** the in-app guide (`#guide`, opened with **? Guide** or H) is the main
  instructions for use. Keep it, the README, and `Noise Contour Mapper - Installation
  Guide.docx` (Quick Start, calculation sections, troubleshooting) in step with features.

## Where it is published
- **Public site:** https://huangmarkc.github.io/noise-contour-mapper/ is served by
  GitHub Pages. Pages deploys the `main` branch from the repository root (a classic
  branch deploy, not Actions). The root `index.html` redirects to `ui/`, and
  `.nojekyll` turns off Jekyll processing. **Pushing to `main` redeploys the site**,
  usually within a minute.
- GitHub account: `huangmarkc`. If a push that includes `.github/workflows/*` is
  rejected for lacking the `workflow` scope, run
  `gh auth refresh -h github.com -s workflow`. Branch deploy needs no workflows.
- **Claude Artifact version (optional):** this is a single-file copy for claude.ai. To
  make it, take `ui/index.html`, remove the doctype/html/head/body skeleton lines and
  the `vendor/pdf.min.js` script tag, and change the `loadPdf` fallback alert so it
  tells users to load PNG/JPG instead. Then publish. The artifact made from Mark's
  personal Claude account can't be edited from another account, so publish a new one
  if needed.

## Ownership notice (keep intact)
The repo is proprietary: © 2026 Mark Huang, all rights reserved (see `LICENSE`). The
notice appears in the README, in the app's sidebar footer and How-to-use section, and
on the title line of exported PNGs. Keep all of them. `ui/vendor/` is Mozilla's pdf.js
under Apache 2.0, which `LICENSE` carves out.

## Roadmap (Mark's phased plan)
1. Done: online version (GitHub Pages and Artifact), the copyright notice, no-storage
   data handling, and walls/barriers/reflections with the built-in user guide.
2. Next options, in any order Mark chooses:
   - **Reverberant-room setting:** per-room diffuse-field term (room size and surface
     finishes) so levels stop dropping with distance in large hard rooms; it would also
     soften the line-of-sight "beams" the free-field model shows through doorways.
   - **Environmental/outdoor mode:** simplified ISO 9613-2 propagation (atmospheric
     absorption, ground effect, Maekawa barriers) as another method choice.
   - **Real-map input:** OpenStreetMap underlay (Leaflet) with GPS coordinates.
3. Later: a full web app with accounts and cloud-saved projects, only once the static
   version reaches its limits.
- Airport/transport (DNL/Lden) contours are deferred. A credible version would need
  FAA AEDT or CNOSSOS-class modeling.

## How Mark wants to work
- Mark doesn't use the terminal much. Explain steps in plain language. When he needs
  to act (sign in, click an installer, approve something), say so clearly, one step at
  a time.
- After a change, verify it in a browser preview (load the demo, check the console),
  then push to deploy and check the live URL.
