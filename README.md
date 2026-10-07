# Noise Contour Mapper

A desktop tool for converting discrete sound level meter (SLM) measurements into a
spatial noise contour map overlaid on a client floor plan. Built for occupational noise
assessment, hearing conservation program planning, and communicating noise risk.
Everything runs locally on your machine — no data leaves it, and no internet connection
is needed.

## Run it

Three options:

- **Online (share this link):** <https://huangmarkc.github.io/noise-contour-mapper/> —
  the same tool, no install needed. It updates automatically whenever this repo's
  `main` branch changes. Floor plans and measurements never leave your browser.
- **Desktop app:** run `Noise Contour Mapper.exe` (or install it via the
  setup exe under `src-tauri/target/.../bundle/nsis/`).
- **Browser (offline):** double-click `ui/index.html` — the same tool opens locally.

## Workflow

The app has a full step-by-step **user guide** built in: click **? Guide** above the map
or press <kbd>H</kbd>.

1. **Load floor plan** — PNG, JPG, or PDF (first page is rendered).
2. **Calibrate scale** (optional) — click two points a known distance apart (e.g., a
   dimensioned wall) and enter the real distance. Adds a scale bar to the map and shows
   cursor positions in meters/feet.
3. **Add measurements** — switch to *Add points* and click each SLM location, entering the
   measured dBA. Or import a CSV (`label,x,y,dBA` or `x,y,dBA`, coordinates in pixels or
   percent of the plan). Drag markers in *Pan* mode to reposition; edit values in the list.
4. **Draw walls and barriers** (optional) — *Draw wall* in the toolbar; click each
   corner, double-click or Esc to finish, leave gaps for doorways. Ends snap to other
   walls so corners close. Each wall has a type preset (concrete, acoustic panels,
   drywall, metal, glass, curtain, partial barrier) with editable sound reduction
   ("Blocks", dB), absorption (%), and — for partial barriers — height.
5. **Draw rooms for reverberation** (optional) — *Draw room* in the toolbar; click
   each corner, then the first corner again (or double-click) to close. Each room has
   a ceiling height and a finish preset that sets its average absorption; the
   reverberation time (Sabine) is shown and can be overridden with a measured value.
6. **Tune the map** — calculation method (IDW with adjustable power, multiquadric RBF,
   or the source model below), grid resolution, banded vs. smooth gradient, overlay
   opacity, and an optional mask that hides interpolation far from any measurement.
   With walls, *Measurements don't pass through walls* makes every method measure
   distance around walls (through openings) rather than through them.
7. **Source model + residuals (optional)** — place known equipment as point noise
   sources (toolbar → *Add source*), each with its sound level at a reference distance.
   Levels are predicted with the inverse square law (`Lp2 = Lp1 − 20·log10(r2/r1)`,
   0.5 m near-field clamp) and combined by logarithmic energy addition, then calibrated
   against the SLM measurements by IDW-interpolating the residuals
   (`measured − predicted`) and adding them back in dB space. Requires a calibrated
   scale. With walls, each source also loses each crossed wall's sound reduction, bends
   around wall ends and over partial barriers (Maekawa screening in the Kurze–Anderson
   form at 500 Hz, capped at 20 dB), and reflects once off each wall (image sources,
   reduced by absorption). With rooms, each room adds its reverberant field
   (`Lrev = Lw + 10·log10(4/R)`, `R = S·ᾱ/(1 − ᾱ)`, `Lw = Lp1 + 20·log10(r1) + 11`),
   with rooms coupled through the doorways and walls between them, and each room's
   readings correct that room. The same math is available as a standalone Python module in
   [noise_model.py](noise_model.py) (verified to match the app to within 1e-12 dB).
8. **Noise bands** — defaults are <80, 80–85, 85–90, 90–95, >95 dBA; boundaries and colors
   are fully editable, and contour lines are drawn at each boundary.
9. **Export** — PNG of the finished map (with legend, scale bar, title block, date, and
   method note), CSV of the measurement points, or save/open the whole project as JSON.

## Project layout

- `ui/` — the entire application (single `index.html` plus a bundled copy of pdf.js in
  `ui/vendor/` so PDF import works offline).
- `src-tauri/` — the Tauri v2 desktop shell that wraps `ui/` as a native Windows app.

## Rebuilding the desktop app

```
npm install
npx tauri build
```

The build writes to `%LOCALAPPDATA%\noise-contour-build` (kept outside OneDrive on
purpose); the exe and NSIS installer land under `...\release\` and
`...\release\bundle\nsis\` there. Dev mode: `npx tauri dev`.

## Notes

- Interpolated values are estimates between measurements. Verify critical boundaries
  (e.g., the 85 dBA action level) with additional measurements before making program
  decisions.
- The wall model is first-order: one reflection per wall, no floor/ceiling reflections,
  single-number (dBA) values. Reverberation is modeled only in drawn rooms, as an
  evenly spread (diffuse) field. Residual correction against the
  measurements absorbs much of what it leaves out near the measured locations.
- Click **Load demo** to see a worked example on a synthetic facility, with walls,
  doorways, acoustic panels, a partial barrier, and six rooms.

## Your data

The tool collects, uploads, and stores nothing. Floor plans and measurements exist only
in the open browser tab or app window and are erased when it closes. Users save their
own work with **Save project** / **Export**, choosing the location (their computer or
their own cloud drive), and are responsible for those files. The page warns before
closing if there is unsaved work.

## License & ownership

Copyright © 2026 Mark Huang. All rights reserved. This software and all associated
materials are the exclusive property of Mark Huang; no permission is granted to copy,
modify, distribute, or create derivative works without prior written consent — see
[LICENSE](LICENSE) for the full notice, including the no-warranty disclaimer.
Exception: `ui/vendor/` contains Mozilla's pdf.js, © Mozilla Foundation and
contributors, under the Apache 2.0 license.
