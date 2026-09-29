# Limitless — Hollow Purple Gesture Experience

An unofficial Jujutsu Kaisen fan experience controlled through hand tracking. Your left hand summons Blue, your right hand summons Red, and their convergence creates a hand-aimed Hollow Purple release. The original native Python prototype is preserved unchanged.

## Run

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/gojo_hollow_purple.html . Stop with Ctrl+C. No package installation is needed for the browser app. MediaPipe loads only when camera mode is requested.

## Gesture ritual

1. Select **Begin With My Hands** and allow camera access.
2. Hold up your index and middle finger on the left hand to summon Blue.
3. Hold the same pose on the right hand to summon Red.
4. Bring both hands’ index-and-middle fingertips together to charge Purple.
5. When Purple is ready, it moves to your right index-and-middle fingertips. Use your right hand to point, cross those fingers, then uncross them to release.

Blue and Red are summoned forces, not projectiles. Purple is the only attack. It follows the visible hand-aimed trajectory and clears simulated targets in its path. Loss of tracking cancels the technique rather than releasing it. The on-screen reticle is the authority for aim and handedness.

Recognition is experimental and lighting-dependent. Exit stops webcam tracks. Backgrounding resets the technique. Microphone is never requested. No video recording or upload is implemented. Reduced Motion follows the device preference and removes high-intensity flashes, camera shake, speed lines, and the Purple screen burst.

## Validation

Requires Node 22.12+ (tested with Node 24.8) for the dependency-free tests:

```sh
node --check experience.js
node --check combat.mjs
node smoke-test.cjs
```

The tests cover the hand-only state flow, no Blue/Red projectile routes, Purple release, directional aim, swept Purple collision, cooldowns, tracking loss, mirror/crop mapping, cleanup, and particle limits. There is no configured TypeScript checker or linter. Browser checks supplement these mocked tests; they do not certify every physical camera or mobile browser.

## Architecture

- `gojo_hollow_purple.html`: landing, help, controls, accessible readouts.
- `experience.css`: responsive presentation.
- `experience.js`: input, state machine, canvas rendering, tracking, audio.
- `combat.mjs`: browser-independent projectile and target simulation.
- `smoke-test.cjs`: regression suite.
- `RESEARCH.md`: sources, adaptation decisions, and limits.
- `gojo_hollow_purple.py` / `requirements.txt`: older native prototype, not revalidated here.

## Deployment readiness

**Local beta: yes. Fully validated public release: no.** The app is static and can be hosted on HTTPS, but camera accuracy, low-end performance and browser compatibility still require a real-device acceptance pass. The local Python server is for development only.

Only publish the HTML, CSS, `experience.js`, and `combat.mjs`; configure the HTML as the default page or serve it at its explicit path. Do not publish the whole development directory. Keep the four files together, ensure `.mjs` uses a JavaScript MIME type, and serve over HTTPS. Verify reloads and camera permission on the final origin before announcing it.

Camera mode downloads version-pinned MediaPipe 0.10.18 from jsDelivr and hand-landmarker model version 1 from Google Storage. GPU setup falls back to CPU. Fonts use Google Fonts with local fallbacks. Network failure must leave practice usable; camera setup has a timeout and retry path. A public production build should review/self-host these dependencies as appropriate for its availability/privacy requirements. No hosted service has been created or deployed.

All graphics and sounds are procedural; no official anime footage, soundtrack, or artwork is bundled. Fan status does not mean this is an official or licensed commercial product.
