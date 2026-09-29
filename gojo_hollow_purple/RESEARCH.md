# From Gojo's techniques to playable mechanics

Research checked September 28, 2026. This is an unofficial, anime-inspired 2D sandbox. Gameplay distances, lifetimes, cooldowns, damage proxies, and gestures are design choices, not canonical measurements.

## What the references support

| Technique | Reference interpretation | Implemented behavior | Deliberate adaptation |
| --- | --- | --- | --- |
| Lapse: Blue / 蒼 | Attraction toward a point; distinct from a generic projectile explosion. | The left-hand gesture summons an inward-moving Blue force. | Blue is a held visual force here, not a fired attack. |
| Reversal: Red / 赫 | Reverses attraction into repulsion. | The right-hand gesture summons an opposing Red force. | Red is also a held visual force, not a physics reconstruction of cursed energy. |
| Hollow Purple / 茈 | A combination of Blue and Red, usable as a directed destructive attack. | Converge both summoned forces, hold the result, then release along a visible trajectory. Swept collision clears targets along that path. | Clearing abstract targets is a visual mechanic. We do not assert universal matter erasure or invincibility. |

Blue and Red remain distinct summoned forces before convergence. Their visual motions differ: inward contracting Blue rings, outward Red shock rings, and a broad Purple trail. Targets respawn after five seconds so the arena stays usable.

## Sources and limits

- [Official anime episode 20 page: 規格外](https://jujutsukaisen.jp/episodes/20.php). Primary production source for the episode identity and synopsis. The page does not explain the full technique mechanics; it must not be used as sole evidence for them.
- [Official anime episode 28 page: 懐玉-肆-](https://jujutsukaisen.jp/episodes/28.php). Primary production source for the Hidden Inventory episode reference. Again, the public synopsis is not a frame-by-frame fight transcript.
- [Shueisha Games: Jujutsu Kaisen Cursed Spirit Escape, English rules](https://shueisha-games.com/wp-content/uploads/2024/05/96bd33ca03eb0b23834861b2301241ef.pdf). Publisher-hosted primary game source listing Blue, Red, and Purple as separate Gojo actions. Its board-game movement/range rules are that game's adaptation, not universal anime rules.
- [Bandai Namco: Cursed Clash overview](https://www.bandainamcoent.com/news/jujutsu-kaisen-cursed-clash-what-you-need-to-know). Official licensed-game framing of Gojo's diverse Limitless moveset. Useful adaptation precedent, not evidence for our numerical mechanics.
- [Blue reference](https://jujutsu-kaisen.fandom.com/wiki/Cursed_Technique_Lapse%3A_Blue), [Red reference](https://jujutsu-kaisen.fandom.com/wiki/Cursed_Technique_Reversal%3A_Red), [Purple reference](https://jujutsu-kaisen.fandom.com/wiki/Hollow_Technique%3A_Purple). Secondary fan-maintained references. Search-indexed excerpts support the attraction/repulsion/combination distinction; full-page retrieval was restricted during this research. They are not treated as primary manga text.

No claim is made that full licensed episodes or manga pages were watched/read in this session. The basic technique distinction is the supported basis for the design; disputed fan explanations are excluded. Crossed-finger controls are an interface shortcut, not a claim that every technique uses that exact ritual. Manga-only remote fusion, maximum-output variants, Domain Expansion, teleportation, and a 3D combat world are outside this build.

## Product boundary

This build uses camera hand tracking only: the left hand summons Blue, the right hand summons Red, and both hands merge them. After that merge, the right hand alone positions, aims, arms, and releases Purple through a crossed-then-uncrossed gesture. Blue and Red are not independently fired in this version. It is a local beta, not a fully certified public product. Before a public launch: test actual gestures on Chrome, Safari, Firefox, iOS Safari, and Android Chrome; test permission denial/timeout/device disconnect; validate low-end performance and HTTPS hosting; decide whether external model/font downloads are acceptable. There is no backend, account system, multiplayer, analytics, or recording.
