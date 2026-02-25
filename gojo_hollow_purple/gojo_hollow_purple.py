"""
Gojo Satoru — Hollow Purple (虚式「茈」) Hand Tracker
=====================================================
Real-time webcam hand tracking that replicates Gojo's Limitless techniques:
  Left hand  → Cursed Technique Lapse: Blue (蒼)
  Right hand → Cursed Technique Reversal: Red (赫)
  Hands merge → Hollow Technique: Purple (茈)
  Release    → Purple beam erases everything in its path

Requires: opencv-python, mediapipe, numpy
Usage:    python gojo_hollow_purple.py
Controls: Q / ESC to quit
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

# ─── Constants ───────────────────────────────────────────────────────────────

BLUE_CORE = (255, 180, 50)       # BGR — bright cyan-blue
BLUE_GLOW = (255, 130, 0)       # BGR — deep blue glow
RED_CORE = (60, 60, 255)        # BGR — bright red
RED_GLOW = (30, 30, 200)        # BGR — deep red glow
PURPLE_CORE = (255, 50, 200)    # BGR — vivid purple
PURPLE_GLOW = (200, 30, 160)    # BGR — deep purple
PURPLE_BEAM = (255, 100, 255)   # BGR — bright magenta-purple
WHITE = (255, 255, 255)

MERGE_DISTANCE_THRESHOLD = 120
CHARGE_TIME = 0.6
SHOOT_SEPARATION_THRESHOLD = 200
BEAM_DURATION = 1.8
COOLDOWN_DURATION = 0.8
PARTICLE_COUNT_BLUE = 150
PARTICLE_COUNT_RED = 150
PARTICLE_COUNT_PURPLE = 300
BEAM_PARTICLES = 500


# ─── Enums ───────────────────────────────────────────────────────────────────

class TechniqueState(Enum):
    IDLE = auto()
    BLUE_ONLY = auto()
    RED_ONLY = auto()
    BOTH_ACTIVE = auto()
    MERGING = auto()
    CHARGING = auto()
    SHOOTING = auto()
    COOLDOWN = auto()


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    lifetime: float
    max_lifetime: float
    color: Tuple[int, int, int]
    size: float
    alpha: float = 1.0
    trail: deque = field(default_factory=lambda: deque(maxlen=6))

    def update(self, dt: float):
        self.trail.append((self.x, self.y))
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        self.alpha = max(0, self.lifetime / self.max_lifetime)
        self.size = max(0.5, self.size * (0.97 + 0.03 * self.alpha))

    @property
    def alive(self) -> bool:
        return self.lifetime > 0


@dataclass
class BeamSegment:
    x: float
    y: float
    radius: float
    alpha: float
    speed: float
    angle: float


# ─── Particle System ────────────────────────────────────────────────────────

class ParticleSystem:
    def __init__(self):
        self.particles: List[Particle] = []

    def emit_blue_inward(self, cx: float, cy: float, count: int = 8):
        """Blue (蒼) — attraction: particles spiral inward toward palm."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(40, 100)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist
            speed = random.uniform(60, 140)
            inward_angle = angle + math.pi + random.uniform(-0.5, 0.5)
            vx = math.cos(inward_angle) * speed
            vy = math.sin(inward_angle) * speed
            lifetime = random.uniform(0.3, 0.8)
            size = random.uniform(1.5, 4.0)
            shade = random.choice([BLUE_CORE, BLUE_GLOW, (255, 200, 100), WHITE])
            self.particles.append(Particle(px, py, vx, vy, lifetime, lifetime, shade, size))

    def emit_red_outward(self, cx: float, cy: float, count: int = 8):
        """Red (赫) — repulsion: particles burst outward from palm."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(80, 200)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.uniform(0.3, 0.7)
            size = random.uniform(1.5, 4.5)
            shade = random.choice([RED_CORE, RED_GLOW, (50, 100, 255), WHITE])
            self.particles.append(Particle(cx, cy, vx, vy, lifetime, lifetime, shade, size))

    def emit_purple_converge(self, cx: float, cy: float, count: int = 15):
        """Purple (茈) — convergence: chaotic energy collapsing inward."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(50, 150)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist
            speed = random.uniform(100, 250)
            inward = angle + math.pi + random.uniform(-0.3, 0.3)
            vx = math.cos(inward) * speed
            vy = math.sin(inward) * speed
            lifetime = random.uniform(0.2, 0.6)
            size = random.uniform(2.0, 5.0)
            shade = random.choice([PURPLE_CORE, PURPLE_GLOW, PURPLE_BEAM, WHITE, (255, 150, 255)])
            self.particles.append(Particle(px, py, vx, vy, lifetime, lifetime, shade, size))

    def emit_beam_trail(self, x: float, y: float, angle: float, count: int = 20):
        """Beam trail particles shooting in a direction."""
        for _ in range(count):
            spread = random.uniform(-0.4, 0.4)
            a = angle + spread
            speed = random.uniform(300, 800)
            vx = math.cos(a) * speed
            vy = math.sin(a) * speed
            lifetime = random.uniform(0.3, 1.0)
            size = random.uniform(2, 7)
            shade = random.choice([PURPLE_CORE, PURPLE_BEAM, WHITE, (255, 180, 255)])
            self.particles.append(Particle(x, y, vx, vy, lifetime, lifetime, shade, size))

    def emit_destruction_burst(self, cx: float, cy: float, count: int = 100):
        """Massive burst when beam fires."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(100, 600)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.uniform(0.5, 1.5)
            size = random.uniform(2, 8)
            shade = random.choice([PURPLE_CORE, PURPLE_BEAM, WHITE, (200, 200, 255)])
            self.particles.append(Particle(cx, cy, vx, vy, lifetime, lifetime, shade, size))

    def update(self, dt: float):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, overlay: np.ndarray):
        for p in self.particles:
            if p.alpha <= 0:
                continue
            ix, iy = int(p.x), int(p.y)
            h, w = overlay.shape[:2]
            if 0 <= ix < w and 0 <= iy < h:
                r = max(1, int(p.size * p.alpha))
                color = tuple(int(c * p.alpha) for c in p.color)
                cv2.circle(overlay, (ix, iy), r, color, -1)
                for i, (tx, ty) in enumerate(p.trail):
                    trail_alpha = (i / len(p.trail)) * p.alpha * 0.4
                    tr = max(1, int(r * trail_alpha))
                    tc = tuple(int(c * trail_alpha) for c in p.color)
                    cv2.circle(overlay, (int(tx), int(ty)), tr, tc, -1)


# ─── Glow Renderer ──────────────────────────────────────────────────────────

class GlowRenderer:
    """Renders glow effects by drawing on a black layer, blurring, and compositing."""

    @staticmethod
    def draw_orb(overlay: np.ndarray, cx: int, cy: int, radius: int,
                 core_color: Tuple, glow_color: Tuple, pulse: float):
        """Draw a pulsing energy orb with multi-layer glow."""
        r = int(radius * (1.0 + 0.2 * math.sin(pulse * 6)))

        for i in range(5, 0, -1):
            layer_r = r + i * 12
            alpha = 0.15 * (1 - i / 6)
            color = tuple(int(c * alpha) for c in glow_color)
            cv2.circle(overlay, (cx, cy), layer_r, color, -1)

        cv2.circle(overlay, (cx, cy), r + 6, glow_color, -1)
        cv2.circle(overlay, (cx, cy), r, core_color, -1)
        cv2.circle(overlay, (cx, cy), max(1, r // 2), WHITE, -1)

    @staticmethod
    def draw_energy_ring(overlay: np.ndarray, cx: int, cy: int, radius: int,
                         color: Tuple, pulse: float, thickness: int = 2):
        """Rotating energy ring around an orb."""
        r = int(radius * (1.0 + 0.15 * math.sin(pulse * 4)))
        start_angle = int(pulse * 120) % 360
        cv2.ellipse(overlay, (cx, cy), (r, r), 0, start_angle, start_angle + 270,
                     color, thickness)
        cv2.ellipse(overlay, (cx, cy), (r + 8, r + 8), 0, start_angle + 90,
                     start_angle + 200, color, max(1, thickness - 1))

    @staticmethod
    def draw_beam(overlay: np.ndarray, start_x: int, start_y: int,
                  angle: float, length: float, progress: float,
                  width: int = 30):
        """Draw the Hollow Purple beam."""
        current_len = length * min(1.0, progress * 2.5)
        end_x = int(start_x + math.cos(angle) * current_len)
        end_y = int(start_y + math.sin(angle) * current_len)

        for i in range(6, 0, -1):
            w = width + i * 10
            alpha = 0.12 * (1 - i / 7)
            color = tuple(int(c * alpha) for c in PURPLE_GLOW)
            cv2.line(overlay, (start_x, start_y), (end_x, end_y), color, w)

        cv2.line(overlay, (start_x, start_y), (end_x, end_y), PURPLE_BEAM, width)
        cv2.line(overlay, (start_x, start_y), (end_x, end_y), PURPLE_CORE, width // 2)
        cv2.line(overlay, (start_x, start_y), (end_x, end_y), WHITE, max(1, width // 5))

        tip_r = int(width * 1.5 * (0.8 + 0.2 * math.sin(progress * 20)))
        cv2.circle(overlay, (end_x, end_y), tip_r, PURPLE_BEAM, -1)
        cv2.circle(overlay, (end_x, end_y), tip_r // 2, WHITE, -1)

        return end_x, end_y

    @staticmethod
    def apply_glow(frame: np.ndarray, overlay: np.ndarray,
                   blur_size: int = 51) -> np.ndarray:
        """Apply glow by blurring the overlay and additively blending."""
        blurred = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
        extra_blur = cv2.GaussianBlur(overlay, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
        result = cv2.add(frame, blurred)
        result = cv2.add(result, extra_blur)
        result = cv2.add(result, overlay)
        return result


# ─── Hand Analyzer ───────────────────────────────────────────────────────────

class HandAnalyzer:
    """Extracts meaningful data from MediaPipe hand landmarks."""

    @staticmethod
    def get_palm_center(landmarks, w: int, h: int) -> Tuple[int, int]:
        indices = [0, 5, 9, 13, 17]
        cx = int(sum(landmarks[i].x for i in indices) / len(indices) * w)
        cy = int(sum(landmarks[i].y for i in indices) / len(indices) * h)
        return cx, cy

    @staticmethod
    def get_palm_size(landmarks, w: int, h: int) -> float:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        dx = (wrist.x - middle_mcp.x) * w
        dy = (wrist.y - middle_mcp.y) * h
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def is_hand_open(landmarks) -> bool:
        """Check if fingers are extended (open hand)."""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = sum(1 for t, p in zip(tips, pips)
                       if landmarks[t].y < landmarks[p].y)
        return extended >= 3

    @staticmethod
    def get_finger_tips(landmarks, w: int, h: int) -> List[Tuple[int, int]]:
        tip_indices = [4, 8, 12, 16, 20]
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in tip_indices]

    @staticmethod
    def get_wrist(landmarks, w: int, h: int) -> Tuple[int, int]:
        return (int(landmarks[0].x * w), int(landmarks[0].y * h))


# ─── HUD Overlay ─────────────────────────────────────────────────────────────

class HUD:
    """Heads-up display with technique names and state info."""

    @staticmethod
    def draw(frame: np.ndarray, state: TechniqueState, charge_pct: float = 0):
        h, w = frame.shape[:2]

        technique_text = {
            TechniqueState.IDLE: "",
            TechniqueState.BLUE_ONLY: "Cursed Technique Lapse: BLUE",
            TechniqueState.RED_ONLY: "Cursed Technique Reversal: RED",
            TechniqueState.BOTH_ACTIVE: "Limitless Active",
            TechniqueState.MERGING: "Converging...",
            TechniqueState.CHARGING: "Hollow Purple Charging...",
            TechniqueState.SHOOTING: "HOLLOW TECHNIQUE: PURPLE",
            TechniqueState.COOLDOWN: "",
        }

        jp_text = {
            TechniqueState.IDLE: "",
            TechniqueState.BLUE_ONLY: "術式順転「蒼」",
            TechniqueState.RED_ONLY: "術式反転「赫」",
            TechniqueState.BOTH_ACTIVE: "無下限呪術",
            TechniqueState.MERGING: "虚式...",
            TechniqueState.CHARGING: "虚式「茈」",
            TechniqueState.SHOOTING: "虚式「茈」",
            TechniqueState.COOLDOWN: "",
        }

        text = technique_text.get(state, "")
        jp = jp_text.get(state, "")

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX

            if state == TechniqueState.SHOOTING:
                scale, thickness = 1.2, 3
                color = PURPLE_BEAM
            elif state == TechniqueState.CHARGING:
                scale, thickness = 0.9, 2
                color = PURPLE_CORE
            elif state == TechniqueState.BLUE_ONLY:
                scale, thickness = 0.7, 2
                color = BLUE_CORE
            elif state == TechniqueState.RED_ONLY:
                scale, thickness = 0.7, 2
                color = RED_CORE
            else:
                scale, thickness = 0.7, 2
                color = WHITE

            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            tx = (w - text_size[0]) // 2
            ty = 50

            cv2.putText(frame, text, (tx + 2, ty + 2), font, scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (tx, ty), font, scale, color, thickness)

            if jp:
                jp_size = cv2.getTextSize(jp, font, scale * 0.7, thickness)[0]
                jx = (w - jp_size[0]) // 2
                cv2.putText(frame, jp, (jx, ty + 40), font, scale * 0.7, color, thickness - 1)

        if state == TechniqueState.CHARGING and charge_pct > 0:
            bar_w = 300
            bar_h = 8
            bx = (w - bar_w) // 2
            by = h - 60
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            fill = int(bar_w * min(1.0, charge_pct))
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), PURPLE_CORE, -1)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), PURPLE_BEAM, 1)

        HUD._draw_corner_marks(frame, w, h, state)

    @staticmethod
    def _draw_corner_marks(frame, w, h, state):
        if state == TechniqueState.IDLE:
            return
        color = PURPLE_CORE if state in (TechniqueState.CHARGING, TechniqueState.SHOOTING) else (100, 100, 100)
        length = 30
        t = 2
        cv2.line(frame, (10, 10), (10 + length, 10), color, t)
        cv2.line(frame, (10, 10), (10, 10 + length), color, t)
        cv2.line(frame, (w - 10, 10), (w - 10 - length, 10), color, t)
        cv2.line(frame, (w - 10, 10), (w - 10, 10 + length), color, t)
        cv2.line(frame, (10, h - 10), (10 + length, h - 10), color, t)
        cv2.line(frame, (10, h - 10), (10, h - 10 - length), color, t)
        cv2.line(frame, (w - 10, h - 10), (w - 10 - length, h - 10), color, t)
        cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - length), color, t)


# ─── Screen Effects ──────────────────────────────────────────────────────────

class ScreenEffects:
    """Full-screen effects: flash, shake, vignette."""

    @staticmethod
    def flash(frame: np.ndarray, intensity: float, color: Tuple = WHITE) -> np.ndarray:
        if intensity <= 0:
            return frame
        flash_overlay = np.full_like(frame, color, dtype=np.uint8)
        alpha = min(1.0, intensity)
        return cv2.addWeighted(frame, 1.0 - alpha * 0.7, flash_overlay, alpha * 0.7, 0)

    @staticmethod
    def shake(frame: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0:
            return frame
        max_offset = int(intensity * 15)
        ox = random.randint(-max_offset, max_offset)
        oy = random.randint(-max_offset, max_offset)
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    @staticmethod
    def vignette(frame: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = math.sqrt(cx ** 2 + cy ** 2)
        mask = 1.0 - intensity * (dist / max_dist) ** 2
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return (frame * mask[:, :, np.newaxis]).astype(np.uint8)

    @staticmethod
    def distortion_lines(overlay: np.ndarray, cx: int, cy: int,
                         radius: int, count: int = 12, pulse: float = 0):
        """Energy distortion lines radiating from a point."""
        for i in range(count):
            angle = (2 * math.pi * i / count) + pulse * 2
            inner_r = radius + 5
            outer_r = radius + 20 + int(10 * math.sin(pulse * 8 + i))
            x1 = int(cx + math.cos(angle) * inner_r)
            y1 = int(cy + math.sin(angle) * inner_r)
            x2 = int(cx + math.cos(angle) * outer_r)
            y2 = int(cy + math.sin(angle) * outer_r)
            cv2.line(overlay, (x1, y1), (x2, y2), WHITE, 1)


# ─── Finger Energy Threads ──────────────────────────────────────────────────

class FingerEnergyThreads:
    """Draw energy arcs from fingertips toward palm — anime-style cursed energy flow."""

    @staticmethod
    def draw(overlay: np.ndarray, palm: Tuple[int, int],
             tips: List[Tuple[int, int]], color: Tuple, pulse: float):
        for i, (tx, ty) in enumerate(tips):
            px, py = palm
            mid_x = (px + tx) // 2 + int(8 * math.sin(pulse * 6 + i * 1.5))
            mid_y = (py + ty) // 2 + int(8 * math.cos(pulse * 5 + i * 1.2))

            pts = []
            steps = 10
            for s in range(steps + 1):
                t = s / steps
                x = int((1 - t) ** 2 * tx + 2 * (1 - t) * t * mid_x + t ** 2 * px)
                y = int((1 - t) ** 2 * ty + 2 * (1 - t) * t * mid_y + t ** 2 * py)
                pts.append((x, y))

            for j in range(len(pts) - 1):
                alpha = 0.3 + 0.7 * (j / len(pts))
                c = tuple(int(v * alpha) for v in color)
                cv2.line(overlay, pts[j], pts[j + 1], c, 1)

            glow_r = int(3 + 2 * math.sin(pulse * 8 + i))
            cv2.circle(overlay, (tx, ty), glow_r, color, -1)


# ─── Main Application ───────────────────────────────────────────────────────

class HollowPurpleApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self.particles = ParticleSystem()
        self.glow = GlowRenderer()
        self.hud = HUD()
        self.screen_fx = ScreenEffects()
        self.hand_analyzer = HandAnalyzer()
        self.finger_threads = FingerEnergyThreads()

        self.state = TechniqueState.IDLE
        self.prev_time = time.time()
        self.pulse = 0.0

        self.left_palm: Optional[Tuple[int, int]] = None
        self.right_palm: Optional[Tuple[int, int]] = None
        self.left_tips: List[Tuple[int, int]] = []
        self.right_tips: List[Tuple[int, int]] = []
        self.left_open = False
        self.right_open = False
        self.left_palm_size = 0
        self.right_palm_size = 0

        self.merge_start_time = 0.0
        self.charge_progress = 0.0
        self.shoot_start_time = 0.0
        self.beam_angle = 0.0
        self.beam_origin = (0, 0)
        self.cooldown_start = 0.0

        self.flash_intensity = 0.0
        self.shake_intensity = 0.0

        self.merge_center = (0, 0)
        self.purple_orb_radius = 0

        self.frame_count = 0

    def _process_hands(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self.left_palm = None
        self.right_palm = None
        self.left_tips = []
        self.right_tips = []
        self.left_open = False
        self.right_open = False

        if not results.multi_hand_landmarks:
            return

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            lm = hand_landmarks.landmark
            label = handedness.classification[0].label

            palm = self.hand_analyzer.get_palm_center(lm, w, h)
            tips = self.hand_analyzer.get_finger_tips(lm, w, h)
            is_open = self.hand_analyzer.is_hand_open(lm)
            palm_size = self.hand_analyzer.get_palm_size(lm, w, h)

            # MediaPipe mirrors: "Left" in results = user's left hand (mirrored view)
            if label == "Right":  # mirrored → user's left hand
                self.left_palm = palm
                self.left_tips = tips
                self.left_open = is_open
                self.left_palm_size = palm_size
            else:  # "Left" in results → user's right hand
                self.right_palm = palm
                self.right_tips = tips
                self.right_open = is_open
                self.right_palm_size = palm_size

    def _get_hand_distance(self) -> float:
        if self.left_palm and self.right_palm:
            dx = self.left_palm[0] - self.right_palm[0]
            dy = self.left_palm[1] - self.right_palm[1]
            return math.sqrt(dx * dx + dy * dy)
        return float('inf')

    def _update_state(self, dt: float):
        dist = self._get_hand_distance()
        both_detected = self.left_palm is not None and self.right_palm is not None
        left_active = self.left_palm is not None and self.left_open
        right_active = self.right_palm is not None and self.right_open

        if self.state == TechniqueState.IDLE:
            if both_detected and left_active and right_active:
                self.state = TechniqueState.BOTH_ACTIVE
            elif left_active:
                self.state = TechniqueState.BLUE_ONLY
            elif right_active:
                self.state = TechniqueState.RED_ONLY

        elif self.state == TechniqueState.BLUE_ONLY:
            if both_detected and right_active:
                self.state = TechniqueState.BOTH_ACTIVE
            elif not left_active:
                self.state = TechniqueState.IDLE

        elif self.state == TechniqueState.RED_ONLY:
            if both_detected and left_active:
                self.state = TechniqueState.BOTH_ACTIVE
            elif not right_active:
                self.state = TechniqueState.IDLE

        elif self.state == TechniqueState.BOTH_ACTIVE:
            if not both_detected:
                if left_active:
                    self.state = TechniqueState.BLUE_ONLY
                elif right_active:
                    self.state = TechniqueState.RED_ONLY
                else:
                    self.state = TechniqueState.IDLE
            elif dist < MERGE_DISTANCE_THRESHOLD:
                self.state = TechniqueState.MERGING
                self.merge_start_time = time.time()

        elif self.state == TechniqueState.MERGING:
            if not both_detected:
                self.state = TechniqueState.IDLE
            elif dist > MERGE_DISTANCE_THRESHOLD * 1.3:
                self.state = TechniqueState.BOTH_ACTIVE
            else:
                elapsed = time.time() - self.merge_start_time
                if elapsed >= CHARGE_TIME * 0.3:
                    self.state = TechniqueState.CHARGING
                    self.charge_progress = 0.0

        elif self.state == TechniqueState.CHARGING:
            self.charge_progress += dt / CHARGE_TIME
            if not both_detected:
                if self.charge_progress >= 0.8:
                    self._fire_beam()
                else:
                    self.state = TechniqueState.IDLE
                    self.charge_progress = 0
            elif dist > SHOOT_SEPARATION_THRESHOLD and self.charge_progress >= 0.5:
                self._fire_beam()
            elif dist > MERGE_DISTANCE_THRESHOLD * 2 and self.charge_progress < 0.5:
                self.state = TechniqueState.BOTH_ACTIVE
                self.charge_progress = 0

        elif self.state == TechniqueState.SHOOTING:
            elapsed = time.time() - self.shoot_start_time
            if elapsed >= BEAM_DURATION:
                self.state = TechniqueState.COOLDOWN
                self.cooldown_start = time.time()

        elif self.state == TechniqueState.COOLDOWN:
            elapsed = time.time() - self.cooldown_start
            if elapsed >= COOLDOWN_DURATION:
                self.state = TechniqueState.IDLE

    def _fire_beam(self):
        self.state = TechniqueState.SHOOTING
        self.shoot_start_time = time.time()
        self.flash_intensity = 1.0
        self.shake_intensity = 1.0

        if self.left_palm and self.right_palm:
            cx = (self.left_palm[0] + self.right_palm[0]) // 2
            cy = (self.left_palm[1] + self.right_palm[1]) // 2
            self.beam_origin = (cx, cy)
            self.beam_angle = 0
        else:
            self.beam_origin = self.merge_center
            self.beam_angle = 0

        self.particles.emit_destruction_burst(self.beam_origin[0], self.beam_origin[1], 200)

    def _emit_particles(self, dt: float):
        if self.state in (TechniqueState.BLUE_ONLY, TechniqueState.BOTH_ACTIVE,
                          TechniqueState.MERGING, TechniqueState.CHARGING):
            if self.left_palm:
                self.particles.emit_blue_inward(self.left_palm[0], self.left_palm[1],
                                                int(8 + self.pulse % 3))

        if self.state in (TechniqueState.RED_ONLY, TechniqueState.BOTH_ACTIVE,
                          TechniqueState.MERGING, TechniqueState.CHARGING):
            if self.right_palm:
                self.particles.emit_red_outward(self.right_palm[0], self.right_palm[1],
                                                int(8 + self.pulse % 3))

        if self.state in (TechniqueState.MERGING, TechniqueState.CHARGING):
            if self.left_palm and self.right_palm:
                cx = (self.left_palm[0] + self.right_palm[0]) // 2
                cy = (self.left_palm[1] + self.right_palm[1]) // 2
                self.merge_center = (cx, cy)
                count = int(15 + self.charge_progress * 25)
                self.particles.emit_purple_converge(cx, cy, count)

        if self.state == TechniqueState.SHOOTING:
            elapsed = time.time() - self.shoot_start_time
            progress = elapsed / BEAM_DURATION
            if progress < 0.8:
                bx = self.beam_origin[0]
                by = self.beam_origin[1]
                self.particles.emit_beam_trail(bx, by, self.beam_angle, 15)

    def _render_effects(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)

        if self.state in (TechniqueState.BLUE_ONLY, TechniqueState.BOTH_ACTIVE,
                          TechniqueState.MERGING, TechniqueState.CHARGING):
            if self.left_palm:
                r = int(self.left_palm_size * 0.4)
                self.glow.draw_orb(overlay, self.left_palm[0], self.left_palm[1],
                                   r, BLUE_CORE, BLUE_GLOW, self.pulse)
                self.glow.draw_energy_ring(overlay, self.left_palm[0], self.left_palm[1],
                                           r + 15, BLUE_CORE, self.pulse, 2)
                self.finger_threads.draw(overlay, self.left_palm, self.left_tips,
                                         BLUE_CORE, self.pulse)
                ScreenEffects.distortion_lines(overlay, self.left_palm[0], self.left_palm[1],
                                               r + 20, 10, self.pulse)

        if self.state in (TechniqueState.RED_ONLY, TechniqueState.BOTH_ACTIVE,
                          TechniqueState.MERGING, TechniqueState.CHARGING):
            if self.right_palm:
                r = int(self.right_palm_size * 0.4)
                self.glow.draw_orb(overlay, self.right_palm[0], self.right_palm[1],
                                   r, RED_CORE, RED_GLOW, self.pulse)
                self.glow.draw_energy_ring(overlay, self.right_palm[0], self.right_palm[1],
                                           r + 15, RED_CORE, self.pulse + 1, 2)
                self.finger_threads.draw(overlay, self.right_palm, self.right_tips,
                                         RED_CORE, self.pulse)
                ScreenEffects.distortion_lines(overlay, self.right_palm[0], self.right_palm[1],
                                               r + 20, 10, self.pulse + 0.5)

        if self.state == TechniqueState.MERGING:
            cx, cy = self.merge_center
            r = 20
            self.glow.draw_orb(overlay, cx, cy, r, PURPLE_CORE, PURPLE_GLOW, self.pulse)

        if self.state == TechniqueState.CHARGING:
            cx, cy = self.merge_center
            r = int(20 + 30 * self.charge_progress)
            self.purple_orb_radius = r
            self.glow.draw_orb(overlay, cx, cy, r, PURPLE_CORE, PURPLE_GLOW, self.pulse)
            self.glow.draw_energy_ring(overlay, cx, cy, r + 20, PURPLE_BEAM,
                                       self.pulse * 2, 3)
            self.glow.draw_energy_ring(overlay, cx, cy, r + 35, WHITE,
                                       -self.pulse * 1.5, 1)
            ScreenEffects.distortion_lines(overlay, cx, cy, r + 25, 16, self.pulse)

            if self.charge_progress > 0.6:
                for i in range(3):
                    angle = self.pulse * 3 + i * 2.09
                    arc_r = r + 40 + int(10 * math.sin(self.pulse * 6))
                    ax = int(cx + math.cos(angle) * arc_r)
                    ay = int(cy + math.sin(angle) * arc_r)
                    cv2.circle(overlay, (ax, ay), 4, WHITE, -1)

        if self.state == TechniqueState.SHOOTING:
            elapsed = time.time() - self.shoot_start_time
            progress = elapsed / BEAM_DURATION
            beam_len = math.sqrt(w * w + h * h)
            end_x, end_y = self.glow.draw_beam(
                overlay, self.beam_origin[0], self.beam_origin[1],
                self.beam_angle, beam_len, progress,
                width=int(25 + 15 * (1 - progress))
            )

        self.particles.draw(overlay)

        result = self.glow.apply_glow(frame, overlay)

        if self.flash_intensity > 0:
            result = self.screen_fx.flash(result, self.flash_intensity, PURPLE_BEAM)
            self.flash_intensity *= 0.88

        if self.shake_intensity > 0:
            result = self.screen_fx.shake(result, self.shake_intensity)
            self.shake_intensity *= 0.92

        if self.state != TechniqueState.IDLE:
            result = self.screen_fx.vignette(result, 0.3)

        return result

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("ERROR: Cannot open camera.")
            return

        print("=" * 55)
        print("  GOJO SATORU — HOLLOW PURPLE HAND TRACKER")
        print("  無下限呪術 — 虚式「茈」")
        print("=" * 55)
        print()
        print("  Left hand open  → Blue  (蒼) — Attraction")
        print("  Right hand open → Red   (赫) — Repulsion")
        print("  Hands together  → Purple(茈) — Annihilation")
        print("  Separate hands  → FIRE BEAM!")
        print()
        print("  Press Q or ESC to quit")
        print("=" * 55)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            self.pulse += dt
            self.frame_count += 1

            self._process_hands(frame)

            self._update_state(dt)

            self._emit_particles(dt)
            self.particles.update(dt)

            result = self._render_effects(frame)

            charge_pct = self.charge_progress if self.state == TechniqueState.CHARGING else 0
            self.hud.draw(result, self.state, charge_pct)

            fps = 1.0 / max(dt, 0.001)
            cv2.putText(result, f"FPS: {int(fps)}", (10, result.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Gojo Satoru - Hollow Purple", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HollowPurpleApp()
    app.run()
