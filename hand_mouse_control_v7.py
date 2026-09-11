"""
Air Mouse v7 — Hand Gesture Mouse Control (Windows)

Gesture set:

  POINTER        Index extended, middle curled -> moves the cursor
                 continuously (thumb/pinky/ring don't affect this).

  LEFT CLICK     While pointing, curl your THUMB down (a "trigger pull" —
                 extend it back out to re-arm for the next click). Two
                 clicks within 0.35s -> double click.

  RIGHT CLICK    While pointing, curl your PINKY down (same trigger-pull
                 motion, re-arm by extending it back out).

  SCROLL UP      Index + middle both extended ("scissors"/peace shape).
                 Scrolls up continuously while held.

  SCROLL DOWN    Index + middle both curled down. Scrolls down
                 continuously while held.

  ZOOM           Pinch thumb + index together to a point (other fingers
                 curled) to arm zoom mode, then spread them apart ->
                 zoom in; bring them back together -> zoom out. Standard
                 Ctrl+Scroll zoom, so it works in browsers, image
                 viewers, PDF readers, code editors, etc. Ends when you
                 open your hand back up.

  MINIMIZE       Full fist — all 5 fingers curled AND fingertips pulled
                 together to a point. Hold ~0.8s, wrist roughly still.

  ON-SCREEN KB   Draw a full circle in the air with your index finger
                 (either direction) while pointing, ending back near
                 where you started.

  Corner dwell   Hover the real screen cursor in the top-right corner
                 ~1.4s -> minimize. Top-left corner ~1.4s -> close window
                 (Alt+F4).

Cursor movement and click detection are fully decoupled: the cursor
tracks your index fingertip any time index is extended and middle is
curled, regardless of what your thumb/pinky are doing. Click gestures are
layered independently on top using an "arm/disarm" trigger-pull pattern —
a click fires once when the finger curls down, and won't fire again until
it's been extended back out, so holding the pose doesn't spam clicks.

Finger extension is detected using the BEND ANGLE at each finger's middle
knuckle (PIP joint), which is orientation-independent — it works whether
your palm faces the camera, faces away, or is rotated to the side.

Dependencies: opencv-python, mediapipe, pyautogui, numpy
Install:      pip install opencv-python mediapipe==0.10.14 pyautogui numpy
Run:          py hand_mouse_control_v7.py
Quit:         focus the preview window and press 'q'
"""

import time
import math
import os
import subprocess
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MODE_SCROLL_UP = "SCROLL_UP"
MODE_SCROLL_DOWN = "SCROLL_DOWN"


class HandMouseController:
    def __init__(
        self,
        cam_index=0,
        cam_width=1280,
        cam_height=720,
        margin_fraction=0.10,
        smooth_fast=2,
        smooth_slow=5,
        fast_speed_threshold=25,
        cursor_gain=1.25,            # amplifies mapped cursor movement around screen center for a faster feel
        detection_confidence=0.5,    # lower = detects the hand more readily, incl. nearer frame edges
        tracking_confidence=0.5,
        extend_angle_threshold=150,  # degrees; finger counts as "extended" above this bend angle
        fist_cluster_ratio=0.30,     # fingertip-cluster threshold (fraction of palm length) for minimize
        fist_hold_time=0.8,
        fist_stability_frames=5,
        wrist_velocity_limit=18,     # px/frame; wrist must be roughly still for the minimize gesture
        corner_size_frac=0.06,
        corner_hold_time=1.4,
        scroll_step=35,              # pixels scrolled per tick
        scroll_tick_interval=0.09,   # seconds between scroll ticks while a scroll pose is held (medium pace)
        double_click_gap=0.35,
        click_cooldown=0.35,
        click_confirm_frames=2,      # consecutive frames a click pose must hold before it fires (filters single-frame jitter, stays responsive)
        zoom_engage_ratio=0.20,      # thumb-index pinch distance (fraction of palm length) below which zoom mode arms — tightened so a normal click's thumb curl can't be mistaken for a pinch
        zoom_engage_frames=6,        # consecutive frames the pinch must hold before zoom mode actually engages — raised so brief incidental proximity during clicking doesn't trigger it
        zoom_deadzone=0.025,         # min change in normalized pinch distance per tick to count as spreading/pinching
        zoom_tick_interval=0.08,
        action_cooldown=2.0,
    ):
        self.cam_index = cam_index
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.margin_fraction = margin_fraction
        self.smooth_fast = smooth_fast
        self.smooth_slow = smooth_slow
        self.fast_speed_threshold = fast_speed_threshold
        self.cursor_gain = cursor_gain
        self.extend_angle_threshold = extend_angle_threshold
        self.fist_cluster_ratio = fist_cluster_ratio
        self.fist_hold_time = fist_hold_time
        self.fist_stability_frames = fist_stability_frames
        self.wrist_velocity_limit = wrist_velocity_limit
        self.corner_hold_time = corner_hold_time
        self.scroll_step = scroll_step
        self.scroll_tick_interval = scroll_tick_interval
        self.double_click_gap = double_click_gap
        self.click_cooldown = click_cooldown
        self.click_confirm_frames = click_confirm_frames
        self.zoom_engage_ratio = zoom_engage_ratio
        self.zoom_engage_frames = zoom_engage_frames
        self.zoom_deadzone = zoom_deadzone
        self.zoom_tick_interval = zoom_tick_interval
        self.action_cooldown = action_cooldown

        self.screen_w, self.screen_h = pyautogui.size()
        self.corner_size = int(self.screen_w * corner_size_frac)

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_hands = mp_hands
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        # (mcp, pip, tip) joint ids for the 4 non-thumb fingers, used for bend-angle checks
        self.finger_joints = {
            "index": (5, 6, 8),
            "middle": (9, 10, 12),
            "ring": (13, 14, 16),
            "pinky": (17, 18, 20),
        }

        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2

        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.thumb_curl_streak = 0
        self.pinky_curl_streak = 0
        self.left_click_armed = True
        self.right_click_armed = True

        self.fist_start_time = None
        self.fist_frame_streak = 0
        self.prev_wrist_pos = None
        self.last_action_time = 0.0

        self.last_scroll_tick = 0.0

        self.zoom_active = False
        self.zoom_engage_streak = 0
        self.zoom_prev_dist = None
        self.last_zoom_tick = 0.0

        self.trail = deque(maxlen=70)
        self.last_circle_time = 0.0

        self.corner_hover_start = None
        self.corner_hover_which = None

    # ------------------------------------------------------------------
    @staticmethod
    def distance(p1, p2):
        return math.hypot(p2[1] - p1[1], p2[2] - p1[2])

    def palm_length(self, lm_list):
        return max(self.distance(lm_list[0], lm_list[9]), 1e-6)

    def finger_extended(self, lm_list, mcp_id, pip_id, tip_id):
        """True if the finger is roughly straight (bend angle at the PIP
        joint is close to 180 degrees); False if bent/curled/hooked. This
        is orientation-independent, unlike comparing raw pixel positions."""
        p_mcp, p_pip, p_tip = lm_list[mcp_id], lm_list[pip_id], lm_list[tip_id]
        v1 = (p_mcp[1] - p_pip[1], p_mcp[2] - p_pip[2])
        v2 = (p_tip[1] - p_pip[1], p_tip[2] - p_pip[2])
        mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
        if mag1 < 1e-3 or mag2 < 1e-3:
            return True
        cos_ang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)))
        angle_deg = math.degrees(math.acos(cos_ang))
        return angle_deg > self.extend_angle_threshold

    def thumb_extended(self, lm_list, palm_len):
        # Thumb: extended if held away from the palm; curled if pulled in
        # close to the index finger's base knuckle. Lowered from the
        # original 0.55 — at that threshold, many people's natural
        # "pointing" hand never actually got the thumb far enough out to
        # count as "extended," so the left-click gesture could never
        # re-arm after the first click.
        return self.distance(lm_list[4], lm_list[5]) / palm_len >= 0.42

    def _end_zoom(self):
        if self.zoom_active:
            try:
                pyautogui.keyUp("ctrl")
            except Exception:
                pass
        self.zoom_active = False
        self.zoom_engage_streak = 0
        self.zoom_prev_dist = None

    # ------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check cam_index or camera permissions.")

        window_name = "Air Mouse v7 - press 'q' to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1100, 620)

        prev_frame_time = time.time()

        while True:
            ok, img = cap.read()
            if not ok:
                break

            img = cv2.flip(img, 1)
            img_h, img_w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            mx = int(img_w * self.margin_fraction)
            my = int(img_h * self.margin_fraction)
            cv2.rectangle(img, (mx, my), (img_w - mx, img_h - my), (255, 0, 255), 2)

            status_text, status_color = "No hand detected", (150, 150, 150)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lm_list = [
                    [i, int(lm.x * img_w), int(lm.y * img_h)]
                    for i, lm in enumerate(hand_landmarks.landmark)
                ]
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                status_text, status_color = self._process_gestures(lm_list, img, img_w, img_h, mx, my)
            else:
                self.fist_start_time = None
                self.fist_frame_streak = 0
                self.prev_wrist_pos = None
                self.thumb_curl_streak = 0
                self.pinky_curl_streak = 0
                self.left_click_armed = True
                self.right_click_armed = True
                self._end_zoom()
                self.trail.clear()

            self._process_corner_dwell(img)

            cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

            now = time.time()
            fps = 1 / max(now - prev_frame_time, 1e-6)
            prev_frame_time = now
            cv2.putText(img, f"FPS: {int(fps)}", (20, img_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._end_zoom()

    # ------------------------------------------------------------------
    def _process_gestures(self, lm_list, img, img_w, img_h, mx, my):
        palm_len = self.palm_length(lm_list)

        idx_ext = self.finger_extended(lm_list, *self.finger_joints["index"])
        mid_ext = self.finger_extended(lm_list, *self.finger_joints["middle"])
        ring_ext = self.finger_extended(lm_list, *self.finger_joints["ring"])
        pinky_ext = self.finger_extended(lm_list, *self.finger_joints["pinky"])
        thumb_ext = self.thumb_extended(lm_list, palm_len)

        all_four_curled = not idx_ext and not mid_ext and not ring_ext and not pinky_ext

        # ---- FULL FIST (+ thumb curled, + clustered) -> minimize ----
        wrist_pos = lm_list[0]
        wrist_velocity = self.distance(wrist_pos, self.prev_wrist_pos) if self.prev_wrist_pos else 0
        self.prev_wrist_pos = wrist_pos

        is_all_curled = all_four_curled and not thumb_ext
        cluster_ok = False
        cx = cy = 0
        if is_all_curled:
            tips = [lm_list[i] for i in self.tip_ids]
            cx = sum(t[1] for t in tips) / 5
            cy = sum(t[2] for t in tips) / 5
            avg_spread = sum(math.hypot(t[1] - cx, t[2] - cy) for t in tips) / 5
            cluster_ok = avg_spread / palm_len < self.fist_cluster_ratio

        hand_is_still = wrist_velocity < self.wrist_velocity_limit
        fist_frame_ok = is_all_curled and cluster_ok and hand_is_still

        if fist_frame_ok:
            self.fist_frame_streak += 1
        else:
            self.fist_frame_streak = 0
            self.fist_start_time = None

        if self.fist_frame_streak >= self.fist_stability_frames:
            now = time.time()
            if self.fist_start_time is None:
                self.fist_start_time = now
            held = now - self.fist_start_time
            progress = min(held / self.fist_hold_time, 1.0)
            cv2.circle(img, (int(cx), int(cy)), int(30 + 20 * progress), (0, 165, 255), 3)
            if held >= self.fist_hold_time and now - self.last_action_time > self.action_cooldown:
                pyautogui.hotkey("win", "down")
                self.last_action_time = now
                self.fist_start_time = None
                self.fist_frame_streak = 0
                return "MINIMIZED (fist gesture)", (0, 165, 255)
            return f"Hold gathered fist to minimize... {int(progress * 100)}%", (0, 165, 255)

        # ---- ZOOM: thumb+index pinched to a point, other 3 curled ----
        # Checked before scroll/pointer since it uses the same general
        # finger shape (index extended-ish, others curled) but is
        # distinguished by requiring the thumb and index tips to actually
        # be touching. Once armed, it keeps tracking pinch distance (and
        # therefore stays active) as long as middle/ring/pinky stay curled,
        # even as the thumb/index bend while spreading apart.
        others_curled_for_zoom = not mid_ext and not ring_ext and not pinky_ext
        pinch_dist = self.distance(lm_list[4], lm_list[8]) / palm_len

        if self.zoom_active:
            if others_curled_for_zoom:
                now = time.time()
                delta = pinch_dist - self.zoom_prev_dist
                label, color = "ZOOM (hold)", (255, 140, 0)
                if now - self.last_zoom_tick >= self.zoom_tick_interval:
                    if delta > self.zoom_deadzone:
                        pyautogui.scroll(self.scroll_step)
                        label, color = "ZOOM IN", (255, 140, 0)
                        self.last_zoom_tick = now
                    elif delta < -self.zoom_deadzone:
                        pyautogui.scroll(-self.scroll_step)
                        label, color = "ZOOM OUT", (255, 140, 0)
                        self.last_zoom_tick = now
                self.zoom_prev_dist = pinch_dist
                self.trail.clear()
                return label, color
            else:
                self._end_zoom()  # hand opened up -> exit zoom mode
        else:
            if others_curled_for_zoom and pinch_dist < self.zoom_engage_ratio:
                self.zoom_engage_streak += 1
            else:
                self.zoom_engage_streak = 0
            if self.zoom_engage_streak >= self.zoom_engage_frames:
                self.zoom_active = True
                self.zoom_prev_dist = pinch_dist
                pyautogui.keyDown("ctrl")
                self.trail.clear()
                return "ZOOM armed (spread to zoom in)", (255, 140, 0)

        # ---- SCROLL: index + middle both up = up, both down = down ----
        if idx_ext and mid_ext:
            self.trail.clear()
            now = time.time()
            if now - self.last_scroll_tick >= self.scroll_tick_interval:
                pyautogui.scroll(self.scroll_step)
                self.last_scroll_tick = now
            return "SCROLL UP (index+middle up)", (0, 200, 255)

        if not idx_ext and not mid_ext:
            self.trail.clear()
            now = time.time()
            if now - self.last_scroll_tick >= self.scroll_tick_interval:
                pyautogui.scroll(-self.scroll_step)
                self.last_scroll_tick = now
            return "SCROLL DOWN (index+middle down)", (0, 200, 255)

        # ---- POINTER + CLICKS: index extended, middle curled ----
        # Cursor tracks the index fingertip continuously here regardless of
        # what the thumb or pinky are doing, so a click gesture never
        # interrupts movement. Clicks use a "trigger pull" pattern: curling
        # the thumb down fires left click, curling the pinky down fires
        # right click. Each re-arms only once you extend that finger back
        # out, so holding the curl doesn't repeat-fire.
        cursor_active = idx_ext and not mid_ext

        if not cursor_active:
            self.trail.clear()
            self.thumb_curl_streak = 0
            self.pinky_curl_streak = 0
            return "IDLE", (150, 150, 150)

        index_tip = lm_list[8]
        x1, y1 = index_tip[1], index_tip[2]

        target_x = np.interp(x1, (mx, img_w - mx), (0, self.screen_w))
        target_y = np.interp(y1, (my, img_h - my), (0, self.screen_h))

        # Amplify movement around the screen center so a given amount of
        # hand travel covers more screen distance (faster feel), since
        # this cursor is positioned absolutely and isn't affected by
        # Windows' own "Mouse pointer speed" setting at all.
        target_x = self.screen_w / 2 + (target_x - self.screen_w / 2) * self.cursor_gain
        target_y = self.screen_h / 2 + (target_y - self.screen_h / 2) * self.cursor_gain
        target_x = min(max(target_x, 0), self.screen_w - 1)
        target_y = min(max(target_y, 0), self.screen_h - 1)

        speed = math.hypot(target_x - self.prev_x, target_y - self.prev_y)
        smoothening = self.smooth_fast if speed > self.fast_speed_threshold else self.smooth_slow

        curr_x = self.prev_x + (target_x - self.prev_x) / smoothening
        curr_y = self.prev_y + (target_y - self.prev_y) / smoothening
        curr_x = min(max(curr_x, 0), self.screen_w - 1)
        curr_y = min(max(curr_y, 0), self.screen_h - 1)

        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

        label, color = "POINTER", (255, 0, 255)
        now = time.time()

        # --- LEFT CLICK: thumb curls down (armed -> fire -> disarmed until re-extended) ---
        if thumb_ext:
            self.thumb_curl_streak = 0
            self.left_click_armed = True
        else:
            self.thumb_curl_streak += 1
            if (self.left_click_armed and self.thumb_curl_streak == self.click_confirm_frames
                    and now - self.last_left_click_time > self.click_cooldown):
                if now - self.last_left_click_time < self.double_click_gap:
                    pyautogui.doubleClick()
                    label, color = "DOUBLE CLICK", (0, 255, 0)
                else:
                    pyautogui.click()
                    label, color = "LEFT CLICK", (0, 255, 0)
                self.last_left_click_time = now
                self.left_click_armed = False

        # --- RIGHT CLICK: pinky curls down (armed -> fire -> disarmed until re-extended) ---
        if pinky_ext:
            self.pinky_curl_streak = 0
            self.right_click_armed = True
        else:
            self.pinky_curl_streak += 1
            if (self.right_click_armed and self.pinky_curl_streak == self.click_confirm_frames
                    and now - self.last_right_click_time > self.click_cooldown):
                pyautogui.rightClick()
                label, color = "RIGHT CLICK", (0, 0, 255)
                self.last_right_click_time = now
                self.right_click_armed = False

        # --- Circle trail -> On-Screen Keyboard ---
        self.trail.append((x1, y1, now))
        if self._detect_circle():
            self._open_on_screen_keyboard()
            self.trail.clear()
            return "Circle detected -> On-Screen Keyboard", (255, 100, 0)

        return label, color

    # ------------------------------------------------------------------
    def _detect_circle(self):
        # Requires a large, deliberate loop — normal cursor wiggling won't
        # accumulate enough size/path-length/turning to pass all three
        # checks at once, which is what was causing the keyboard to pop
        # up during ordinary cursor movement before.
        if len(self.trail) < 35:
            return False
        if time.time() - self.last_circle_time < self.action_cooldown:
            return False

        pts = list(self.trail)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width < 150 or height < 150:
            return False

        path_length = sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )
        if path_length < 450:
            return False

        start, end = pts[0], pts[-1]
        start_end_dist = math.hypot(end[0] - start[0], end[1] - start[1])
        if start_end_dist > 0.3 * max(width, height):
            return False

        total_turn = 0.0
        for i in range(2, len(pts)):
            v1 = (pts[i - 1][0] - pts[i - 2][0], pts[i - 1][1] - pts[i - 2][1])
            v2 = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            if math.hypot(*v1) < 1 or math.hypot(*v2) < 1:
                continue
            a1 = math.atan2(v1[1], v1[0])
            a2 = math.atan2(v2[1], v2[0])
            da = a2 - a1
            while da > math.pi:
                da -= 2 * math.pi
            while da < -math.pi:
                da += 2 * math.pi
            total_turn += da

        if abs(total_turn) > math.radians(330):
            self.last_circle_time = time.time()
            return True
        return False

    @staticmethod
    def _open_on_screen_keyboard():
        attempts = [
            lambda: subprocess.Popen(["osk.exe"]),
            lambda: subprocess.Popen([r"C:\Windows\System32\osk.exe"]),
            lambda: subprocess.Popen("osk", shell=True),
            lambda: os.startfile("osk"),
        ]
        for attempt in attempts:
            try:
                attempt()
                return
            except Exception:
                continue

    # ------------------------------------------------------------------
    def _process_corner_dwell(self, img):
        px, py = pyautogui.position()

        in_top_right = px > self.screen_w - self.corner_size and py < self.corner_size
        in_top_left = px < self.corner_size and py < self.corner_size
        which = "minimize" if in_top_right else "close" if in_top_left else None

        if which is None:
            self.corner_hover_start = None
            self.corner_hover_which = None
            return

        now = time.time()
        if self.corner_hover_which != which:
            self.corner_hover_which = which
            self.corner_hover_start = now

        held = now - self.corner_hover_start
        progress = min(held / self.corner_hold_time, 1.0)

        img_h, img_w = img.shape[:2]
        cx = img_w - 40 if which == "minimize" else 40
        color = (0, 165, 255) if which == "minimize" else (0, 0, 255)
        cv2.circle(img, (cx, 40), int(10 + 25 * progress), color, 3)
        cv2.putText(img, which.upper(), (cx - 60 if which == "minimize" else cx, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if held >= self.corner_hold_time and now - self.last_action_time > self.action_cooldown:
            if which == "minimize":
                pyautogui.hotkey("win", "down")
            else:
                pyautogui.hotkey("alt", "f4")
            self.last_action_time = now
            self.corner_hover_start = None
            self.corner_hover_which = None


if __name__ == "__main__":
    controller = HandMouseController()
    controller.run()
