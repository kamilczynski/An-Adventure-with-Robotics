"""
mainauto.py — autonomous pick/drop loop for reBot B601

Keys:
  F      freeze current detections for mouse selection
  LEFT CLICK on a frozen class label
         select / deselect object; selection order = grasp order
  A      run selected sequence; without a frozen selection, start normal AUTO
  S      stop AUTO / selected sequence after current cycle
  G      one manual pick/drop cycle
  R      clear selection and resume live main preview
  U/D    move observation pose up/down by 5 cm
  J/L    rotate observation pose left/right by 0.10 rad
  Q/ESC  exit

Behavior:
  - YOLO + RGB-D detection.
  - Selects nearest valid object by camera Z distance.
  - Picks object using current tuned grasp logic.
  - Returns to ready/observation pose.
  - Returns to grasp point and releases object.
  - Repeats until no targets are detected or user presses S.
  - Keeps clean RGB live preview running during robot motion.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from pynput import mouse as pynput_mouse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT,):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from drivers.camera import make_camera
from drivers.robot.rebot_arm import RebotArm
from utils.camera_utils import load_config, load_hand_eye
from utils.ordinary_grasp import GraspPose, estimate_grasps, select_best_grasp
from utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base,
)
from utils.yolo_utils import load_yolo


# Current working tuned values
EXTRA_INSERT_M = 0.070
GRASP_WIDTH_M = 0.052
RELEASE_WIDTH_M = 0.090
APPROACH_OPEN_WIDTH_M = 0.090

# Table-object correction:
# positive value raises final grasp point in robot/base Z.
GRASP_Z_OFFSET_M = 0.015

# If True, EXTRA_INSERT_M will not push the gripper downward/upward.
# This prevents the gripper from diving below table objects.
EXTRA_INSERT_LOCK_Z = True

# Auto behavior
AUTO_SCAN_INTERVAL_S = 1.0
AUTO_NO_TARGET_LIMIT = 5
AUTO_SETTLE_AFTER_PICK_S = 1.5

# U key: raise observation pose by this much
OBS_LIFT_STEP_M = 0.050

# J/L keys: rotate observation pose left/right by this many radians
OBS_YAW_STEP_RAD = 0.100

# Lightweight GUI font — native OpenCV, no Pillow/FreeType overhead.
UI_FONT = cv2.FONT_HERSHEY_DUPLEX

def _draw_ui_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.50,
    thickness: int = 1,
) -> None:
    """Fast OpenCV text with subtle shadow."""
    x, y = org

    cv2.putText(
        img,
        text,
        (x + 1, y + 1),
        UI_FONT,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        UI_FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )





def _open_gripper_max_observation(
    robot: RebotArm,
    tag: str = "OBS",
    wait_s: float = 1.2,
) -> None:
    print(f"[{tag}] otwieram chwytak MAX w pozycji obserwacyjnej: {APPROACH_OPEN_WIDTH_M * 1000:.0f}mm")
    robot.open_gripper(distance_m=APPROACH_OPEN_WIDTH_M)
    time.sleep(wait_s)


def _move_ready(robot: RebotArm, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    robot.move_to(
        float(ready_cfg.get("x", 0.25)),
        float(ready_cfg.get("y", 0.0)),
        float(ready_cfg.get("z", 0.35)),
        float(ready_cfg.get("roll", 0.0)),
        float(ready_cfg.get("pitch", 1.2)),
        float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    robot.wait_motion(duration)


def _cam_to_base(T_hand_eye: np.ndarray, robot: RebotArm) -> np.ndarray:
    return robot.get_tcp_pose() @ T_hand_eye


def _is_valid_grasp(grasp: Optional[GraspPose]) -> bool:
    return (
        grasp is not None
        and grasp.position is not None
        and grasp.rotation is not None
        and grasp.tcp_rotation is not None
        and getattr(grasp, "rejected_reason", None) is None
    )


def _select_closest_grasp(grasps: list[GraspPose]) -> Optional[GraspPose]:
    """Select nearest valid object by camera Z distance."""
    candidates = [g for g in grasps if _is_valid_grasp(g)]
    if not candidates:
        return None
    return min(candidates, key=lambda g: float(g.position[2]))


def _execute_pick_drop(
    robot: RebotArm,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
    tag: str = "AUTO",
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    # Move final grasp deeper along approach vector.
    vx, vy, vz = xg - xp, yg - yp, zg - zp
    vn = math.sqrt(vx * vx + vy * vy + vz * vz)

    if vn > 1e-6:
        xg += EXTRA_INSERT_M * vx / vn
        yg += EXTRA_INSERT_M * vy / vn

        if not EXTRA_INSERT_LOCK_Z:
            zg += EXTRA_INSERT_M * vz / vn

        print(
            f"[{tag}] [Tune] extra insertion +{EXTRA_INSERT_M:.3f}m "
            f"lock_z={EXTRA_INSERT_LOCK_Z} "
            f"-> grasp xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})"
        )

    if abs(GRASP_Z_OFFSET_M) > 1e-6:
        zg += GRASP_Z_OFFSET_M
        print(
            f"[{tag}] [Tune] grasp Z offset {GRASP_Z_OFFSET_M:+.3f}m "
            f"-> grasp z={zg:+.3f}"
        )

    print(f"[{tag}] pregrasp xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[{tag}] grasp    xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print(f"[{tag}] --dry-run: skipping robot motion")
        return True

    print(f"[{tag}] otwieram chwytak do {APPROACH_OPEN_WIDTH_M * 1000:.0f}mm przed podejściem...")
    robot.open_gripper(distance_m=APPROACH_OPEN_WIDTH_M)
    time.sleep(0.6)

    print(f"[{tag}] ruch do pregrasp...")
    if not robot.move_to(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print(f"[{tag}] pregrasp IK failed — abort")
        return False
    robot.wait_motion(2.0)

    print(f"[{tag}] ruch do grasp...")
    if not robot.move_to(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print(f"[{tag}] grasp IK failed — abort")
        return False
    robot.wait_motion(1.5)

    print(f"[{tag}] zamykam chwytak do {GRASP_WIDTH_M * 1000:.0f}mm...")
    robot.open_gripper(distance_m=GRASP_WIDTH_M)
    time.sleep(0.6)

    print(f"[{tag}] powrót do pozycji obserwacyjnej z obiektem...")
    _move_ready(robot, ready_cfg)

    print(f"[{tag}] otwieram chwytak do {RELEASE_WIDTH_M * 1000:.0f}mm i wypuszczam obiekt...")
    robot.open_gripper(distance_m=RELEASE_WIDTH_M)
    time.sleep(1.0)

    print(f"[{tag}] sekwencja zakończona.")
    return True


def _class_color(cls_id: int) -> tuple[int, int, int]:
    # BGR colors — lightweight fixed palette.
    palette = (
        (70, 230, 90),
        (70, 210, 255),
        (255, 130, 60),
        (210, 120, 255),
        (255, 210, 80),
    )
    return palette[int(cls_id) % len(palette)]



def _extract_detection_items(
    results: list[Any],
    grasps: Optional[list[GraspPose]] = None,
) -> list[dict[str, Any]]:
    """
    Convert current YOLO boxes into lightweight UI items.

    When grasps are supplied, each detection is matched to the nearest valid
    grasp center inside that detection box. Matching is performed only when
    F is pressed, so it does not add cost to the normal live loop.
    """
    items: list[dict[str, Any]] = []
    if not results:
        return items

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    names = getattr(r0, "names", {}) or {}

    if boxes is None or len(boxes) == 0:
        return items

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()

    used_grasp_ids: set[int] = set()

    for det_id, (box, conf, cls_f) in enumerate(zip(xyxy, confs, classes)):
        cls_id = int(cls_f)
        class_name = str(names.get(cls_id, cls_id))
        x1, y1, x2, y2 = [int(v) for v in box]

        matched_grasp: Optional[GraspPose] = None

        if grasps is not None:
            candidates: list[tuple[float, int, GraspPose]] = []
            dcx = 0.5 * (x1 + x2)
            dcy = 0.5 * (y1 + y2)

            for grasp_idx, grasp in enumerate(grasps):
                if grasp_idx in used_grasp_ids or not _is_valid_grasp(grasp):
                    continue

                try:
                    gcx, gcy = [float(v) for v in grasp.center_px]
                except Exception:
                    continue

                if not (x1 <= gcx <= x2 and y1 <= gcy <= y2):
                    continue

                grasp_name = str(getattr(grasp, "class_name", "")).strip().lower()
                if grasp_name and grasp_name != class_name.strip().lower():
                    continue

                dist2 = (gcx - dcx) ** 2 + (gcy - dcy) ** 2
                candidates.append((dist2, grasp_idx, grasp))

            if candidates:
                _, grasp_idx, matched_grasp = min(candidates, key=lambda x: x[0])
                used_grasp_ids.add(grasp_idx)

        items.append(
            {
                "id": det_id,
                "cls_id": cls_id,
                "class_name": class_name,
                "conf": float(conf),
                "xyxy": (x1, y1, x2, y2),
                "grasp": matched_grasp,
            }
        )

    return items


def _label_geometry(item: dict[str, Any]) -> tuple[str, tuple[int, int], tuple[int, int, int, int]]:
    """Return label text, baseline position and clickable label rectangle."""
    x1, y1, _, _ = item["xyxy"]
    text = f"{item['class_name']} {float(item['conf']):.2f}"
    scale = 0.46
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, UI_FONT, scale, thickness)
    text_y = max(th + 3, y1 - 6)

    # Slightly enlarged hit area around text, still visually background-free.
    hit_rect = (
        max(0, x1 - 4),
        max(0, text_y - th - 5),
        x1 + tw + 6,
        text_y + baseline + 5,
    )
    return text, (x1, text_y), hit_rect


def _render_detection_items(
    display: np.ndarray,
    items: list[dict[str, Any]],
    selected_ids: Optional[list[int]] = None,
) -> None:
    """Draw class/confidence labels and optional ordered selection markers."""
    selected_ids = selected_ids or []
    order_map = {det_id: i + 1 for i, det_id in enumerate(selected_ids)}

    for item in items:
        cls_id = int(item["cls_id"])
        color = _class_color(cls_id)
        x1, y1, x2, y2 = item["xyxy"]
        is_selected = int(item["id"]) in order_map

        # Selected objects get a thicker bounding box.
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            color,
            3 if is_selected else 1,
            cv2.LINE_AA,
        )

        # Class label remains strictly: class name + confidence.
        # No filled background.
        label, org, _ = _label_geometry(item)
        _draw_ui_text(
            display,
            label,
            org,
            color,
            scale=0.46,
            thickness=1,
        )

        # Selection order is drawn separately from the class label.
        if is_selected:
            order = order_map[int(item["id"])]
            _draw_ui_text(
                display,
                f"#{order}",
                (max(x1, x2 - 34), max(20, y1 + 18)),
                (255, 255, 255),
                scale=0.48,
                thickness=1,
            )


def _render_display(
    image: np.ndarray,
    results: list[Any],
    best: Optional[GraspPose],
    status_text: str,
    frozen_items: Optional[list[dict[str, Any]]] = None,
    selected_ids: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Lightweight analytic view.

    Displays:
      - detection bounding box
      - class name + confidence only
      - selected grasp center / jaw line
      - mode + FPS
      - selection order marker when F-selection is active

    Deliberately does NOT display XYZ, object position or jaw width as text.
    """
    display = image.copy()

    items = frozen_items if frozen_items is not None else _extract_detection_items(results)
    _render_detection_items(display, items, selected_ids)

    # Selected grasp marker only.
    # No XYZ / jaw text.
    if best is not None and best.position is not None:
        try:
            cx, cy = [int(v) for v in best.center_px]
            jaw_px = int(np.clip(float(best.jaw_width_m) * 900.0, 35, 90))

            cv2.line(
                display,
                (cx - jaw_px // 2, cy),
                (cx + jaw_px // 2, cy),
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                display,
                (cx, cy),
                4,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    # Minimal status / FPS.
    _draw_ui_text(
        display,
        status_text,
        (10, 28),
        (245, 245, 245),
        scale=0.58,
        thickness=1,
    )

    return display


def _print_best_grasp(grasp: GraspPose, prefix: str = "G") -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print(f"\n[{prefix}] 当前最佳夹取:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous reBot grasp loop")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate and print only; no robot motion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )

    print("=== 初始化机械臂 ===")
    robot = RebotArm(
        config_path=robot_cfg.get("config_path"),
        urdf_path=robot_cfg.get("urdf_path"),
        repo_root=robot_cfg.get("repo_root"),
    )
    robot.connect(enable=True)
    robot.init_gripper()

    print("[Robot] 移动到预备位置...")
    _move_ready(robot, ready_cfg)
    _open_gripper_max_observation(robot, tag="START")

    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
        print("[WARN] 手眼标定不可用或非 eye_in_hand，夹取执行将被禁用")
        T_hand_eye = None

    print(f"=== 相机: {cfg.get('camera', {}).get('type')} ===")
    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(15)
    K = cam.K.astype(np.float32)

    yolo_cfg = cfg.get("yolo", {})
    yolo_cfg["model_name"] = "/workspace/bestapplesad.pt"
    cfg["yolo"] = yolo_cfg
    gp_cfg = cfg.get("grasp_pipeline", {})
    grasp_cfg = gp_cfg.get("grasp", {})

    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
    infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

    print(f"=== 加载 YOLO: {model_name} ===")
    model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None

    # F-selection state.
    frozen_image: Optional[np.ndarray] = None
    frozen_items: list[dict[str, Any]] = []
    frozen_best: Optional[GraspPose] = None
    selected_ids: list[int] = []

    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    auto_enabled = False
    auto_misses = 0
    next_auto_scan_time = 0.0

    grasp_busy = threading.Event()
    grasp_thread: Optional[threading.Thread] = None

    sequence_active = threading.Event()
    sequence_stop_requested = threading.Event()
    sequence_done = threading.Event()

    window_name = "MainAuto — Ordinary Grasp"
    clean_window_name = "Live RGB Clean"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(clean_window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)


    def _refresh_frozen_display() -> None:
        nonlocal last_display

        if frozen_image is None:
            return

        label = f"FROZEN | selected {len(selected_ids)} | click labels | A run"
        last_display = _render_display(
            frozen_image,
            last_results,
            frozen_best,
            label,
            frozen_items=frozen_items,
            selected_ids=selected_ids,
        )

    def _toggle_selection_at(x: int, y: int) -> None:
        """Toggle a frozen detection by image-local coordinates."""
        if not frozen or grasp_busy.is_set() or auto_enabled:
            return
        if not frozen_items:
            return

        for item in frozen_items:
            _, _, (rx1, ry1, rx2, ry2) = _label_geometry(item)
            if not (rx1 <= x <= rx2 and ry1 <= y <= ry2):
                continue

            det_id = int(item["id"])
            grasp = item.get("grasp")

            if grasp is None or not _is_valid_grasp(grasp):
                print(
                    f"[SELECT] {item['class_name']} {item['conf']:.2f}: "
                    "no valid grasp candidate for this detection."
                )
                return

            if det_id in selected_ids:
                selected_ids.remove(det_id)
                print(
                    f"[SELECT] removed: {item['class_name']} {item['conf']:.2f}"
                )
            else:
                selected_ids.append(det_id)
                print(
                    f"[SELECT] #{len(selected_ids)}: "
                    f"{item['class_name']} {item['conf']:.2f}"
                )

            _refresh_frozen_display()
            return

    def _global_mouse_click(
        x: int,
        y: int,
        button: Any,
        pressed: bool,
    ) -> None:
        """
        Global X11 mouse listener.

        OpenCV 4.11 Qt can throw NULL window handler from setMouseCallback().
        pynput avoids that HighGUI path entirely. We map the screen click to
        image-local coordinates using cv2.getWindowImageRect().
        """
        if not pressed or button != pynput_mouse.Button.left:
            return
        if not frozen or last_display is None:
            return

        try:
            wx, wy, ww, wh = cv2.getWindowImageRect(window_name)
        except Exception:
            return

        if ww <= 0 or wh <= 0:
            return

        if not (wx <= x < wx + ww and wy <= y < wy + wh):
            return

        img_h, img_w = last_display.shape[:2]

        ix = int((x - wx) * img_w / ww)
        iy = int((y - wy) * img_h / wh)

        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))

        _toggle_selection_at(ix, iy)

    # Start lightweight event-driven mouse listener.
    # No polling and no extra work in the 31+ FPS video loop.
    mouse_listener = pynput_mouse.Listener(on_click=_global_mouse_click)
    mouse_listener.start()


    print("\n[Keys]")
    print("  F      freeze current detections for mouse selection")
    print("  CLICK  left-click a frozen class label to select/deselect it")
    print("  A      run selected sequence; otherwise start normal AUTO")
    print("  S      stop after current cycle")
    print("  G      one manual pick/drop")
    print("  R      clear selection and resume live main preview")
    print("  U      lift observation pose up by +5cm when idle")
    print("  D      lower observation pose down by -5cm when idle")
    print("  J      look left by yaw +0.10 rad when idle")
    print("  L      look right by yaw -0.10 rad when idle")
    print("  Q/ESC  exit")
    print("\n[Selection] F -> click labels in desired order -> A\n")

    def _start_pick_thread(
        grasp6d: tuple[float, ...],
        pre6d: tuple[float, ...],
        tag: str,
        clear_frozen_after: bool,
    ) -> None:
        nonlocal grasp_thread, next_auto_scan_time, frozen, last_display

        if grasp_busy.is_set():
            print(f"[{tag}] Robot busy — ignoring new request.")
            return

        def _worker() -> None:
            nonlocal next_auto_scan_time, frozen, last_display
            try:
                _execute_pick_drop(robot, grasp6d, pre6d, ready_cfg, dry_run=args.dry_run, tag=tag)
            except Exception as exc:
                print(f"[{tag}] Worker error: {exc}")
            finally:
                if clear_frozen_after:
                    frozen = False
                    last_display = None
                next_auto_scan_time = time.perf_counter() + AUTO_SETTLE_AFTER_PICK_S
                grasp_busy.clear()

        grasp_busy.set()
        grasp_thread = threading.Thread(target=_worker, name=f"{tag.lower()}-pick-worker", daemon=True)
        grasp_thread.start()
        print(f"[{tag}] Pick/drop sequence started in background.")

    def _start_selected_sequence(
        plans: list[tuple[tuple[float, ...], tuple[float, ...], str]],
    ) -> None:
        nonlocal grasp_thread

        if grasp_busy.is_set():
            print("[SEQ] Robot busy — ignoring sequence start.")
            return
        if not plans:
            print("[SEQ] No selected objects.")
            return

        sequence_stop_requested.clear()
        sequence_done.clear()
        sequence_active.set()
        grasp_busy.set()

        def _worker() -> None:
            try:
                total = len(plans)

                for index, (grasp6d, pre6d, class_name) in enumerate(plans, start=1):
                    if index > 1 and sequence_stop_requested.is_set():
                        print("[SEQ] STOPPED before next selected object.")
                        break

                    tag = f"SEQ {index}/{total} {class_name}"
                    print(f"\n[SEQ] Starting {index}/{total}: {class_name}")

                    ok = _execute_pick_drop(
                        robot,
                        grasp6d,
                        pre6d,
                        ready_cfg,
                        dry_run=args.dry_run,
                        tag=tag,
                    )

                    if not ok and not args.dry_run:
                        print(f"[SEQ] {class_name}: motion failed — aborting remaining sequence.")
                        break

                    if sequence_stop_requested.is_set():
                        print("[SEQ] STOP requested — current object finished, sequence ended.")
                        break

                    if index < total:
                        time.sleep(AUTO_SETTLE_AFTER_PICK_S)

            except Exception as exc:
                print(f"[SEQ] Worker error: {exc}")
            finally:
                sequence_active.clear()
                grasp_busy.clear()
                sequence_done.set()

        grasp_thread = threading.Thread(
            target=_worker,
            name="selected-sequence-worker",
            daemon=True,
        )
        grasp_thread.start()
        print(f"[SEQ] Started selected sequence with {len(plans)} object(s).")

    try:
        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            cv2.imshow(clean_window_name, color_bgr)

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # Return to normal live mode after a finite user-selected sequence.
            if sequence_done.is_set():
                sequence_done.clear()
                frozen = False
                frozen_image = None
                frozen_items.clear()
                selected_ids.clear()
                frozen_best = None
                last_display = None
                print("[SEQ] Finished. Live view resumed.")

            if (not frozen) and (not grasp_busy.is_set()) and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(last_results, depth_mm, K, depth_quantile=depth_quantile)

            best_live = _select_closest_grasp(last_grasps) or select_best_grasp(last_grasps)

            mode_text = "AUTO" if auto_enabled else "MANUAL"
            if grasp_busy.is_set():
                mode_text += " | MOVING"
            elif frozen:
                mode_text += " | FROZEN"
            else:
                mode_text += " | LIVE"

            status = f"{mode_text} {fps_value:.1f}fps"

            if frozen and last_display is not None:
                # Cached frozen frame: no inference and no repeated selection rendering.
                display = last_display
            else:
                display = _render_display(color_bgr, last_results, best_live, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(clean_window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key in (ord("q"), ord("Q"), 27):
                print("[Key] Quit requested.")
                break

            if key in (ord("r"), ord("R")):
                if grasp_busy.is_set():
                    print("[R] Robot busy — wait until the current cycle finishes.")
                    continue

                frozen = False
                frozen_image = None
                frozen_items.clear()
                selected_ids.clear()
                frozen_best = None
                last_display = None
                print("[R] Selection cleared. Live view resumed.")
                continue

            if key in (ord("f"), ord("F")):
                if grasp_busy.is_set():
                    print("[F] Robot busy — freeze is available only when the robot is idle.")
                    continue
                if auto_enabled:
                    print("[F] AUTO is active — press S first.")
                    continue

                print("\n[F] Freezing current detections for ordered selection...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[F] Failed to capture RGB-D frame.")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(
                    snap_results,
                    snap_depth,
                    K,
                    depth_quantile=depth_quantile,
                )

                frozen_image = snap_color.copy()
                frozen_items = _extract_detection_items(snap_results, snap_grasps)
                selected_ids.clear()
                frozen_best = _select_closest_grasp(snap_grasps) or select_best_grasp(snap_grasps)

                last_results = snap_results
                last_grasps = snap_grasps
                frozen = True

                _refresh_frozen_display()

                selectable = sum(
                    1 for item in frozen_items
                    if item.get("grasp") is not None and _is_valid_grasp(item.get("grasp"))
                )
                print(
                    f"[F] Frozen {len(frozen_items)} detection(s), "
                    f"{selectable} selectable. Click labels in desired grasp order."
                )
                continue

            if key in (ord("a"), ord("A")):
                if T_hand_eye is None:
                    print("[AUTO] Cannot start: hand-eye calibration unavailable.")
                    continue
                if grasp_busy.is_set():
                    print("[A] Robot busy — ignoring A.")
                    continue

                # If F-selection is active, A runs exactly the selected order.
                if frozen and frozen_items:
                    if not selected_ids:
                        print("[SEQ] No objects selected. Click frozen class labels first.")
                        continue

                    T_cam2base = _cam_to_base(T_hand_eye, robot)
                    plans: list[tuple[tuple[float, ...], tuple[float, ...], str]] = []

                    for det_id in selected_ids:
                        item = next(
                            (it for it in frozen_items if int(it["id"]) == int(det_id)),
                            None,
                        )
                        if item is None:
                            continue

                        grasp = item.get("grasp")
                        if grasp is None or not _is_valid_grasp(grasp):
                            print(
                                f"[SEQ] Skipping {item['class_name']}: "
                                "valid grasp is unavailable."
                            )
                            continue

                        grasp6d, pre6d = transform_grasp_pose_to_base(
                            grasp.position,
                            grasp.tcp_rotation,
                            T_cam2base,
                            pregrasp_offset_m,
                        )
                        plans.append((grasp6d, pre6d, str(item["class_name"])))

                    if not plans:
                        print("[SEQ] No valid selected grasps to execute.")
                        continue

                    print(
                        "[SEQ] Order: "
                        + " -> ".join(plan[2] for plan in plans)
                    )
                    _start_selected_sequence(plans)
                    continue

                # No F-selection: preserve the original normal AUTO behavior.
                auto_enabled = True
                auto_misses = 0
                next_auto_scan_time = 0.0
                frozen = False
                last_display = None
                print("[AUTO] STARTED")
                continue

            if key in (ord("s"), ord("S")):
                auto_enabled = False
                auto_misses = 0

                if sequence_active.is_set():
                    sequence_stop_requested.set()
                    print("[SEQ] STOP requested. Current selected object will finish first.")
                else:
                    print("[AUTO] STOP requested. Current robot cycle will finish if already moving.")
                continue

            if key in (ord("u"), ord("U")):
                if grasp_busy.is_set():
                    print("[U] Robot jest jeszcze w cyklu — najpierw poczekaj aż zakończy ruch po S.")
                    continue
                if auto_enabled:
                    print("[U] Auto nadal aktywne — najpierw naciśnij S.")
                    continue

                old_z = float(ready_cfg.get("z", 0.35))
                new_z = old_z + OBS_LIFT_STEP_M
                ready_cfg["z"] = new_z

                print(f"[U] Podnoszę pozycję obserwacyjną: z {old_z:.3f} -> {new_z:.3f} m")
                _move_ready(robot, ready_cfg)
                print("[U] Gotowe. Nowa pozycja obserwacyjna aktywna tylko w tej sesji.")
                continue

            if key in (ord("d"), ord("D")):
                if grasp_busy.is_set():
                    print("[D] Robot jest jeszcze w cyklu — najpierw poczekaj aż zakończy ruch po S.")
                    continue
                if auto_enabled:
                    print("[D] Auto nadal aktywne — najpierw naciśnij S.")
                    continue

                old_z = float(ready_cfg.get("z", 0.35))
                new_z = old_z - OBS_LIFT_STEP_M

                # Basic safety clamp: do not let observation pose go too low.
                if new_z < 0.120:
                    print(f"[D] Blokuję zejście za nisko: żądane z={new_z:.3f} m, minimum 0.120 m")
                    continue

                ready_cfg["z"] = new_z

                print(f"[D] Obniżam pozycję obserwacyjną: z {old_z:.3f} -> {new_z:.3f} m")
                _move_ready(robot, ready_cfg)
                print("[D] Gotowe. Nowa pozycja obserwacyjna aktywna tylko w tej sesji.")
                continue

            if key in (ord("j"), ord("J"), ord("l"), ord("L")):
                if grasp_busy.is_set():
                    print("[LOOK] Robot jest jeszcze w cyklu — najpierw poczekaj aż zakończy ruch po S.")
                    continue
                if auto_enabled:
                    print("[LOOK] Auto nadal aktywne — najpierw naciśnij S.")
                    continue

                old_yaw = float(ready_cfg.get("yaw", 0.0))
                if key in (ord("j"), ord("J")):
                    new_yaw = old_yaw + OBS_YAW_STEP_RAD
                    direction = "LEFT"
                else:
                    new_yaw = old_yaw - OBS_YAW_STEP_RAD
                    direction = "RIGHT"

                ready_cfg["yaw"] = new_yaw
                print(f"[LOOK] {direction}: yaw {old_yaw:.3f} -> {new_yaw:.3f} rad")
                _move_ready(robot, ready_cfg)
                print("[LOOK] Gotowe. Nowa pozycja obserwacyjna aktywna tylko w tej sesji.")
                continue

            if key in (ord("g"), ord("G")):
                if grasp_busy.is_set():
                    print("[G] Robot busy — ignoring G.")
                    continue

                print("\n[G] 采帧并估计夹取姿态...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] 采帧失败")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(snap_results, snap_depth, K, depth_quantile=depth_quantile)
                best = _select_closest_grasp(snap_grasps) or select_best_grasp(snap_grasps)
                if best is None or not _is_valid_grasp(best):
                    print("[G] 未找到有效夹取候选")
                    continue

                _print_best_grasp(best, prefix="G")

                snap_display = _render_display(snap_color, snap_results, best, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] 手眼标定不可用，无法执行夹取")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, robot)
                grasp6d, pre6d = transform_grasp_pose_to_base(
                    best.position,
                    best.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                )
                _start_pick_thread(grasp6d, pre6d, tag="G", clear_frozen_after=False)
                continue

            # Autonomous scan/trigger
            if auto_enabled and (not grasp_busy.is_set()) and now >= next_auto_scan_time:
                next_auto_scan_time = now + AUTO_SCAN_INTERVAL_S

                auto_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                auto_grasps = estimate_grasps(auto_results, depth_mm, K, depth_quantile=depth_quantile)
                best_auto = _select_closest_grasp(auto_grasps)

                last_results = auto_results
                last_grasps = auto_grasps

                if best_auto is None or not _is_valid_grasp(best_auto):
                    auto_misses += 1
                    print(f"[AUTO] No target scan {auto_misses}/{AUTO_NO_TARGET_LIMIT}")
                    if auto_misses >= AUTO_NO_TARGET_LIMIT:
                        auto_enabled = False
                        auto_misses = 0
                        print("[AUTO] STOPPED: no targets detected.")
                    continue

                auto_misses = 0
                _print_best_grasp(best_auto, prefix="AUTO")

                auto_display = _render_display(color_bgr, auto_results, best_auto, "AUTO SNAPSHOT")
                frozen = True
                last_display = auto_display

                if T_hand_eye is None:
                    print("[AUTO] 手眼标定不可用，无法执行夹取")
                    auto_enabled = False
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, robot)
                grasp6d, pre6d = transform_grasp_pose_to_base(
                    best_auto.position,
                    best_auto.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                )
                _start_pick_thread(grasp6d, pre6d, tag="AUTO", clear_frozen_after=True)

    finally:
        print("\n[退出] 释放夹爪并回零...")
        try:
            auto_enabled = False
            sequence_stop_requested.set()
            if grasp_thread is not None and grasp_thread.is_alive():
                print("[Exit] Waiting for active robot cycle to finish before cleanup...")
                grasp_thread.join()

            robot.release_gripper()
            robot.safe_home()
        except Exception as exc:
            print(f"[退出] {exc}")

        try:
            mouse_listener.stop()
        except Exception:
            pass

        robot.disconnect()
        cam.close()
        cv2.destroyAllWindows()
        print("已退出。")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
# TUTAJ WKLEJASZ CAŁY KOD
