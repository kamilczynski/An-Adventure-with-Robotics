from __future__ import annotations

import atexit
import threading
import time
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request

from drivers.camera import make_camera
from drivers.robot.rebot_arm import RebotArm
from utils.camera_utils import load_config, load_hand_eye
from utils.yolo_utils import load_yolo

from scripts.ball_box_obstacle_auto import (
    PROJECT_ROOT,
    DEFAULT_MODEL_NAME,
    PREGRASP_DEFAULT_M,
    DEPTH_QUANTILE_DEFAULT,
    RELEASE_WIDTH_M,
    BOX_DROP_Z_OFFSET_M,
    BOX_DROP_MIN_Z_M,
    BOX_DROP_MAX_Z_M,
    _best_det,
    _detect_scene,
    _move_ready,
    _render_display,
    _select_fallback_grasp,
    _select_closest_grasp_by_role,
    _cam_to_base,
    _point_cam_to_base,
    _grasp_to_base_6d,
    _execute_pick,
    _safe_move_to,
    _safe_open_gripper,
    _clamp,
)


HOST = "0.0.0.0"
PORT = 8000

app = Flask(__name__)

state_lock = threading.RLock()
camera_lock = threading.Lock()

robot: RebotArm | None = None
cam: Any = None
model: Any = None
yolo_opts: dict[str, Any] | None = None

cfg: dict[str, Any] | None = None
K: np.ndarray | None = None
T_hand_eye: np.ndarray | None = None

ready_cfg: dict[str, Any] | None = None
pregrasp_offset_m: float = PREGRASP_DEFAULT_M
depth_quantile: float = DEPTH_QUANTILE_DEFAULT

initialized = False


def _get_camera_frame():
    """
    Thread-safe Gemini frame acquisition.

    The API and live preview may both request frames, but only one
    thread accesses the physical camera at a time.
    """
    assert cam is not None

    with camera_lock:
        return cam.get_frame()



def _serialize_detection(det: dict[str, Any] | None) -> dict[str, Any]:
    if det is None:
        return {
            "detected": False,
        }

    x1, y1, x2, y2 = det["xyxy"]

    return {
        "detected": True,
        "class_name": str(det.get("class_name", "")),
        "confidence": float(det.get("conf", 0.0)),
        "bbox_xyxy": [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ],
        "center_px": [
            float(det.get("cx", 0.0)),
            float(det.get("cy", 0.0)),
        ],
    }


def _serialize_grasp(grasp: Any) -> dict[str, Any] | None:
    if grasp is None:
        return None

    result: dict[str, Any] = {
        "available": True,
        "class_name": str(getattr(grasp, "class_name", "")),
        "confidence": float(getattr(grasp, "conf", 0.0)),
        "jaw_width_m": float(getattr(grasp, "jaw_width_m", 0.0)),
    }

    center_px = getattr(grasp, "center_px", None)
    if center_px is not None:
        result["center_px"] = [
            int(center_px[0]),
            int(center_px[1]),
        ]

    position = getattr(grasp, "position", None)
    if position is not None:
        result["position_camera_m"] = [
            float(v) for v in np.asarray(position).reshape(-1)[:3]
        ]

    return result


def initialize() -> None:
    global robot
    global cam
    global model
    global yolo_opts
    global cfg
    global K
    global T_hand_eye
    global ready_cfg
    global pregrasp_offset_m
    global depth_quantile
    global initialized

    if initialized:
        return

    print("========================================")
    print(" reBot Agent API")
    print("========================================")

    cfg = load_config(PROJECT_ROOT / "config/default.yaml")

    yolo_cfg = cfg.get("yolo", {})
    yolo_cfg["model_name"] = DEFAULT_MODEL_NAME
    cfg["yolo"] = yolo_cfg

    robot_cfg = cfg.get("robot", {})

    ready_cfg = robot_cfg.get(
        "ready_pose",
        {
            "x": 0.25,
            "y": 0.0,
            "z": 0.35,
            "roll": 0.0,
            "pitch": 1.2,
            "yaw": 0.0,
            "duration": 3.0,
        },
    )

    gp_cfg = cfg.get("grasp_pipeline", {})
    grasp_cfg = gp_cfg.get("grasp", {})

    pregrasp_offset_m = float(
        grasp_cfg.get(
            "pregrasp_offset_m",
            PREGRASP_DEFAULT_M,
        )
    )

    depth_quantile = float(
        grasp_cfg.get(
            "depth_quantile",
            DEPTH_QUANTILE_DEFAULT,
        )
    )

    print("[1/4] Connecting B601...")

    robot = RebotArm(
        config_path=robot_cfg.get("config_path"),
        urdf_path=robot_cfg.get("urdf_path"),
        repo_root=robot_cfg.get("repo_root"),
    )

    robot.connect(enable=True)
    robot.init_gripper()

    print("[2/4] Moving to observation pose...")
    _move_ready(robot, ready_cfg)

    cam_type = str(
        cfg.get("camera", {}).get("type", "")
    ).lower()

    T_hand_eye, hand_eye_mode = load_hand_eye(
        PROJECT_ROOT,
        cam_type,
    )

    if T_hand_eye is None:
        raise RuntimeError(
            "Hand-eye calibration not found"
        )

    if hand_eye_mode != "eye_in_hand":
        raise RuntimeError(
            f"Expected eye_in_hand calibration, got: {hand_eye_mode}"
        )

    print("[3/4] Starting Gemini 336...")

    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(15)

    K = cam.K.astype(np.float32)

    print("Camera K:")
    print(K)

    print("[4/4] Loading YOLO:")
    print(DEFAULT_MODEL_NAME)

    model, yolo_opts = load_yolo(
        cfg,
        project_root=PROJECT_ROOT,
    )

    initialized = True

    print()
    print("SYSTEM READY")
    print(f"API: http://127.0.0.1:{PORT}")
    print()


def observe_scene() -> dict[str, Any]:
    if not initialized:
        raise RuntimeError("System is not initialized")

    assert cam is not None
    assert model is not None
    assert yolo_opts is not None
    assert K is not None

    with state_lock:
        # Several attempts are useful because RGB may occasionally
        # start one or two frames later than depth.
        color_bgr = None
        depth_mm = None

        for _ in range(20):
            color_bgr, depth_mm = _get_camera_frame()

            if color_bgr is not None and depth_mm is not None:
                break

            time.sleep(0.03)

        if color_bgr is None or depth_mm is None:
            raise RuntimeError(
                "Could not obtain RGB + depth frame"
            )

        _, grasps, dets, blocked, info = _detect_scene(
            model,
            yolo_opts,
            color_bgr,
            depth_mm,
            K,
            depth_quantile,
        )

        ball = _best_det(dets, "ball")
        box = _best_det(dets, "box")
        obstacle = _best_det(dets, "obstacle")

        ball_grasp = _select_closest_grasp_by_role(
            grasps,
            "ball",
        )

        obstacle_grasp = _select_closest_grasp_by_role(
            grasps,
            "obstacle",
        )

        box_grasp = _select_closest_grasp_by_role(
            grasps,
            "box",
        )

        objects = {
            "ball": {
                **_serialize_detection(ball),
                "grasp": _serialize_grasp(ball_grasp),
            },
            "box": {
                **_serialize_detection(box),
                "grasp": _serialize_grasp(box_grasp),
            },
            "obstacle": {
                **_serialize_detection(obstacle),
                "grasp": _serialize_grasp(obstacle_grasp),
            },
        }

        return {
            "success": True,
            "timestamp": time.time(),
            "objects": objects,
            "relations": {
                "obstacle_on_ball": bool(blocked),
                "overlap_ball_ratio": float(
                    info.get("overlap_ball_ratio", 0.0)
                ),
                "obstacle_center_inside_ball": bool(
                    info.get(
                        "obstacle_center_inside_ball",
                        False,
                    )
                ),
            },
            "robot": {
                "pose": "observation",
            },
        }


@app.get("/health")
def health():
    return jsonify(
        {
            "success": True,
            "initialized": initialized,
            "robot": robot is not None,
            "camera": cam is not None,
            "yolo": model is not None,
            "hand_eye": T_hand_eye is not None,
        }
    )


@app.post("/observe")
def observe():
    try:
        result = observe_scene()
        return jsonify(result)

    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )



def move_object_to_box_action(
    role: str,
    execute: bool = False,
) -> dict[str, Any]:
    """
    Detect the CURRENT scene and prepare or execute a move of
    ball/obstacle into the detected box.

    execute=False:
        perception + geometry only, absolutely no manipulation.

    execute=True:
        physically execute the pick-and-place.
    """

    if role not in {"ball", "obstacle"}:
        raise ValueError(
            "object must be 'ball' or 'obstacle'"
        )

    if not initialized:
        raise RuntimeError("System is not initialized")

    assert robot is not None
    assert cam is not None
    assert model is not None
    assert yolo_opts is not None
    assert K is not None
    assert T_hand_eye is not None
    assert ready_cfg is not None

    with state_lock:
        print(f"[ACTION] Fresh observation for object={role}")

        color_bgr = None
        depth_mm = None

        for _ in range(20):
            color_bgr, depth_mm = _get_camera_frame()

            if color_bgr is not None and depth_mm is not None:
                break

            time.sleep(0.03)

        if color_bgr is None or depth_mm is None:
            raise RuntimeError(
                "Could not obtain RGB + depth frame"
            )

        _, grasps, dets, blocked, block_info = _detect_scene(
            model,
            yolo_opts,
            color_bgr,
            depth_mm,
            K,
            depth_quantile,
        )

        target_det = _best_det(dets, role)
        box_det = _best_det(dets, "box")

        if target_det is None:
            raise RuntimeError(
                f"{role} is not currently detected"
            )

        if box_det is None:
            raise RuntimeError(
                "box is not currently detected"
            )

        target_grasp = _select_closest_grasp_by_role(
            grasps,
            role,
        )

        box_grasp = _select_closest_grasp_by_role(
            grasps,
            "box",
        )

        if target_grasp is None:
            raise RuntimeError(
                f"No valid grasp found for {role}"
            )

        if box_grasp is None:
            raise RuntimeError(
                "No valid 3D position found for box"
            )

        # CURRENT robot pose + hand-eye transform.
        T_cam2base = _cam_to_base(
            T_hand_eye,
            robot,
        )

        grasp6d, pre6d = _grasp_to_base_6d(
            target_grasp,
            T_cam2base,
            pregrasp_offset_m,
        )

        box_base_xyz = _point_cam_to_base(
            T_cam2base,
            np.asarray(
                box_grasp.position,
                dtype=np.float32,
            ),
        )

        box_x, box_y, box_z = [
            float(v)
            for v in box_base_xyz.tolist()
        ]

        drop_z = _clamp(
            box_z + BOX_DROP_Z_OFFSET_M,
            BOX_DROP_MIN_Z_M,
            BOX_DROP_MAX_Z_M,
        )

        plan = {
            "object": role,
            "object_confidence": float(
                target_det.get("conf", 0.0)
            ),
            "box_confidence": float(
                box_det.get("conf", 0.0)
            ),
            "object_grasp_camera_m": [
                float(v)
                for v in np.asarray(
                    target_grasp.position
                ).reshape(-1)[:3]
            ],
            "object_grasp_base_6d": [
                float(v) for v in grasp6d
            ],
            "object_pregrasp_base_6d": [
                float(v) for v in pre6d
            ],
            "box_base_xyz": [
                box_x,
                box_y,
                box_z,
            ],
            "drop_base_xyz": [
                box_x,
                box_y,
                drop_z,
            ],
            "jaw_width_m": float(
                target_grasp.jaw_width_m
            ),
            "scene_relations": {
                "obstacle_on_ball": bool(blocked),
                "overlap_ball_ratio": float(
                    block_info.get(
                        "overlap_ball_ratio",
                        0.0,
                    )
                ),
            },
        }

        if not execute:
            print(
                f"[ACTION] DRY PLAN: {role} -> box"
            )

            return {
                "success": True,
                "executed": False,
                "plan": plan,
            }

        tag = f"{role.upper()}_TO_BOX"

        print(
            f"[ACTION] EXECUTING: {role} -> box"
        )

        picked = _execute_pick(
            robot,
            grasp6d,
            pre6d,
            role,
            target_width_m=float(
                target_grasp.jaw_width_m
            ),
            dry_run=False,
            tag=tag,
        )

        if picked is None:
            raise RuntimeError(
                f"Failed to pick {role}"
            )

        xg, yg, zg, rxg, ryg, rzg = picked

        print(
            f"[{tag}] Returning to observation "
            "pose while holding object..."
        )

        _move_ready(
            robot,
            ready_cfg,
        )

        print(
            f"[{tag}] Moving above box "
            f"xyz=({box_x:+.3f},"
            f"{box_y:+.3f},"
            f"{drop_z:+.3f})"
        )

        ok = _safe_move_to(
            robot,
            box_x,
            box_y,
            drop_z,
            rxg,
            ryg,
            rzg,
            duration=2.0,
            tag=tag,
        )

        if not ok:
            raise RuntimeError(
                "Move above box failed"
            )

        print(
            f"[{tag}] Releasing {role} into box..."
        )

        ok = _safe_open_gripper(
            robot,
            RELEASE_WIDTH_M,
            tag=tag,
        )

        if not ok:
            raise RuntimeError(
                "Gripper release failed"
            )

        time.sleep(0.4)

        print(
            f"[{tag}] Returning to observation pose..."
        )

        _move_ready(
            robot,
            ready_cfg,
        )

        return {
            "success": True,
            "executed": True,
            "plan": plan,
        }


@app.post("/move-object-to-box")
def move_object_to_box_route():
    data = request.get_json(
        silent=True
    ) or {}

    role = str(
        data.get("object", "")
    ).strip().lower()

    # Only literal JSON true enables physical motion.
    execute = data.get(
        "execute",
        False,
    ) is True

    try:
        result = move_object_to_box_action(
            role=role,
            execute=execute,
        )

        return jsonify(result)

    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "executed": False,
                    "error": str(exc),
                }
            ),
            500,
        )



def preview_loop() -> None:
    """
    Two-window live preview.

    Live RGB Clean:
        always displays current Gemini RGB, including while
        the agent is physically moving the robot.

    BallBoxObstacle:
        runs perception only while state_lock is available.
        During an agent action it intentionally keeps the last
        analyzed frame to avoid interfering with manipulation.
    """
    assert cam is not None
    assert model is not None
    assert yolo_opts is not None
    assert K is not None

    window_name = "BallBoxObstacle — Agentic Pick Place"
    clean_window_name = "Live RGB Clean"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL,
    )

    cv2.namedWindow(
        clean_window_name,
        cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL,
    )

    last_clean = None
    last_display = None

    print("[PREVIEW] Two live windows enabled")
    print("[PREVIEW] Clean RGB remains LIVE during agent actions")
    print("[PREVIEW] Q / ESC = stop Agent API")

    try:
        while True:

            # ------------------------------------------------
            # ALWAYS acquire a fresh camera frame.
            #
            # camera_lock protects Gemini from simultaneous
            # access by Flask API and preview.
            # ------------------------------------------------

            color_bgr, depth_mm = _get_camera_frame()

            if color_bgr is not None:
                last_clean = color_bgr.copy()

            # ------------------------------------------------
            # Annotated perception window.
            #
            # Only update YOLO / grasp analysis when the main
            # manipulation state is not locked by an API action.
            # ------------------------------------------------

            perception_updated = False

            if color_bgr is not None and depth_mm is not None:

                acquired = state_lock.acquire(blocking=False)

                if acquired:
                    try:
                        _, grasps, dets, blocked, _info = _detect_scene(
                            model,
                            yolo_opts,
                            color_bgr,
                            depth_mm,
                            K,
                            depth_quantile,
                        )

                        best = (
                            _select_closest_grasp_by_role(
                                grasps,
                                "obstacle",
                            )
                            if blocked
                            else _select_closest_grasp_by_role(
                                grasps,
                                "ball",
                            )
                        )

                        if best is None:
                            best = _select_fallback_grasp(
                                grasps
                            )

                        display = _render_display(
                            color_bgr,
                            grasps,
                            best,
                            dets,
                            blocked,
                            "AGENT API READY",
                        )

                        last_display = display.copy()
                        perception_updated = True

                    finally:
                        state_lock.release()

            # ------------------------------------------------
            # CLEAN WINDOW
            #
            # This is deliberately independent of state_lock.
            # It therefore stays live while the robot moves.
            # ------------------------------------------------

            if last_clean is not None:
                cv2.imshow(
                    clean_window_name,
                    last_clean,
                )

            # ------------------------------------------------
            # ANNOTATED WINDOW
            #
            # Freeze latest perception during manipulation.
            # ------------------------------------------------

            if last_display is not None:

                shown = last_display.copy()

                if not perception_updated:
                    cv2.putText(
                        shown,
                        "AGENT ACTION IN PROGRESS",
                        (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 215, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(
                    window_name,
                    shown,
                )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("[PREVIEW] Stop requested")
                break

            if (
                cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                )
                < 1
            ):
                break

            if (
                cv2.getWindowProperty(
                    clean_window_name,
                    cv2.WND_PROP_VISIBLE,
                )
                < 1
            ):
                break

            time.sleep(0.005)

    finally:
        cv2.destroyAllWindows()


def cleanup() -> None:
    global initialized

    if not initialized:
        return

    print()
    print("[API] Cleaning up...")

    try:
        if robot is not None:
            robot.release_gripper()
    except Exception as exc:
        print("[API] release error:", exc)

    try:
        if robot is not None:
            robot.safe_home()
    except Exception as exc:
        print("[API] home error:", exc)

    try:
        if robot is not None:
            robot.disconnect()
    except Exception as exc:
        print("[API] disconnect error:", exc)

    try:
        if cam is not None:
            cam.close()
    except Exception as exc:
        print("[API] camera error:", exc)

    initialized = False


atexit.register(cleanup)


def _run_api_server() -> None:
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    initialize()

    api_thread = threading.Thread(
        target=_run_api_server,
        name="rebot-agent-api",
        daemon=True,
    )
    api_thread.start()

    print(f"[API] Flask server running on port {PORT}")

    try:
        preview_loop()
    finally:
        cleanup()
