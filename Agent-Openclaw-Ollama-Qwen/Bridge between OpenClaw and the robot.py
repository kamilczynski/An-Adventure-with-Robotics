mkdir -p ~/.local/bin


cat > ~/.local/bin/robot-health <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
curl -fsS \
  http://127.0.0.1:8000/health | jq .
EOF
chmod +x ~/.local/bin/robot-health

cat > ~/.local/bin/robot-observe <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
curl -fsS \
  -X POST \
  http://127.0.0.1:8000/observe \
  -H 'Content-Type: application/json' \
  -d '{}' | jq .
EOF
chmod +x ~/.local/bin/robot-observe

cat > ~/.local/bin/robot-move-to-box <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
OBJECT="${1:-}"
if [[ "$OBJECT" != "ball" && "$OBJECT" != "obstacle" ]]; then
    echo "Usage: robot-move-to-box <ball|obstacle>" >&2
    exit 1
fi
curl -fsS \
  -X POST \
  http://127.0.0.1:8000/move-object-to-box \
  -H 'Content-Type: application/json' \
  -d "{\"object\":\"${OBJECT}\",\"execute\":true}" | jq .
EOF
chmod +x ~/.local/bin/robot-move-to-box

openclaw config set tools.profile coding

openclaw config set tools.exec.mode auto

openclaw config set tools.exec.host gateway

mkdir -p ~/.openclaw/workspace/skills/rebot-vision

cat > ~/.openclaw/workspace/skills/rebot-vision/SKILL.md <<'EOF'
---
name: rebot-vision
description: Observe the physical reBot B601 workspace using Gemini 336, YOLO and depth perception.
---
# reBot Vision
Use this skill when the user asks about the current physical scene observed by the reBot B601 robot.
The robot perception system can detect:
- ball
- box
- obstacle
It can also report:
- detection confidence
- object image position
- estimated 3D position
- grasp availability
- whether an obstacle overlaps or blocks the ball
## Robot status
To check whether the robot perception system is running, use the exec tool to run:
$HOME/.local/bin/robot-health
## Observe current scene
To obtain the current physical scene, use the exec tool to run:
$HOME/.local/bin/robot-observe
The command returns JSON from the live Gemini 336 RGB-D camera, YOLO detector and grasp perception pipeline.
Important JSON fields include:
objects.ball.detected
objects.box.detected
objects.obstacle.detected
objects.ball.confidence
objects.box.confidence
objects.obstacle.confidence
objects.ball.grasp
objects.box.grasp
objects.obstacle.grasp
relations.obstacle_on_ball
relations.overlap_ball_ratio
relations.obstacle_center_inside_ball
Treat the returned JSON as the authoritative current state of the physical workspace.
If the user asks what the robot currently sees, always run robot-observe.
Do not answer from memory.
Do not invent positions, detections or spatial relationships.
This skill is perception-only.
EOF

mkdir -p ~/.openclaw/workspace/skills/rebot-manipulation

cat > ~/.openclaw/workspace/skills/rebot-manipulation/SKILL.md <<'EOF'
---
name: rebot-manipulation
description: Control the physical reBot B601 using live perception and high-level manipulation actions.
---
# reBot Manipulation
Use this skill when the user asks the physical reBot B601 to manipulate detected objects.
Available high-level commands:
$HOME/.local/bin/robot-observe
$HOME/.local/bin/robot-move-to-box ball
$HOME/.local/bin/robot-move-to-box obstacle
## Rules
Always observe the current scene before deciding on a physical action.
Treat robot-observe output as the authoritative state of the physical workspace.
Do not invent object locations or spatial relationships.
Do not calculate or command joint positions, IK solutions or raw XYZ robot coordinates.
Python handles perception geometry, transforms, grasp planning, IK and motor control.
You decide which high-level action should happen and in what order.
## Put the ball in the box
When the user asks to put the ball in the box:
1. Run robot-observe.
2. Check that the ball and box are detected.
3. Check relations.obstacle_on_ball.
4. If obstacle_on_ball is true and an obstacle is detected:
   run:
   $HOME/.local/bin/robot-move-to-box obstacle
5. Observe the scene again:
   $HOME/.local/bin/robot-observe
6. If the ball and box are available, run:
   $HOME/.local/bin/robot-move-to-box ball
7. Observe the scene again to verify the result.
Always use a fresh observation after a physical action before deciding what to do next.
EOF

openclaw skills list

