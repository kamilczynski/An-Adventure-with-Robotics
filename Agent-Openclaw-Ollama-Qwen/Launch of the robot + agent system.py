Terminal 1:
sudo xhost +si:localuser:root

docker start -ai b601_grasp_agent

conda activate rebotarm

cd /workspace/JETSON/rebot_grasp_jetson

export DISPLAY=:0
python agent_api.py

Terminal 2:
systemctl is-active ollama || sudo systemctl start ollama

ollama launch openclaw --model qwen3.5-9b-local

Use the reBot arm to put the ball in the box, but if the obstacle is on the ball, put the obstacle in the box first and then the ball.
