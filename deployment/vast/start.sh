#!/bin/bash
# Start everything for the demo (run as user carla). Safe to re-run: skips what is already up.
# CARLA flags: no -quality-level=Low (it crashes with RGB+depth cameras in sync mode on this box);
# -game + .uproject make the project accept the packaged server as its own. Perception demo on CPU
# because the driving agent refuses to start while another process uses the GPU.
cd ~
up() { tmux has-session -t "$1" 2>/dev/null; }
up carla || tmux new -d -s carla "cd ~/carla-0.9.16 && ./CarlaUE4.sh -game /home/carla/carla_selfdrivingcar/Unreal/CarlaUE4/CarlaUE4.uproject -RenderOffScreen -nosound -carla-rpc-port=2000 > ~/carla.log 2>&1"
up proxy || tmux new -d -s proxy "caddy run --config ~/demo/Caddyfile --adapter caddyfile > ~/demo/caddy.log 2>&1"
up perception || tmux new -d -s perception "cd ~/carla_selfdrivingcar && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 16-23 venv/bin/python perception/demo.py --device cpu --port 8773 > ~/demo/perception.log 2>&1"
if [ -f ~/carla_selfdrivingcar/final/viewer.json ] && [ -f ~/carla_selfdrivingcar/final/latest.pt ] && ! up viewer; then
  cd ~/carla_selfdrivingcar
  venv/bin/python ~/demo/load_map.py || exit 1
  tmux new -d -s viewer "cd ~/carla_selfdrivingcar && OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 taskset -c 0-15 venv/bin/python -m dreamerv3_attention viewer --config final/viewer.json --checkpoint final/latest.pt --budget-seconds 1800 > ~/demo/viewer.log 2>&1"
fi
tmux ls
