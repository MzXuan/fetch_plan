# fetch planner

---
## Environment

* python3.6
* tensorflow==1.12

---
## Install
1. install the latest [gym](https://github.com/openai/gym)
 version

2. install [mujoco-py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)

3. setup environment
``` shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-396
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$YOUR_HOME_DIR/.mujoco/mjpro150/bin
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-396/libGL.so
```

4. copy gym files to your gym directory
``` shell
cp gym_file/jointvel.xml to $GYM_PATH/gym/envs/robotics/assets/fetch
cp gym_file/shared_xz.xml to $GYM_PATH/gym/envs/robotics/assets/fetch
```

5. install baselines
``` shell
cd baselines
pip install -e .
```

---
## Test
run test script as follows
``` shell
python env_test.py
```
---
## Run
``` shell
cd baselines/baselines/ppo2
python run.py 
```

---
## Change log
1. 0.1.0
* complete environment test

2. 0.2.0
* complete reward function for env
* complete reset function for env

3. 0.3.0
* add reinforcement learning code to train fetch
* complete no predictable reward training