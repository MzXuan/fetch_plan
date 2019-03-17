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
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-396`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$YOUR_HOME_DIR/.mujoco/mjpro150/bin`
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-396/libGL.so`

4. copy gym files to your gym directory
`cp gym_file/jointvel.xml to $GYM_PATH/gym/envs/robotics/assets/fetch`
`cp gym_file/shared_xz.xml to $GYM_PATH/gym/envs/robotics/assets/fetch`

---
## Test
run test script as follows

`python env_test.py`

---
## Change log
1. 0.1.0
* complete environment test