# fetch planner

---
## Details need to be determinted
+ change loss function (reference to huzhe's code)
+ Analysis the performance of predictor training
+ Unify the coordinate of mujoco and ros fetch simulation
+ joint training

---
## TODO List
+ Reduce RL traning steps
+ Baseline training
+ Whether fine-tuning for predictor training
+ Whether smooth training process (two datasets)
+ Different predictor network sizes
+ Predict only end-effector
+ GUI (@xuanz)

---
## Environment

* python3.6
* tensorflow==1.12

---
## Install
1. install the latest [gym](https://github.com/openai/gym)
 version

2. install [mujoco-py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key) == 1.5.1.1

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

## Run
1. Download pretrained [model](https://www.dropbox.com/s/xngkz330rnw70f8/models.zip?dl=0)

2. joint training RL policy with seq2seq predictor
``` shell
bash train_cycle.sh ${ITER_STEP} ${PRED_WEIGHT}
``` 

3. visualize rl training process
``` shell
python results_plotter.py --log_num=${ITER_STEP}
```

---
## Unit Test
1. Env code
``` shell
python env_test.py
```

2. RL code
``` shell
cd baselines/baselines/ppo2
python run.py 
```

For training policy, please set
``` shell
--train=True
--display=False
--load=False
```

For sampling dataset, please set
``` shell
--train=False
--display=False
--load=True
--point="$YOUR_CHECKPOINT_NUMBER"
```

For displaying performance, please set
``` shell
--train=False
--display=True
--load=True
--point="$YOUR_CHECKPOINT_NUMBER"
```

3. LSTM training code
``` shell
python predictor_new.py

python predictor_new.py --test
```

---
## How to get the observation without normalization
``` python
obs = env.reset()
origin_obs = env.origin_obs
done = False
while not done:
    act = actor.act(obs)
    obs, rew, done, _ = env.step(act)
    origin_obs = env.origin_obs
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

4. 0.3.5
* add visualization of obs in ppo2.py (example in line 389 to 402)

5. 0.3.6
* change prediction to sequence to sequence mode
* use new tensorflow seq2seq api

6. 0.4.0
* add a script for training
* finish two reward framework

7. 0.5.0
* joint training

8. 0.6.0
* smooth traning process (two datasets)
* reset entropy for rl training