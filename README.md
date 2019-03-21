# fetch planner

---
## TODO List
1. Predictor training (@xuan_z)
``` python
class Predictor(object):
    def __init__(self,sess,FLAGS, batch_size, max_timestep, train_flag):
        pass
    
    def run_training(self):
        #load previously saved dataset and train the network
        
    def predict(self, obs, dones, mean, var):
        # function: predict the goal position
        # input: 
        # obs.shape = [batch_size, ob_shape] include joint angle etc.
        # dones.shape = [batch_size]
        # return:
        # batch_loss.shape =[batch_size]
        return batch_loss

```
2. Address smoothness issue (@xuan_z)
3. Add predictable reward (@tingxfan)
4. Two reward structure (@tingxfan)

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
### Test env
run test script as follows
``` shell
python env_test.py
```

## Run
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