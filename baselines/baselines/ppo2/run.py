import argparse
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from baselines import logger, bench

def train(env_id, num_timesteps, seed, d_targ, load, point,
          pred_weight=0.01, ent_coef=0.0, iter=0):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import LstmMlpPolicy, MlpPolicy
    import gym
    import gym_fetch
    import multiprocessing
    import tensorflow as tf
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk

    set_global_seeds(seed)

    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    
    nenvs = 32
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    if load:
        curr_path = sys.path[0]
        ob_mean = np.load('{}/log/ob_mean.npy'.format(curr_path))
        ob_var = np.load('{}/log/ob_var.npy'.format(curr_path))
        ob_count = np.load('{}/log/ob_count.npy'.format(curr_path))
        ret_mean = np.load('{}/log/ret_mean.npy'.format(curr_path))
        ret_var = np.load('{}/log/ret_var.npy'.format(curr_path))
        ret_count = 10

        env = VecNormalize(env,
            ob_mean=ob_mean, ob_var=ob_var, ob_count=ob_count,
            ret_mean=ret_mean, ret_var=ret_var, ret_count=ret_count)
    else:
        env = VecNormalize(env)

    policy = MlpPolicy

    def constant_lr(lr, kl=0.0, d_targ=0.0):
        return lr

    ppo2.learn(policy=policy, env=env, nsteps=400, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=15, log_interval=1,
        ent_coef=ent_coef,
        lr=constant_lr,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        load=load,
        point=point,
        init_targ=d_targ,
        predictor_flag=True,
        pred_weight=pred_weight,
        iter=iter)

def test(env_id, num_timesteps, seed, d_targ, load, point):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalizeTest
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import LstmMlpPolicy, MlpPolicy
    import gym
    import gym_fetch
    import tensorflow as tf
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    ncpu = 16
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
            env.seed(seed + rank + 100)
            return env
        return _thunk

    curr_path = sys.path[0]
    nenvs = 32
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    ob_mean = np.load('{}/log/ob_mean.npy'.format(curr_path))
    ob_var = np.load('{}/log/ob_var.npy'.format(curr_path))
    env = VecNormalizeTest(env, ob_mean, ob_var)

    set_global_seeds(seed + 100)
    policy = MlpPolicy

    def constant_lr(lr, kl=0.0, d_targ=0.0):
        return lr

    ppo2.test(policy=policy, env=env, nsteps=400, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=15, log_interval=1,
        ent_coef=0.00,
        lr=constant_lr,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        load=True,
        point=point,
        init_targ=d_targ,
        predictor_flag=False)

def display(env_id, num_timesteps, seed, curr_path, log_file, point):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalizeTest
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import LstmMlpPolicy, MlpPolicy
    import gym
    import gym_fetch
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecTestEnv

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
        return env
    env = DummyVecTestEnv([make_env])
    ob_mean = np.load('{}/{}/ob_mean.npy'.format(curr_path, log_file))
    ob_var = np.load('{}/{}/ob_var.npy'.format(curr_path, log_file))
    env = VecNormalizeTest(env, ob_mean, ob_var)

    set_global_seeds(seed)
    policy = MlpPolicy

    ppo2.display(policy=policy, env=env, nsteps=2048, nminibatches=32, 
        load_path='{}/{}/checkpoints/{}'.format(curr_path, log_file, point))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='FetchPlan-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=100)
    parser.add_argument('--num-timesteps', type=int, default=int(1.8e6))
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--d_targ', type=float, default=0.012)
    parser.add_argument('-p', '--point', type=str, default='last')
    parser.add_argument('--pred_weight', default=0.01, type=float)
    parser.add_argument('--ent_coef', default=0.0, type=float)
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    args = parser.parse_args()

    each_iter_num = 500

    curr_path = sys.path[0]
    if args.display:
        display(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            curr_path=curr_path, log_file = args.logdir, point=args.point)
    elif args.train:
        logger.configure(dir='{}/log'.format(curr_path), format_strs=['stdout',
                                                                      'log',
                                                                      'csv',
                                                                      'tensorboard'])
        iter_countings = 800+ (args.iter-1) * each_iter_num if  args.iter >=1 else 0
        logger.tb_start_step(iter_countings , 3)

        print("iter countings: ", iter_countings)
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            d_targ=args.d_targ, load=args.load, point=args.point,
              pred_weight=args.pred_weight, ent_coef=args.ent_coef, iter=args.iter)
    else:
        print("test branch, collecting data....")
        test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            d_targ=args.d_targ, load=True, point=args.point)



if __name__ == '__main__':
    main()
    # script for testing
    # python run.py --display --env=FetchPlanTest-v0 --log-file=log_0

