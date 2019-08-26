import os, sys
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

from predictors import ShortPred
from predictors import LongPred
from create_traj_set import RLDataCreator
from tqdm import tqdm
import pred_flags


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, iter=0):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, iter=iter, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps,  iter=iter, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss + entropy * ent_coef + vf_loss * vf_coef
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, 
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="model"
                )
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="model"
                )
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class Runner(object):
    def __init__(self, *, env, model, 
                 nsteps, gamma, lam, load, point, 
                predictor_flag=False, pred_weight=0.01):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.env_obs = np.copy(self.obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.predictor_flag = predictor_flag
        self.pred_weight = pred_weight
        self.dones = [False for _ in range(nenv)]

        # self.short_term_predictor = ShortPred(nenv, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps = pred_flags.out_steps,
        #                            train_flag=False)

        self.long_term_predictor = LongPred(nenv, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=pred_flags.out_steps,
                              train_flag=True)

        self.dataset_creator = RLDataCreator(nenv)

        self.pred_obs = np.zeros((nenv, pred_flags.num_layers*pred_flags.num_units))
        self.pred_result = [np.zeros(3) for _ in range(nenv)]


        self.collect_flag = False
        if load:
            self.model.load("{}/checkpoints/{}".format(logger.get_dir(), point))
            # self.predictor.load()

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_weighted_ploss, mb_origin_ploss, mb_origin_rew = [],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            #----- edit obs, make it env_obs + pred_obs
            # print("obs shape is: ", self.obs.shape)
            # expand obs with latent space
            self.obs = np.concatenate((self.env_obs, self.pred_obs), axis=1)
            # print("obs shape after edited is: ", self.obs.shape)

            #--------end extend obs------
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.env_obs, rewards, self.dones, infos = self.env.step(actions)
            self.obs = np.concatenate((self.env_obs, self.pred_obs), axis=1)


            mb_origin_rew.append(np.mean(np.asarray(rewards)))


            #---- predict reward-------
            pred_weight = self.pred_weight
            if self.predictor_flag and pred_weight != 0.0: #predict process
                origin_obs = self.env.origin_obs
                xs, goals = self.dataset_creator.collect_online(origin_obs, self.dones)

                #-----short term prediction-----
                # origin_pred_loss = self.short_term_predictor.run_online_prediction(xs)

                #-----------------------------------------------------------------
                # predict and get new latent space every n steps
                # or update latent space based on last prediction
                # calculate loss based on previous calculated result
                # ----------------------------------------------------------
                self.pred_obs[:], pred_result, origin_pred_loss = \
                    self.long_term_predictor.run_online_prediction(xs, self.pred_obs, self.pred_result)

                predict_loss = pred_weight * origin_pred_loss
                rewards -= predict_loss

                #---for display---
                # print("predict_loss: {}".format(predict_loss))
                # print("final_reward: {}".format(rewards))
                #---------------

            elif pred_weight != 0.0 and self.collect_flag is not True: #collect process
                origin_obs = self.env.origin_obs   # [achieved goal; true goal; joint obs states]
                self.collect_flag =  self.dataset_creator.collect(origin_obs, self.dones, infos)
                origin_pred_loss = 0.0
                predict_loss = 0.0
            else:
                origin_pred_loss = 0.0
                predict_loss = 0.0

            rewards = self.env.normalize_rew(rewards)
            mb_rewards.append(rewards)
            mb_origin_ploss.append(np.mean(np.asarray(origin_pred_loss)))
            mb_weighted_ploss.append(np.mean(np.asarray(predict_loss)))
            
            # mb_traj_len.append(np.nanmean(np.asarray(traj_len)))

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)


        if self.pred_weight != 0.0 and self.collect_flag is True:
            print("transfer raw data in to delta x, please wait....")
            self.dataset_creator.get_mean_std()


        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0        
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), 
            mb_states, mb_origin_ploss, mb_weighted_ploss, mb_origin_rew, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=50, load=False, point='00100', init_targ=0.1,
            predictor_flag=False, pred_weight=0.01, iter=0):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, iter=iter)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(
        env=env, model=model, 
        nsteps=nsteps, gamma=gamma, lam=lam, load=load, point=point,
        predictor_flag=predictor_flag, pred_weight=pred_weight)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    lrnow = 3e-4
    kl = 0.01

    #test weight parameter
    print("current pred_weight")
    print(pred_weight)
    if pred_weight!=0:
        loss = []
        rew = []
        print("finding best pred weight... this will take 2 epochs...")
        for _ in tqdm(range(2)):
            print("start finding...")
            obs, returns, masks, actions, values, neglogpacs, states, origin_ploss, pred_loss, origin_rew, epinfos = runner.run()  # pylint: disable=E0632
            loss.append(origin_ploss)
            rew.append(origin_rew)

        runner.pred_weight = np.mean(rew)/np.mean(loss) * (pred_weight)
        print("current pred weight is: ")
        print(runner.pred_weight)

    # learning
    nupdates = total_timesteps//nbatch
    for update in tqdm(range(1, nupdates+1)):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        curr_step = update*nbatch
        step_percent = float(curr_step / total_timesteps)

        if step_percent < 0.1:
            d_targ = init_targ
        elif step_percent < 0.4:
            d_targ = init_targ / 2.
        else:
            d_targ = init_targ / 4.

        # d_targ = init_targ

        lrnow = lr(lrnow, kl, d_targ)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, origin_ploss, pred_loss, origin_rew, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))            

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        kl = lossvals[3]
        # print ("kl: {}".format(kl))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            logger.logkv('origin_pred_loss', np.mean(origin_ploss))
            logger.logkv('weighted_pred_loss', np.mean(pred_loss))
            logger.logkv('origin_rew', np.mean(origin_rew))
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            np.save('{}/ob_mean'.format(logger.get_dir()), runner.env.ob_rms.mean)
            np.save('{}/ob_var'.format(logger.get_dir()), runner.env.ob_rms.var)
            np.save('{}/ob_count'.format(logger.get_dir()), runner.env.ob_rms.count)
            np.save('{}/ret_mean'.format(logger.get_dir()), runner.env.ret_rms.mean)
            np.save('{}/ret_var'.format(logger.get_dir()), runner.env.ret_rms.var)

    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)
    savepath = osp.join(checkdir, 'last')
    print('Saving to', savepath)
    model.save(savepath)
    np.save('{}/ob_mean'.format(logger.get_dir()), runner.env.ob_rms.mean)
    np.save('{}/ob_var'.format(logger.get_dir()), runner.env.ob_rms.var)
    np.save('{}/ob_count'.format(logger.get_dir()), runner.env.ob_rms.count)
    np.save('{}/ret_mean'.format(logger.get_dir()), runner.env.ret_rms.mean)
    np.save('{}/ret_var'.format(logger.get_dir()), runner.env.ret_rms.var)

    env.close()

def test(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=50, load=True, point='00200', init_targ=0.1,
            predictor_flag=False):

    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    model = make_model()
    runner = Runner(
        env=env, model=model, 
        nsteps=nsteps, gamma=gamma, lam=lam, load=False, point=point,
        predictor_flag=predictor_flag)

    def load_net(load_path):
        sess = tf.get_default_session()
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="model"
        )
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)

    curr_path = sys.path[0]
    load_path = '{}/log/checkpoints/{}'.format(curr_path, point)
    load_net(load_path)
    
    while not runner.collect_flag:
        runner.run() #pylint: disable=E0632
        
    env.close()

def display(policy, env, nsteps, nminibatches, load_path):
    nenv = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenv * nsteps
    nbatch_train = nbatch // nminibatches

    sess = tf.get_default_session()
    act_model = policy(sess, ob_space, ac_space, 1, 1, reuse=False)
    train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)
    params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="model"
        )

    predictor = ShortPred(nenv, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=pred_flags.out_steps,
                               train_flag=False, model_name=pred_flags.model_name)

    dataset_creator = RLDataCreator(nenv)

    def load(load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)

    load(load_path)

    def run_episode(env, agent):
        import visualize

        obs = env.reset()
        env_obs = np.copy(obs)
        pred_obs = np.zeros((1, pred_flags.num_units * pred_flags.num_layers))

        score = 0
        done = [False]
        state = agent.initial_state
        obs_list = None
        obs_list_3d = None


        traj = []
        env.render()
        time.sleep(2)

        while not done[0]:
            env.render()
            obs = np.concatenate((env_obs,pred_obs), axis=1)
            act, state = agent.mean(obs, state, done)
            env_obs, rew, done, info = env.step(act)
            obs = np.concatenate((env_obs, pred_obs), axis=1)

            # print("goal: ", obs[0][3:6])
            # print("eef position:", obs[0][0:3])

            origin_obs = env.origin_obs
            traj.append(origin_obs[0][0:3])

            xs, goals = dataset_creator.collect_online(origin_obs, done)
            origin_pred_loss = predictor.run_online_prediction(xs, goals)


            # #---- plot result ---
            #
            # obs_list = visualize.plot_obs(
            #     env.origin_obs, obs_list)
            #
            # obs_list_3d = visualize.plot_3d_obs(
            #     env.origin_obs, obs_list_3d)
            #
            # #--- end plot ---#
            score += rew[0]

        #if done, save trajectory
        traj = np.asarray(traj)
        # if done, pause 2 s
        time.sleep(1)
        return score, traj


    for e in range(10000):
        score, traj = run_episode(env, act_model)
        print ('episode: {} | score: {}'.format(e, score))
        # print("episode: {} traj: {}".format(e, traj))
        # np.savetxt("/home/xuan/Videos/trajs/traj_ep_"+str(e)+".csv", traj, delimiter=",", fmt="%.3e")

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)




