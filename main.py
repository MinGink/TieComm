import os
import numpy as np
import random
import time
import yaml
import wandb
import argparse
import sys
import torch
import datetime
import signal
from os.path import dirname, abspath
from envs import REGISTRY as env_REGISTRY, data
from baselines import REGISTRY as agent_REGISTRY
from modules.utils import get_config, recursive_dict_update
from modules import RunnerDual
from modules.multi_processing import MultiPeocessRunner
from modules import Runner, magicRunner

def main(args):

    #======================================load config==============================================
    #defualt config
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)


    #env config
    env_config = get_config(args, args.env, "envs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict['env_args']['key'] = args.env_map

    #algo config
    alg_config = get_config(args, args.algo, "algos")
    config_dict = recursive_dict_update(config_dict, alg_config)

    config_dict = recursive_dict_update(config_dict, vars(args))


    args = argparse.Namespace(**config_dict)
    args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"

    if args.seed == None:
        args.seed = np.random.randint(0, 10000)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.env_args['seed'] = args.seed


    #======================================wandb==============================================
    results_path = os.path.join(dirname(dirname(abspath(__file__))), "../results")
    args.experiment_id = f"{args.memo}_{args.algo}_{args.seed}_{datetime.datetime.now().strftime('%d_%H_%M')}"

    if args.use_offline_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    wandb.init(project='new', name=args.experiment_id, tags=['Dong'], dir=results_path)
    wandb.config.update(args)

    #======================================register environment==============================================
    if args.env == 'traffic_junction':
        env = data.init(args.env, args, False)
        args.obs_shape = env.observation_dim
        args.n_actions = env.num_actions
        args.dim_actions = env.dim_actions

        # Hard attention
        if args.hard_attn and args.commnet:
        # add comm_action as last dim in actions
            args.n_actions = [*args.n_actions, 2]
            args.dim_actions = env.dim_actions + 1
    else:
        env = env_REGISTRY[args.env](**args.env_args)
        env_info = env.get_env_info()
        args.obs_shape = env_info["obs_shape"]
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.episode_length = env_info["episode_limit"]



    agent = agent_REGISTRY[args.algo](args)


    if args.algo == 'tiecomm':
        run = RunnerDual
    else:
        run = Runner


    if args.use_multiprocessing:
        for p in agent.parameters():
            p.data.share_memory_()
        runner = MultiPeocessRunner(args, lambda: run(args, env, agent))
        epoch_size = 1
    elif args.magic==True:
        runner = magicRunner(args, env, agent)
        epoch_size = args.epoch_size
    else:
        runner = run(args, env,agent)
        epoch_size = args.epoch_size



    total_num_episodes = 0
    total_num_steps = 0


    for epoch in range(args.total_epoches):
        epoch_begin_time = time.time()

        log = runner.train_batch(epoch_size)

        total_num_episodes += log['num_episodes']
        total_num_steps += log['num_steps']

        epoch_time = time.time() - epoch_begin_time

        if args.algo == 'tiecomm':
            wandb.log({"epoch": epoch,
                        'episode': total_num_episodes,
                        'epoch_time': epoch_time,
                        'total_steps': total_num_steps,
                        'episode_return': np.mean(log['episode_return']),
                        "episode_steps": np.mean(log['episode_steps']),
                        'action_loss': log['action_loss'],
                        'value_loss': log['value_loss'],
                        'total_loss': log['total_loss'],
                       'god_action_loss': log['god_action_loss'],
                       'god_value_loss': log['god_value_loss'],
                       'god_total_loss': log['god_total_loss'],
                        })
        else:
            wandb.log({"epoch": epoch,
                        'episode': total_num_episodes,
                        'epoch_time': epoch_time,
                        'total_steps': total_num_steps,
                        'episode_return': np.mean(log['episode_return']),
                        "episode_steps": np.mean(log['episode_steps']),
                        'action_loss': log['action_loss'],
                        'value_loss': log['value_loss'],
                        'total_loss': log['total_loss'],
                        })

        print('current epoch: {}/{}'.format(epoch, args.total_epoches))

    if sys.flags.interactive == 0 and args.use_multiprocessing:
        runner.quit()
        os._exit(0)

    print("=====Done!!!=====")



def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    sys.exit(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TieComm')
    parser.add_argument('--memo', type=str, default="TieComm", help='memo')
    parser.add_argument('--env', type=str, default="lbf", help='environment name')
    parser.add_argument('--env_map', type=str, default="lbforaging:Foraging-10x10-3p-3f-v2", help='environment map name')
    parser.add_argument('--algo', type=str, default="tiecomm", help='algorithm name',choices='tiecomm, ac_basic，commnet')
    parser.add_argument('--seed', type=int, default= None, help='random seed')
    parser.add_argument('--use_offline_wandb', action='store_true', help='use offline wandb')
    parser.add_argument('--total_epoches', type=int, default=2000, help='total number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=10, help='epoch size')
    args = parser.parse_args()

    training_begin_time = time.time()
    signal.signal(signal.SIGINT, signal_handler)

    main(args)

    training_time = time.time() - training_begin_time

    print('training time: {} h'.format(training_time/3600))


    #======================================end==============================================
    # Clean up after finishing
    # print("Exiting Main")


    # for t in threading.enumerate():
    #     if t.name != "MainThread":
    #         #print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
    #         t.join(timeout=1)
    #         #print("Thread joined")
    # print("Stopping all threads")
    # # Making sure framework really exits
    # os._exit(os.EX_OK)