import os
import numpy as np
import random
import time
import wandb
import argparse
import sys
import torch
import signal
from os.path import dirname, abspath
from envs import REGISTRY as env_REGISTRY
from baselines import REGISTRY as agent_REGISTRY
from runner import REGISTRY as runner_REGISTRY
from modules.multi_processing import MultiPeocessRunner
from configs.utils import get_config, recursive_dict_update, signal_handler, merge_dict




def main(args):

    default_config = get_config('experiment')
    env_config = get_config(args.env, 'envs')
    agent_config = get_config(args.agent, 'agents')

    if args.seed == None:
        args.seed = np.random.randint(0, 10000)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_config['seed'] = args.seed



    #update configs
    exp_config = recursive_dict_update(default_config,vars(args))
    env_config = recursive_dict_update(env_config, vars(args))
    agent_config = recursive_dict_update(agent_config, vars(args))


    #merge config
    config = {}
    config.update(default_config)
    config.update(env_config)
    config.update(agent_config)

    #======================================load config==============================================
    args = argparse.Namespace(**config)
    args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"


    #======================================wandb==============================================
    results_path = os.path.join(dirname(abspath(__file__)), "results")
    args.exp_id = f"{args.env}_{args.agent}_{args.memo}" #_{datetime.datetime.now().strftime('%d_%H_%M')}"

    if args.use_offline_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    wandb.init(project='AAAI', name=args.exp_id, tags=['Ming'], dir=results_path)
    wandb.config.update(args)


    #======================================register environment==============================================
    env = env_REGISTRY[args.env](env_config)

    env_info = env.get_env_info()
    agent_config['obs_shape'] = env_info["obs_shape"]
    agent_config['n_actions'] = env_info["n_actions"]
    agent_config['n_agents'] = env_info["n_agents"]
    exp_config['episode_length'] = env_info["episode_length"]
    exp_config['n_agents'] = env_info["n_agents"]
    agent = agent_REGISTRY[args.agent](agent_config)


    epoch_size = exp_config['epoch_size']
    run = runner_REGISTRY[args.agent]
    if args.use_multiprocessing:
        for p in agent.parameters():
            p.data.share_memory_()
        runner = MultiPeocessRunner(exp_config, lambda: run(exp_config, env, agent))
        epoch_size = 1
    else:
        runner = run(exp_config, env, agent)




    total_num_episodes = 0
    total_num_steps = 0

    for epoch in range(args.total_epoches):
        epoch_begin_time = time.time()

        log = runner.train_batch(epoch_size)

        total_num_episodes += log['num_episodes']
        total_num_steps += log['num_steps']

        epoch_time = time.time() - epoch_begin_time

        if args.agent == 'tiecomm':
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TieComm')
    parser.add_argument('--memo', type=str, default="debug", help='memo')
    parser.add_argument('--env', type=str, default="rware", help='environment name',
                        choices=['mpe','lbforaging','rware','tj'])
    parser.add_argument('--map', type=str, default="rware-1color-medium-6ag-hard-v1", help='environment map name')
    parser.add_argument('--agent', type=str, default="tiecomm", help='algorithm name',
                        choices=['tiecomm','tiecomm_random','tiecomm_no', 'ac_basic','commnet'])
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--use_offline_wandb', action='store_true', help='use offline wandb')
    parser.add_argument('--use_multiprocessing', action='store_true', help='use multiprocessing')
    parser.add_argument('--total_epoches', type=int, default=2000, help='total number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=2, help='epoch size')
    args = parser.parse_args()

    training_begin_time = time.time()

    signal.signal(signal.SIGINT, signal_handler)
    main(args)
    training_time = time.time() - training_begin_time
    print('training time: {} h'.format(training_time/3600))
