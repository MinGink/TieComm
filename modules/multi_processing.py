import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
import random
from .utils import merge_dict
import argparse


class MultiProcessWorker(mp.Process):
    def __init__(self, worker_id, runner, sender, seed, **kwargs):
        super(MultiProcessWorker, self).__init__()

        self.worker_id = worker_id
        self.seed = seed
        self.runner = runner()
        self.worker = sender



    def run(self):

        torch.manual_seed(self.seed + self.worker_id + 1)
        np.random.seed(self.seed + self.worker_id + 1)
        random.seed(self.seed + self.worker_id + 1)

        while True:
            task = self.worker.recv()

            if type(task) == list:
                task, batch_size = task

            if task == 'quit':
                return

            elif task == 'train_batch':
                batch_data, batch_log = self.runner.collect_batch_data(batch_size)
                self.runner.optimizer.zero_grad()
                train_log = self.runner.compute_grad(batch_data)
                merge_dict(batch_log, train_log)
                self.worker.send(train_log)

            elif task == 'send_grads':
                grads = []
                for p in self.runner.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)
                self.worker.send(grads)





class MultiPeocessRunner():
    def __init__(self, config, runner):

        self.args = argparse.Namespace(**config)
        self.batch_size = self.args.batch_size
        self.runner = runner()
        self.n_workers = self.args.n_processes -1

        self.pool = []
        for i in range(self.n_workers):
            reciver, sender = mp.Pipe()
            self.pool.append(reciver)
            worker = MultiProcessWorker(i, runner, sender, seed=self.args.seed)
            worker.start()

        self.grads = None
        self.worker_grads = None


    def quit(self):
        for i in range(self.n_workers):
            self.pool[i].send('quit')


    def train_batch(self,batch_size):
        for worker in self.pool:
            worker.send(['train_batch', batch_size])

        # run its own trainer
        batch_data, batch_log = self.runner.collect_batch_data(batch_size)
        self.runner.optimizer.zero_grad()
        main_log = self.runner.compute_grad(batch_data)
        merge_dict(batch_log, main_log)

        # check if workers are finished
        for worker in self.pool:
            worker_log = worker.recv()
            merge_dict(worker_log, main_log)


        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                try:
                    self.grads[i] += g[i]
                except:
                    pass
            self.grads[i] /= main_log['num_steps']

        #nn.utils.clip_grad_norm_(self.runner.params, self.args.grad_norm_clip)
        self.runner.optimizer.step()
        return main_log


    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.runner.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for reciver in self.pool:
                reciver.send('send_grads')
                self.worker_grads.append(reciver.recv())