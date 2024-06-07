import os
from datetime import datetime

import h5py
import copy
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler

from util.functions import cycle, EMA
from util.datasets import DemoDataset

class SMILETrainer(object):
    def __init__(
        self,
        smile,
        env,
        device,
        args,
    ):
        super().__init__()
        self.smile = smile
        self.device= device
        self.env_name=env.spec.id
        self.env=env

        self.ema = EMA(args.ema_decay)
        self.ema_smile = copy.deepcopy(self.smile)
        self.update_ema_every = args.update_ema_every

        self.denoiser_lr = args.denoiser_lr
        self.policy_lr = args.policy_lr
        self.ema_decay=args.ema_decay

        self.step_start_ema = args.step_start_ema
        self.eval_every = args.eval_every
        self.filter_every = args.filter_every

        self.batch_size = args.batch_size
        self.policy_update_iter = args.policy_update_iter
        self.denoiser_update_iter = args.denoiser_update_iter
        self.num_samples = args.num_samples

        self.no_filtering = args.no_filtering
        self.stop_filtering = args.no_filtering
        self.stop_epsilon = False
        self.zero_sigma = args.zero_sigma
        self.naive_reverse = args.naive_reverse
        self.important_sampling = False

        self.demo_path = args.demo_path

        self.ds = self.load_demos(self.demo_path)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, sampler= None, shuffle=True, pin_memory=True))
        self.denoise_opt = Adam(self.smile.denoiser.parameters(), lr=self.denoiser_lr)
        self.policy_opt = Adam(self.smile.policy.parameters(), lr=self.policy_lr)

        self.step = 0

        self.results_folder = Path('./results')
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters(self.ema_smile,self.smile)

        self.start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    def reset_parameters(self,ema_md,md):
        ema_md.load_state_dict(md.state_dict())

    def load_demos(self, path):

        print('loading dataset...')
        dataset = DemoDataset()

        hf = h5py.File(path, 'r')
        states = np.stack(hf.get('states'))
        actions = np.stack(hf.get('actions'))
        rewards = np.stack(hf.get('rewards')).squeeze(-1)
        terminals = np.stack(hf.get('dones')).squeeze(-1)
        sar = list(
            [states[i], actions[i], rewards[i], terminals[i]] for i in range(len(states)))
        dataset.demo_list += sar


        rew_sum, start, num_demo = 0, 0, 0
        demo_info = []
        for i, (s, a, r, t) in enumerate(dataset.demo_list):
            rew_sum += r
            if t == True or i + 1 - start >= 1000:
                num_demo += 1
                demo_info.append((rew_sum, i + 1 - start))
                rew_sum, start = 0, i + 1

        print('Total number of demonstrations is', num_demo, ', the info of them is', demo_info, ', total number of state-action pairs is', len(dataset.demo_list))

        return dataset

    def step_ema(self,ema_md,md):
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_md,md)
            return
        self.ema.update_model_average(ema_md, md)

    @torch.no_grad()
    def eval(self,smile):
        length=0
        obs = self.env.reset()
        done = False
        reward_sum , cnt_succ= 0,0
        while not done:
            length+=1
            if not self.naive_reverse:
                action = smile.policy.sample_action(obs)
            else:
                action = smile.sample_action(torch.tensor(obs).to(self.device))
            obs, reward, done, info = self.env.step(action)
            if 'goal_achieved' in info.keys():
                if info['goal_achieved'] is True:
                    cnt_succ+=1
                    if(cnt_succ>20):
                        reward_sum=1
            else:
                reward_sum += reward

        return reward_sum,length

    @torch.no_grad()
    def filter_demos(self,data,cond_Q):

        rew_sum, start, demos, demo_rews = 0, 0, [], []
        weights=[]

        for i in range(len(data)):
            rew_sum += data[i][2]
            if data[i][3] == True or i + 1 - start >= 1000:
                argmax_t = torch.argmax(torch.mean(cond_Q[start:i + 1], dim=0).cpu())

                if argmax_t > 0 :
                    demos += data[start:i + 1]
                    weights += [argmax_t.item() for _ in range(i+1-start)]
                    demo_rews.append(rew_sum)

                start = i + 1
                rew_sum = 0

        return demos,len(demo_rews),weights

    @torch.no_grad()
    def filter_dataset(self, smile):

        states = torch.tensor([demo[0] for demo in self.ds.demo_list]).to(self.device)
        actions = torch.tensor([demo[1] for demo in self.ds.demo_list]).to(self.device)

        if not self.naive_reverse:
            a_policy = smile.policy.sample_action(states.cpu().numpy())
        else:
            a_policy = smile.sample_action(states)
        a_policy = torch.tensor(a_policy).to(self.device)

        cond_Q = []
        if self.zero_sigma:
            cond_Q += [-1.0 * torch.sum((actions - a_policy) ** 2, dim=1)]
        t_prime = torch.zeros((states.shape[0],), dtype=torch.int64).to(self.device)
        for _ in range(smile.num_timesteps):
            cond_Q += [smile.compute_Q(states, actions, a_policy, t_prime)]
            t_prime += 1
        cond_Q = torch.stack(cond_Q).T

        demos, num_demos, weights = self.filter_demos(self.ds.demo_list, cond_Q)

        if not self.important_sampling:
            weights = []

        if num_demos >= 10:
            self.ds.demo_list = demos

            sampler,shuffle = None,True
            if weights != []:
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_demos, replacement=True)
                shuffle=None

            self.dl = cycle(
                data.DataLoader(self.ds, batch_size=self.batch_size,sampler=sampler, shuffle=shuffle, pin_memory=True))
        else:
            self.stop_epsilon=True
            self.stop_filtering=True


    def train(self):

        while self.step * self.batch_size < self.num_samples:

            state, action, reward = next(self.dl)
            state , action = state.to(self.device) , action.to(self.device)

            if not self.stop_epsilon:

                for i in range(self.denoiser_update_iter):
                    denoiser_loss = self.smile.denoiser_loss(state, action)
                    (denoiser_loss / (self.denoiser_update_iter)).backward()

                self.denoise_opt.step()
                self.denoise_opt.zero_grad()


            for i in range(self.policy_update_iter):
                policy_loss = self.smile.policy_loss(state, action)
                (policy_loss / (self.policy_update_iter)).backward()

            self.policy_opt.step()
            self.policy_opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                if not self.stop_epsilon:
                    self.step_ema(self.ema_smile, self.smile)
                else:
                    self.step_ema(self.ema_smile.policy, self.smile.policy)


            if self.step != 0 and self.step % self.eval_every == 0:

                rewards,len_sum=[],0
                for i in range(10):
                    reward,len = self.eval(self.ema_smile)
                    rewards.append(reward)
                    len_sum+=len

                writer=SummaryWriter(f"results/log_{self.env_name}_"  
                                     f"denoisierupdateevery{self.denoiser_update_iter}_"
                                     f"policyupdateevery{self.policy_update_iter}_diffstep{self.smile.num_timesteps}_"
                                     f"denoisingloss{self.smile.denoiser_loss_type}_policyloss{self.smile.policy_loss_type}_"
                                     f"batchsize{self.batch_size}_denoiserlr{self.denoiser_lr}_policylr{self.policy_lr}_"
                                     f"emadecay{self.ema_decay}_filterevery{self.filter_every}_"
                                     f"emaevery{self.update_ema_every}_nofilter{self.no_filtering}_"
                                     f"naivereverse{self.naive_reverse}_"
                                     f"runtime{self.start_time}")

                writer.add_scalar("average return",np.mean(rewards),self.step)
                print(self.step * self.batch_size, np.mean(rewards),np.std(rewards),len_sum/10)
                writer.close()

            if self.step != 0 and self.step % self.filter_every == 0 and self.stop_filtering == False:
                self.filter_dataset(self.ema_smile)

            self.step += 1

        print('training completed')