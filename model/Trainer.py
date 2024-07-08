import os
from loguru import logger
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
from util.consts import *

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
        self.args=args

        self.ema = EMA(args.ema_decay)
        self.ema_smile = copy.deepcopy(self.smile)
        self.update_ema_every = args.update_ema_every

        self.denoiser_lr = args.denoiser_lr
        self.policy_lr = args.policy_lr
        self.ema_decay=args.ema_decay

        self.step_start_ema = args.step_start_ema
        self.eval_every = args.eval_every
        self.filter_every = args.filter_every
        self.threshold=args.threshold

        self.batch_size = args.batch_size
        self.policy_update_iter = args.policy_update_iter
        self.denoiser_update_iter = args.denoiser_update_iter
        self.pretrain_num_samples = args.pretrain_num_samples
        self.num_samples = args.num_samples

        self.no_filtering = args.no_filtering
        self.stop_filtering = args.no_filtering
        self.stop_epsilon = False
        self.zero_sigma = args.zero_sigma
        self.naive_reverse = args.naive_reverse
        self.important_sampling = False

        self.demo_path = args.demo_path
        
        self.results_folder = Path('./results')
        self.results_folder.mkdir(exist_ok = True)

        self.start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.log_path=f"results/log_{self.env_name}_runtime{self.start_time}"
        logger.add(self.log_path+"/logging.log")
        logger.info(self.args)
        self.writer=SummaryWriter(self.log_path)
        
        if self.env_name in D4RL_ENVS:
            self.ds = self.load_demos_from_d4rl(self.env)
        elif self.env_name in MUJOCO_ENVS:
            self.ds = self.load_demos(self.demo_path)
        else:
            return NotImplementedError
        
        self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, sampler= None, shuffle=True, pin_memory=True))
        self.denoise_opt = Adam(self.smile.denoiser.parameters(), lr=self.denoiser_lr)
        self.policy_opt = Adam(self.smile.policy.parameters(), lr=self.policy_lr)

        self.step = 0

        self.reset_parameters(self.ema_smile,self.smile)


    def reset_parameters(self,ema_md,md):
        ema_md.load_state_dict(md.state_dict())

    def load_demos(self, path):

        logger.info('loading dataset...')
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

        logger.info('Total number of demonstrations is', num_demo, ', the info of them is', demo_info, ', total number of state-action pairs is', len(dataset.demo_list))

        return dataset
    
    def load_demos_from_d4rl(self, env):

        logger.info('loading dataset...')
        d4rl_dataset = env.get_dataset()
        dataset = DemoDataset()

        states = d4rl_dataset['observations']
        actions = d4rl_dataset['actions']
        rewards = d4rl_dataset['rewards']
        terminals = d4rl_dataset['terminals']
        sar = list(
            [states[i], actions[i], rewards[i], terminals[i]] for i in range(len(states)))
        
        rew_sum, start, num_demo, quality_cnt = 0, 0, 0, [0,0,0,0,0]
        demo_info = []
        selected_idx=[]
        for i, (s, a, r, t) in enumerate(sar):
            rew_sum += r
            if t == True or i + 1 - start >= 1000:
                
                # if rew_sum > 4000 and quality_cnt[4]<20:
                #     quality_cnt[4] += 1
                #     num_demo += 1
                #     demo_info.append((rew_sum, i + 1 - start))
                #     selected_idx+=[idx for idx in range(start,i+1)]
                # if rew_sum > 10000 and quality_cnt[1]<50:
                #     quality_cnt[1] += 1
                #     num_demo += 1
                #     demo_info.append((rew_sum, i + 1 - start))
                #     selected_idx+=[idx for idx in range(start,i+1)]
                if rew_sum > 3000 and quality_cnt[3]<50:
                    quality_cnt[3] += 1
                    num_demo += 1
                    demo_info.append((rew_sum, i + 1 - start))
                    selected_idx+=[idx for idx in range(start,i+1)]
                elif 3000 > rew_sum > 2000 and quality_cnt[2]<50:
                    quality_cnt[2] += 1
                    num_demo += 1
                    demo_info.append((rew_sum, i + 1 - start))
                    selected_idx+=[idx for idx in range(start,i+1)]
                elif 2000 > rew_sum > 1000 and quality_cnt[1]<50:
                    quality_cnt[1] += 1
                    num_demo += 1
                    demo_info.append((rew_sum, i + 1 - start))
                    selected_idx+=[idx for idx in range(start,i+1)]
                elif 1000 > rew_sum > 0 and quality_cnt[0]<50:
                    quality_cnt[0] += 1
                    num_demo += 1
                    demo_info.append((rew_sum, i + 1 - start))
                    selected_idx+=[idx for idx in range(start,i+1)]
                
                if num_demo>200:
                    break
                
                rew_sum, start = 0, i + 1
        
        dataset.demo_list += list(
            [states[i], actions[i], rewards[i], terminals[i]] for i in selected_idx)

        logger.info(f"Total number of demonstrations is {num_demo}" 
                    f" , the info of them is {demo_info}"
                    f" , total number of state-action pairs is {len(dataset.demo_list)}"
                    )

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
        filtered_demos=[]
        
        threshold=self.threshold
        
        if self.pretrain_num_samples > 0:   
            # 收集所有轨迹的argmax_t
            trajectory_argmax_ts = []
            temp_start = 0
            for i in range(len(data)):
                if data[i][3] == True or i + 1 - temp_start >= 1000:
                    argmax_t = torch.argmax(torch.mean(cond_Q[temp_start:i + 1], dim=0).cpu()).item()
                    trajectory_argmax_ts.append(argmax_t)
                    temp_start = i + 1
            
            # 计算所有轨迹argmax_t的平均值
            if len(trajectory_argmax_ts) > 0:
                avg_argmax_t = sum(trajectory_argmax_ts) / len(trajectory_argmax_ts)
                if avg_argmax_t < self.threshold:
                    threshold = avg_argmax_t

        for i in range(len(data)):
            rew_sum += data[i][2]
            if data[i][3] == True or i + 1 - start >= 1000:
                argmax_t = torch.argmax(torch.mean(cond_Q[start:i + 1], dim=0).cpu())
                
                if argmax_t > threshold :
                    demos += data[start:i + 1]
                    weights += [argmax_t.item() for _ in range(i+1-start)]
                    demo_rews.append((rew_sum,argmax_t))
                else:
                    filtered_demos.append((rew_sum,argmax_t))

                start = i + 1
                rew_sum = 0
        
        logger.info(f"demos_reserved:{demo_rews}")
        logger.info(f"demos_filtered:{filtered_demos}")
        logger.info(f"numbers of demos reserved:{len(demo_rews)}")

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
        
        
            
        self.ds.demo_list = demos

        sampler,shuffle = None,True
        if weights != []:
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_demos, replacement=True)
            shuffle=None

        self.dl = cycle(
            data.DataLoader(self.ds, batch_size=self.batch_size,sampler=sampler, shuffle=shuffle, pin_memory=True))
        
        if num_demos <= 30:
            self.stop_filtering = True

        # if num_demos >= 10:
        #     self.ds.demo_list = demos

        #     sampler,shuffle = None,True
        #     if weights != []:
        #         sampler = WeightedRandomSampler(weights=weights, num_samples=num_demos, replacement=True)
        #         shuffle=None

        #     self.dl = cycle(
        #         data.DataLoader(self.ds, batch_size=self.batch_size,sampler=sampler, shuffle=shuffle, pin_memory=True))
        # else:
            
        #     # self.stop_epsilon=True
        #     self.stop_filtering=True


    def train(self):
        
        
        logger.info("==================Pretraining...=========================")
        while self.step * self.batch_size < self.pretrain_num_samples:
            
            state, action, reward = next(self.dl)
            state , action = state.to(self.device) , action.to(self.device)
                
            d_losses=[]
            for i in range(self.denoiser_update_iter):
                denoiser_loss = self.smile.denoiser_loss(state, action)
                d_losses.append(denoiser_loss.item())
                (denoiser_loss / (self.denoiser_update_iter)).backward()
            
            self.denoise_opt.step()
            self.denoise_opt.zero_grad()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema(self.ema_smile.denoiser, self.smile.denoiser)
            
            self.step += 1
            # self.stop_epsilon=True
        
        logger.info('pretraining completed')
        
        self.step=0
        logger.info("==================Start Training...=========================")
        while self.step * self.batch_size < self.num_samples:

            state, action, reward = next(self.dl)
            state , action = state.to(self.device) , action.to(self.device)
            
            d_losses=[]
            if not self.stop_epsilon:
                
                for i in range(self.denoiser_update_iter):
                    denoiser_loss = self.smile.denoiser_loss(state, action)
                    d_losses.append(denoiser_loss.item())
                    (denoiser_loss / (self.denoiser_update_iter)).backward()
                
                self.denoise_opt.step()
                self.denoise_opt.zero_grad()
            if len(d_losses)==0:
                d_losses.append(0)
            
            p_losses=[]
            for i in range(self.policy_update_iter):
                policy_loss = self.smile.policy_loss(state, action)
                p_losses.append(policy_loss.item())
                (policy_loss / (self.policy_update_iter)).backward()
            
            self.policy_opt.step()
            self.policy_opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                if not self.stop_epsilon:
                    self.step_ema(self.ema_smile, self.smile)
                else:
                    self.step_ema(self.ema_smile.policy, self.smile.policy)


            if self.step % self.eval_every == 0:
                
                rewards,len_sum=[],0
                for i in range(10):
                    reward,len_traj = self.eval(self.ema_smile)
                    rewards.append(reward)
                    len_sum+=len_traj

                self.writer.add_scalar("average return",np.mean(rewards),self.step)
                self.writer.add_scalar("denoiser loss",np.mean(d_losses),self.step)
                self.writer.add_scalar("policy loss",np.mean(p_losses),self.step)
                
                logger.info(f"num_samples:{self.step * self.batch_size}, return:{np.mean(rewards)}, "
                            f"max_return:{np.max(rewards)}, min_return:{np.min(rewards)}, return_std:{np.std(rewards)}, avg_len:{len_sum/10}")

            if self.step != 0 and self.step % self.filter_every == 0 and self.stop_filtering == False:
            # if self.step != 0 and self.step % self.filter_every == 0:
                self.filter_dataset(self.ema_smile)

            self.step += 1
        
        self.writer.close()
        logger.info('training completed')