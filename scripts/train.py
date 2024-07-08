import torch
import numpy as np
import random
import sys

sys.path.append('/workspace/SMILE')
from model import Denoiser,Policy,SMILE,SMILETrainer
from util.arguments import get_args
import gym
# import gymnasium as gym
import d4rl

def main():

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu_id) if use_gpu else "cpu")

    denoiser = Denoiser(
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)

    policy = Policy(
        obs_dim=state_dim,
        action_dim=action_dim,
        device=device,
    ).to(device)

    smile = SMILE(
        denoiser,
        policy,
        args,
        betas=torch.arange(start=args.betas_low, end=args.betas_high, step=((args.betas_high - args.betas_low) / args.diffusion_steps)),
    ).to(device)

    trainer = SMILETrainer(
        smile,
        env,
        device,
        args
    )

    trainer.train()


if __name__ == "__main__":
    main()
