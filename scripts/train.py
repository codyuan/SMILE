import torch
import numpy as np
import random
from model import Denoiser,Policy,SMILE,SMILETrainer
from util.arguments import get_args
import gym

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
        betas=torch.arange(start=0.05, end=0.6, step=((0.6 - 0.05) / args.diffusion_steps)),
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






