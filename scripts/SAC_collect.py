import gym
import argparse
import mujoco_py

import numpy as np
import h5py

from stable_baselines3 import SAC

def get_args():
    parser = argparse.ArgumentParser(description='SAC_collect')

    parser.add_argument(
        '--env_name',
        default="HalfCheetah-v2",
        choices=["HalfCheetah-v2", "Ant-v2", "Hopper-v2", "Walker2d-v2", "Humanoid-v2", "Reacher-v2"],
        help='environments to collect diverse-quality demos.')

    parser.add_argument(
        '--demo_folder',
        default="../demos",
        help='folder to save collected demonstrations.')

    parser.add_argument(
        '--exp_folder',
        default="../policies",
        help='folder containing expert models.')

    parser.add_argument(
        '--num_demos_each',
        default = 10,
        help='numbers of collected demos for each demonstrators.')

    parser.add_argument(
        '--max_length',
        default=1000,
        help='numbers of maximum length for each interactions.')

    args = parser.parse_args()

    args.demo_path = args.demo_folder+'/'+args.env_name+'.h5'
    args.exp_path = args.exp_folder+ '/' + args.env_name + '_expert.pth'

    return args

def collect_diverse_quality_demos(args):
    noisy_level=[0.00,0.05,0.10,0.20,0.30,0.40,0.60,0.70,0.80,0.90,1.00]

    expert_path = args.exp_path
    demo_path = args.demo_path

    num_demos_each = args.num_demos_each
    max_length = args.max_length

    env = gym.make(args.env_name)

    print("=======================================================================")
    print("Start training and collecting.")
    model = SAC.load(expert_path)
    state = []
    action = []
    state_next = []
    reward = []
    done = []
    flag = []
    for nl in noisy_level:
        reward_avg = 0
        for k1 in range(1, num_demos_each + 1):
            s = env.reset()
            reward_each = 0
            for k2 in range(1, max_length + 1):
                state.append(s)
                a, _ = model.predict(s, deterministic=True)
                a = a + nl * np.random.randn(a.size)
                action.append(a)
                s, r, d, info = env.step(a)
                state_next.append(s)
                reward.append(r)
                done.append(d)
                reward_avg += r
                reward_each += r
                if d:
                    flag.append(1)
                    break
                else:
                    if k2 == max_length:
                        flag.append(1)
                    else:
                        flag.append(0)
            print("reward: " + str(round(reward_each, 2)))
        reward_avg /= num_demos_each
        print("Expert:" + expert_path + ",noisy_level:" + str(nl) + ",average reward is: " + str(round(reward_avg, 2)))

    state = np.vstack(state)
    action = np.vstack(action)
    state_next = np.vstack(state_next)
    reward = np.vstack(reward)
    done = np.vstack(done)
    flag = np.vstack(flag)

    file = demo_path
    with h5py.File(file, 'w') as f:
        f.create_dataset("states", data=state)
        f.create_dataset("actions", data=action)
        f.create_dataset("next_states", data=state_next)
        f.create_dataset("rewards", data=reward)
        f.create_dataset("dones", data=done)
        f.create_dataset("flags", data=flag)
    print("end up collecting trajectory at " + file + ".")


if __name__ == "__main__":
    args=get_args()
    collect_diverse_quality_demos(args)