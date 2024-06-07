import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SMILE')

    parser.add_argument(
        '--env_name',
        default="HalfCheetah-v2",
        choices=["HalfCheetah-v2", "Ant-v2", "Hopper-v2", "Walker2d-v2", "Humanoid-v2", "Reacher-v2"],
        help='environment to train on.')

    parser.add_argument(
        '--demo_folder',
        default="../demos",
        help='folder containing demonstrations.')

    parser.add_argument(
        '--denoiser_loss_type',
        default='l1',choices=['l1', 'l2', 'cosine','kl'],
        help='objective function to train denoiser network.' )

    parser.add_argument(
        '--policy_loss_type',
        default='l1', choices=['l1', 'l2', 'cosine', 'mle'],
        help='objective function to train policy network.')

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='id of used gpu.')

    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='random seed.')

    parser.add_argument(
        '--diffusion_steps',
        default=10,
        type=int,
        help='number of diffusion steps.')

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='batch size for every iteration.')

    parser.add_argument(
        '--denoiser_lr',
        default=1e-3,
        type=float,
        help='learning rate of denoiser.')

    parser.add_argument(
        '--policy_lr',
        default=1e-3,
        type=float,
        help='learning rate of policy.')

    parser.add_argument(
        '--num_samples',
        default=2e7,
        type=int,
        help='total samples for training.')

    parser.add_argument(
        '--eval_every',
        default=2500,
        type=int,
        help='evaluate policy every x iterations.')

    parser.add_argument(
        '--filter_every',
        default=2500,
        type=int,
        help='filter dataset every x iterations.')

    parser.add_argument(
        '--ema_decay',
        default=0.995,
        type=float,
        help='the decay rate of ema update.')

    parser.add_argument(
        '--denoiser_update_iter',
        default=10,
        type=int,
        help='update denoiser x times every iteration.')

    parser.add_argument(
        '--policy_update_iter',
        default=1,
        type=int,
        help='update policy x times every iteration.')

    parser.add_argument(
        '--no_filtering',
        default = False,
        type=bool,
        help='whether training without filtering dataset.')

    parser.add_argument(
        '--naive_reverse',
        default = False,
        type=bool,
        help='whether applies multi-step generation.')

    parser.add_argument(
        '--zero_sigma',
        default=False,
        type=bool,
        help='whether to consider denoising to a sample itself in filtering.')

    parser.add_argument(
        '--step_start_ema',
        default=1000,
        type=int,
        help='when to start ema update.')

    parser.add_argument(
        '--update_ema_every',
        default=10,
        type=int,
        help='update ema every x iterations.')

    args = parser.parse_args()

    args.demo_path = args.demo_folder+'/'+args.env_name+'.h5'

    return args
