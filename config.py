import argparse
from typing import List


def get_config():

    parser = argparse.ArgumentParser(description='DDPM')

    # Data settings
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='celebA', choices=['celebA', 'tree'])
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')

    # Training settings
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--channel_multipliers', type=List[int], default=[1, 2, 3, 4])
    parser.add_argument('--is_attention', type=List[int], default=[False, False, False, True])
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--n_steps', type=float, default=1000)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    return args
