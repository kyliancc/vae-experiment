import torch
from torchvision.utils import save_image

import argparse
from datetime import datetime
import os
import math

from vae import VAEDecoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-f', type=str, default=None, help='path to checkpoint', required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--number', '-n', type=int, default=1, help='how many images to generate')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args.__dict__}')

    n_loops = math.ceil(args.number / args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    decoder = VAEDecoder()
    decoder.to(device)
    state_dict = torch.load(args.load)
    decoder.load_state_dict(state_dict['decoder'])
    iterations = state_dict['iterations']

    decoder.eval()

    cnt = 0

    for i in range(n_loops):
        batch_size = args.batch_size
        if batch_size > args.number:
            batch_size = args.number

        z = torch.randn(batch_size, 128, device=device)
        y = decoder(z).cpu()

        for j in range(y.size(0)):
            save_path = os.path.join('./out/', f'{current_time}-{cnt}.png')
            save_image(y[j,:,:,:], save_path)
            print(f'Image saved to {save_path}')
            cnt += 1


if __name__ == '__main__':
    main()
