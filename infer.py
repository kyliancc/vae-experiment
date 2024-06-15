import torch
from torchvision.utils import save_image

import argparse
from datetime import datetime
import os

from vae import VAEDecoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-f', type=str, default=None, help='path to checkpoint', required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args.__dict__}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    decoder = VAEDecoder()
    decoder.to(device)
    state_dict = torch.load(args.load)['decoder']
    decoder.load_state_dict(state_dict)

    decoder.eval()

    z = torch.randn(1, 128, device=device)
    y = decoder(z).cpu()

    save_path = os.path.join('./out/', f'{current_time}.png')
    save_image(y[0,:,:,:], save_path)
    print(f'Image saved to {save_path}')


if __name__ == '__main__':
    main()
