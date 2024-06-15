import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST

import argparse

from vae import VAE, VAELoss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--lr', '-l', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='how many dataloader workers')
    parser.add_argument('--load', '-f', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--save', '-s', type=int, default=500, help='how many iterations to save')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args.__dict__}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = VAE()
    model.to(device)
    criterion = VAELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    iterations = 0

    if args.load:
        state_dict = torch.load(args.load)
        iterations = state_dict['iterations']
        model.encoder.load_state_dict(state_dict['encoder'])
        model.decoder.load_state_dict(state_dict['decoder'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded checkpoint from {args.load}.')

    print('START TRAINING')

    for epoch in range(args.epochs):
        for x, _ in train_loader:
            x = x.to(device)

            model.train()
            optimizer.zero_grad()

            y, mean, logvar = model(x)
            loss = criterion(x, y, mean, logvar)

            loss.backward()

            optimizer.step()

            print(f'Iteration {iterations} finished with loss: {loss.item()}')
            iterations += 1

            if iterations % args.save == 0:
                state_dict = {
                    'iterations': iterations,
                    'encoder': model.encoder.state_dict(),
                    'decoder': model.decoder.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state_dict, f'./checkpoints/vae-experiment-{iterations}.pth')
                print(f'Checkpoint saved to ./checkpoints/vae-experiment-{iterations}.pth')


if __name__ == '__main__':
    main()
