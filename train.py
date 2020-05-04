import argparse

import os

import torch
import torch.utils.data
from PIL import Image
import numpy as np

from dataset import Dataset
from discriminator import Discriminator
from generator import Generator


def train(opts):
    # Define environment
    set_gpus(opts.gpu)
    device = torch.device("cuda")

    # Other params
    batch_size: int = 32
    latent_dimension: int = 1

    os.makedirs(opts.output_path, exist_ok=True)
    for sample_index in range(batch_size):
        sample_path = os.path.join(opts.output_path, f"{sample_index:03d}")
        os.makedirs(sample_path, exist_ok=True)

    # Define models
    generator = Generator(latent_dimension).to(device, non_blocking=True)
    discriminator = Discriminator().to(device, non_blocking=True)

    # Define train data loader
    max_iterations: int = 200000
    dataset = Dataset(max_iterations * batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Define optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.99))

    criterion = torch.nn.functional.binary_cross_entropy_with_logits

    # Define validation params
    z_validation = torch.randn(batch_size, latent_dimension, 1, 1, device=device)

    # Export some real images
    real_sample_images = next(iter(dataloader))
    real_sample_path = os.path.join(opts.output_path, f"real")
    os.makedirs(real_sample_path, exist_ok=True)

    for sample_id in range(batch_size):
        sample_iteration_path = os.path.join(real_sample_path, f"{sample_id + 1:05d}.jpg")
        image_sample = (real_sample_images[sample_id].permute(1, 2, 0).numpy() + 1) / 2
        image_sample = Image.fromarray((image_sample * 255).astype(np.uint8), mode="RGB")
        image_sample.save(sample_iteration_path)

    # Train loop
    for iteration, images in enumerate(dataloader):
        # Move data to gpu
        images = images.to(device, non_blocking=True)

        # Define targets
        fake_target = torch.zeros(batch_size, 1, 1, 1, device=device)
        real_target = torch.ones(batch_size, 1, 1, 1, device=device)

        # Train generator
        # sample z
        z = torch.randn(batch_size, latent_dimension, 1, 1, device=device)
        # get G(z): pass z through generator --> get prediction
        fake_sample = generator(z)
        # pass G(z) through discriminator
        fake_prediction = discriminator(fake_sample)
        # compute fake loss
        loss_generator = criterion(fake_prediction, real_target)

        # backprop through generator
        optimizer_g.zero_grad()
        loss_generator.backward()
        optimizer_g.step()

        # Train discriminator
        # pass real data through discriminator
        real_prediction = discriminator(images)
        # pass G(z).detach() through discriminator
        fake_prediction = discriminator(fake_sample.detach())

        # compute real loss
        loss_real = criterion(real_prediction, real_target)

        # compute fake loss
        loss_fake = criterion(fake_prediction, fake_target)
        loss_discriminator = (loss_real + loss_fake) / 2

        # backprop through discriminator
        optimizer_d.zero_grad()
        loss_discriminator.backward()
        optimizer_d.step()

        if iteration % opts.log_frequency == opts.log_frequency - 1:
            log_fragments = [
                f"{iteration + 1:>5}:",
                f"Loss(G): {loss_generator.item():>5.4f}",
                f"Loss(D): {loss_discriminator.item():>5.4f}",
                f"Real Pred.: {torch.sigmoid(real_prediction).mean().item():>5.4f}",
                f"Fake Pred.: {torch.sigmoid(fake_prediction).mean().item():>5.4f}",
            ]
            print(*log_fragments, sep="\t")

        # Validation
        if iteration % opts.validation_frequency == opts.validation_frequency - 1:
            with torch.no_grad():
                generator.eval()
                val_samples = generator(z_validation).to("cpu")
                generator.train()

            # output image
            for sample_id in range(batch_size):
                sample_iteration_path = os.path.join(opts.output_path, f"{sample_id:03d}", f"{iteration+1:05d}.jpg")
                image_sample = (val_samples[sample_id].permute(1, 2, 0).numpy() + 1) / 2
                image_sample = Image.fromarray((image_sample * 255).astype(np.uint8), mode="RGB")
                image_sample.save(sample_iteration_path)


def set_gpus(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--validation_frequency", type=int, default=100)

    args = parser.parse_args()
    train(args)
