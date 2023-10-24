from torch import nn
from unet import *
from DenoiseDiffusion import *
from typing import List
from torchvision.utils import save_image
from tqdm import tqdm


class DDPM(nn.Module):

    def __init__(self, image_channels=3, image_size=32, n_channels=64, channel_multipliers = [1, 2, 3, 4],
                 is_attention=[False, False, False, True], n_steps=1000, batch_size=64, n_samples=16, learning_rate=2e-5, n_epochs=1000, device=torch.device):
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.n_channels = n_channels
        self.channel_multipliers = channel_multipliers
        self.is_attention = is_attention
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device

        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        )
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )


    def sample(self, epoch, save_dir, epochs):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            pbsr = tqdm(range(epochs), desc=f'Sample {epoch}')
            for t_ in pbsr:
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            save_image(x, save_dir + str(epoch) + '.png', normalize=True, scale_each=True, nrow=4)


    def train(self, data_loader, pbar, optimizer):
        """
        ### Train
        """

        # Iterate through the dataset
        for batch_idx, data in enumerate(pbar):
            # Increment global step
            # Move data to device
            data = data.to(self.device)
            # Make the gradients zero
            optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            optimizer.step()
            # Track the loss
            if batch_idx == 0:
                print(loss)