from dataset import *
import os
from config import *
from diffusion import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

args = get_config()

save_dir = './result/{}/seed_{}/'.format(
    args.dataset, args.seed)
make_folder(save_dir)

if args.dataset == 'celebA':
    dataset = CelebADataset(args.image_size, args.data_dir)

data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True)

model = DDPM(args.image_channels, args.image_size, args.n_channels, args.channel_multipliers,
                 args.is_attention, args.n_steps, args.batch_size, args.n_samples, args.learning_rate, args.n_epochs, device)

optimizer = torch.optim.Adam(model.eps_model.parameters(), lr=args.learning_rate)

model = nn.DataParallel(model.to(device))


print('Start training...')
for epoch in range(args.n_epochs):
    pbar = tqdm(data_loader, total=len(data_loader), desc=f'Epoch {epoch}')
    # Train the model
    model.module.train(data_loader, pbar, optimizer)
    # Sample some images
    model.module.sample(epoch, save_dir, args.n_epochs)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_dir + "model_" + str(epoch) + ".pth")





