import torch
import torchvision
from PIL import Image
import os
class CelebADataset(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset
    """

    def __init__(self, image_size: int, data_dir):
        super().__init__()

        # CelebA images folder
        folder = data_dir + 'celebA/img_align_celeba/'
        # List of files
        imgs = os.listdir(folder)
        self._files = [os.path.join(folder, k) for k in imgs]
        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_size, image_size)),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
