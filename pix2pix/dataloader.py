import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ABDataset(Dataset):
    def __init__(self, root_path, phase, data_augmentation=True, crop_size=450, crop_prob=0.5, flip_prob=0.5, reverse=False):
        super(ABDataset, self).__init__()
        A_path = os.path.join(root_path, "A" if not reverse else "B", phase)
        B_path = os.path.join(root_path, "B" if not reverse else "A", phase)
        self.A_imgs = os.listdir(A_path)
        self.B_imgs = os.listdir(B_path)
        self.A_imgs = [os.path.join(A_path, A_img) for A_img in sorted(self.A_imgs)]
        self.B_imgs = [os.path.join(B_path, B_img) for B_img in sorted(self.B_imgs)]
        print(self.A_imgs[0])
        assert len(self.A_imgs) == len(self.B_imgs)
        self.flip_prob = flip_prob
        self.data_augmentation = data_augmentation
        self.crop_size = crop_size
        self.crop_prob = crop_prob

        self.transforms = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.A_imgs)

    def __getitem__(self, index):
        A_img_path = self.A_imgs[index]
        A = Image.open(A_img_path).convert("RGB")
        B_img_path = self.B_imgs[index]
        B = Image.open(B_img_path).convert("RGB")

        if self.data_augmentation:
            #Random Crop
            if random.random() < self.crop_prob:
                w, h = A.size
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
                A = A.crop((x, y, x + self.crop_size, y + self.crop_size))
                B = B.crop((x, y, x + self.crop_size, y + self.crop_size))
            #Random Flip
            if random.random() < self.flip_prob:
                A = A.transpose(Image.FLIP_LEFT_RIGHT)
                B = B.transpose(Image.FLIP_LEFT_RIGHT)

        A = self.transforms(A)
        B = self.transforms(B)

        return {'A': A, 'B': B, 'A_path': A_img_path, 'B_path': B_img_path}

class ParallelDataloader(object):
    def __init__(self, root_path, phase, data_augmentation=True, crop_size=450, crop_prob=0.5, flip_prob=0.5, reverse=False, batch_size=32, shuffle=True, num_workers=1):
        self.dataset = ABDataset(root_path, phase, data_augmentation, crop_size, crop_prob, flip_prob, reverse)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

