import torch
import torch.utils.data
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, max_iterations):
        super().__init__()

        # ColorBrewer Qualitative 8-class Set1 color palette
        self.colors = np.array([
            (228, 26, 28),
            (55, 126, 184),
            (77, 175, 74),
            (152, 78, 163),
            (254, 127, 1),
            (254, 254, 51),
            (166, 86, 40),
            (247, 129, 191)
        ])

        self.max_iterations = max_iterations

    def __getitem__(self, index):
        random_color_index = np.random.randint(0, self.colors.shape[0]-1, size=1).item()
        random_color = self.colors[random_color_index][:, np.newaxis, np.newaxis]

        image = np.repeat(random_color, 64, axis=1)
        image = np.repeat(image, 64, axis=2)
        image = (image.astype(np.float32) / 255.0) * 2 - 1

        return image

    def __len__(self):
        return self.max_iterations
