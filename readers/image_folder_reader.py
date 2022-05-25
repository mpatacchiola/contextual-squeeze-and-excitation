import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import numpy as np


class ImageFolderReader:
    def __init__(self, path_to_images, context_batch_size, target_batch_size, image_size, device,
                 train_fraction=0.7, val_fraction=0.1, test=0.2):
        self.device = device
        self.path_to_images = path_to_images
        self.context_batch_size = context_batch_size

        transforms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        ])

        data = ImageFolder(root=path_to_images, transform=transforms)
        dataset_length = len(data)
        train_size = int(round(train_fraction * dataset_length))
        val_size = int(round(val_fraction * dataset_length))
        self.test_size = dataset_length - train_size - val_size
        train_set, val_set, test_set = torch.utils.data.random_split(data, [train_size, val_size, self.test_size],
                                                                     generator=torch.Generator().manual_seed(15))
        self.context_iterator = iter(torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=1,
            shuffle=True,
            num_workers=4))
        self.target_iterator = iter(torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=target_batch_size,
            shuffle=False,
            num_workers=4
        ))

    def get_target_dataset_length(self):
        return self.test_size

    def get_context_batch(self):
        return self._get_sun397_context_batch(self.context_iterator)

    def get_target_batch(self):
        return self._get_batch(self.target_iterator, is_target=True)

    def _get_batch(self, iterator, is_target):
        images, labels = iterator.next()

        # move the images and labels to the device
        images = images.to(self.device)
        if is_target:
            labels = labels.type(torch.LongTensor).to(self.device)
        else:
            labels = labels.to(self.device)

        return images, labels

    def _get_sun397_context_batch(self, iterator):
        # This code is slow and hacky, but assures we get a context set
        # of the correct size with at least one example per class.
        images = []
        labels = []
        label_counts = np.zeros(397, dtype=np.int)
        count = 0
        while True:
            image, label = iterator.next()
            index = label.cpu().numpy()
            if label_counts[index] < 2:
                images.append(image)
                labels.append(label)
                label_counts[index] += 1
            all_labels = torch.hstack(labels)
            count += 1
            if len(torch.unique(all_labels)) == 397 or count == 10000:
                break

        current_count = len(labels)
        to_get = 1000 - current_count

        for _ in range(to_get):
            image, label = iterator.next()
            images.append(image)
            labels.append(label)

        images = torch.vstack(images)
        labels = torch.hstack(labels)

        images = images.to(self.device)
        labels = labels.to(self.device)

        return images, labels
