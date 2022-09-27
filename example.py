import torch
import torchvision

from backbones import efficientnet
from models.uppercase import UpperCaSE
from adapters.case import CaSE      

# Example script that runs an UpperCaSE model on CIFAR100 and SVHN.
# We randomly sample 4096 images and use only 25% as context set.
# The model will provide a prediction on the remaining images in the batch (target set).
#
# Installation: you just need Pytorch and basic libraries (e.g. Numpy).
# Usage: just run `python example.py`, modify the parameters below for different configurations.

dataset_name = "svhn" # supports ["cifar100", "svhn"]
batch_size = 4096 # With batch 4096 we take ~1000 images, same as in VTAB benchmark
device = "cpu" # Use "cuda" if your GPU has enough memory
adapter = CaSE # The adapter can be a standard SE or a CaSE layer

# We are loading our modified version of EfficientNetB0, note that it takes as input an adaptive layer
backbone = efficientnet.efficientnet_b0(pretrained=True, progress=True, norm_layer=torch.nn.BatchNorm2d, adaptive_layer=adapter)
checkpoint = torch.load("./checkpoints/UpperCaSE_CaSE64_min16_EfficientNetB0.dat")
backbone.load_state_dict(checkpoint["backbone"], strict=True)
backbone = backbone.to(device)

# Normalization values and associate transform for EfficientNetB0
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = torchvision.transforms.Resize(size=[224,])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), resize, normalize])  

# We get only one single mini-batch from the iterator (we are in the few-shot setting)
if(dataset_name=="cifar100"): train_set = torchvision.datasets.CIFAR100(root="./datasets", train=True, download=True, transform=transform)
elif(dataset_name=="svhn"): train_set = torchvision.datasets.SVHN(root="./datasets", split="test", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
images, labels = next(iter(train_loader))

# Take 25% as context set and the remaining as target
context_images = images[0:batch_size//4].to(device)
context_labels = labels[0:batch_size//4].to(device)
target_images = images[batch_size//4:].to(device)
target_labels = labels[batch_size//4:].to(device)

# Defining an UpperCaSE model with linear head fine-tuned for 500 iterations with linearly decayed learning rate.
model = UpperCaSE(backbone, adapter, device, tot_iterations=500, start_lr=1e-3, stop_lr=1e-5)

# The predict() method performs adaptation and inference.
# The model will return the log probabilities for the target images.
log_probs = model.predict(context_images, context_labels, target_images)

# Estimating the accuracy and printing
target_predictions = torch.argmax(log_probs, dim=1)
accuracy = (torch.eq(target_labels, target_predictions).sum() / len(target_labels)) * 100.0
print(f"Dataset name ..... {dataset_name}")
print(f"Context size ..... {len(context_labels)}")
print(f"Test accuracy .... {accuracy.item():.1f}%")
