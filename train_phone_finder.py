"""
Trains a pretrained model by freezing all its weights, replace the last
layer with a linear layer, and training this last layer. After that, the
whole model is fine-tuned over the train set.

The model used for this purpose is AlexNet, which uses about 2099.17 MB of
memory and is trained using early stopping and LROnPlateau learning rate
scheduler.
"""
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2
from modules.data import PhoneDataset
from modules.training import EarlyStopping, ModelTrainer
from modules.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomTranslation
)
from modules.utilities import get_data, train_test_split


seed = 0
folder: os.PathLike = sys.argv[1]
np.random.seed(seed)
torch.manual_seed(seed)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Images shape
c_in = 3  # Number of channels
h_in, w_in = 326, 490  # Height and width

test_size = 0.1
batch_size = 128
patience = 30
min_delta = 0.0
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam
epochs = 1000
lr = 1e-1
scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau

image_paths, locations = get_data(folder)
train_image_paths, test_image_paths, train_locations, test_locations = train_test_split(
    image_paths,
    locations,
    test_size,
    seed
)
aware_transforms = v2.Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomTranslation(),
])
train_dataset = PhoneDataset(train_image_paths, train_locations, aware_transforms)
test_dataset = PhoneDataset(test_image_paths, test_locations)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
early_stopping = EarlyStopping(patience, min_delta)
trainer = ModelTrainer(
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    early_stopping,
    scheduler_class,
    device
)

# Build model
model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'alexnet',
    weights='AlexNet_Weights.DEFAULT'
)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(in_features, 2)

# Train model
print('\n\nBeggining training of AlexNet model.')
trainer(epochs, model, lr)

# Fine-tune model
print('\n\nFine-tuning AlexNet.')
for param in model.parameters():
    param.grad_requires = True

trainer(epochs, model, lr)
