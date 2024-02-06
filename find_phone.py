import sys
import os
import cv2
import torch
import torchvision.transforms.functional as tf


image_path: os.PathLike = sys.argv[1]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

image = cv2.imread(image_path) / 255.0
image = tf.to_tensor(image.astype('float32')).to(device)

# Load model
model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'alexnet',
    weights=None
)
in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(in_features, 2)

model.load_state_dict(torch.load('best_model_wts.pth'))
model.eval()
model = model.to(device)
with torch.no_grad():
    image = model.features(image)
    image = model.avgpool(image)
    image = torch.flatten(image)
    pred = model.classifier(image).numpy()
pred = (pred[1], pred[0])

print(f'{pred[0]:>6f} {pred[1]:>6f}')
    