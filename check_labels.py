from torchvision import datasets

dataset = datasets.ImageFolder("data")
print(dataset.class_to_idx)