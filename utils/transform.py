
import torchvision.transforms as transforms

normalization = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465), 
    std=(0.2023, 0.1994, 0.2010)
)

training_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalization
])

validation_transform = transforms.Compose([
    transforms.ToTensor(),
    normalization
])
