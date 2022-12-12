from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment,AutoAugmentPolicy
from data.data_aug.utils import Cutout
from data.data_aug.gaussian_blur import GaussianBlur

def cifar10_transform(size,padding,fill,length,n_holes,train):
    if train:
        data_transforms=transforms.Compose([
                                            transforms.RandomCrop(size, padding=padding, fill=fill),
                                            transforms.RandomHorizontalFlip(),
                                            AutoAugment(AutoAugmentPolicy.CIFAR10),
                                            transforms.ToTensor(),
                                            Cutout(n_holes=n_holes,length=length),
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2032, 0.1994, 0.2010])
                                            ])
    else:
        data_transforms=transforms.Compose([
                                            transforms.RandomCrop(size, padding=padding, fill=fill),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2032, 0.1994, 0.2010])
                                            ])
    return data_transforms

def cifar100_transform(size,padding,fill,length,n_holes,train):
    if train:
        data_transforms=transforms.Compose([
                                                        transforms.RandomCrop(size, padding=padding, fill=fill),
                                                        transforms.RandomHorizontalFlip(),
                                                        AutoAugment(AutoAugmentPolicy.CIFAR10),
                                                        transforms.ToTensor(),
                                                        Cutout(n_holes=n_holes,length=length),
                                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2032, 0.1994, 0.2010])
                                                        ])
    else:
        data_transforms=transforms.Compose([
                                                    transforms.RandomCrop(size, padding=padding, fill=fill),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2032, 0.1994, 0.2010])
                                                    ])
    return data_transforms


def stl10_transform(size,padding,fill,length,n_holes,train):
    if train:
        data_transforms=transforms.Compose([
                                                        transforms.RandomCrop(size, padding=padding, fill=fill),
                                                        transforms.RandomHorizontalFlip(),
                                                        AutoAugment(AutoAugmentPolicy.CIFAR10),
                                                        transforms.ToTensor(),
                                                        Cutout(n_holes=n_holes,length=length),
                                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2032, 0.1994, 0.2010])
                                                        ])
    else:
        data_transforms=transforms.Compose([
                                                    transforms.RandomCrop(size, padding=padding, fill=fill),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2032, 0.1994, 0.2010])
                                                    ])
    return data_transforms


def mnist_transform(size,train):
    data_transforms = transforms.Compose([
                                                    transforms.Resize(size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                    ])
    return data_transforms

def fashionmnist_transform(size,train):
    data_transforms = transforms.Compose([
                                                    transforms.Resize(size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                    ])
    return data_transforms


def mini_imagenet_transform(size,train):
    if train:                    
        data_transforms = transforms.Compose([
                                                        transforms.Resize((size,size)),
                                                        transforms.RandomHorizontalFlip(),
                                                        AutoAugment(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
    else:
        data_transforms = transforms.Compose([
                                                        transforms.Resize((size,size)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])            
    return data_transforms

def cub_transform(size,train):
    if train:                    
        data_transforms = transforms.Compose([
                                                        transforms.Resize((size,size)),
                                                        transforms.RandomCrop(int(size*0.875)),
                                                        transforms.RandomHorizontalFlip(),
                                                        AutoAugment(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
    else:
        data_transforms = transforms.Compose([
                                                        transforms.Resize((size,size)),
                                                        transforms.RandomCrop(int(size*0.875)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])            
    return data_transforms




def con_transform(train):
    if train:
        color_jitter = transforms.ColorJitter(0.4 * 1, 0.4 * 1, 0.4 * 1, 0.1 * 1)                    
        data_transforms = transforms.Compose([
                                                transforms.Resize((256,256)),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * 224)),
                                                transforms.RandomVerticalFlip(0.5),
                                                AutoAugment(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
                                                transforms.Resize((256,256)),
                                                transforms.RandomCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])            
    return data_transforms


def default_transform(size,train):
    #  size,padding,fill,length,n_holes,train
    if train:                    
        data_transforms = transforms.Compose([
                                                        transforms.Resize((size,size)),
                                                        transforms.RandomCrop(int(size*0.875)),
                                                        transforms.RandomHorizontalFlip(),
                                                        AutoAugment(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                        # RandomErasing(mean=[0.485, 0.456, 0.406])
                                                        ])
    else:
        data_transforms = transforms.Compose([
                                                        transforms.Resize((size,size)),
                                                        transforms.RandomCrop(int(size*0.875)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])            
    return data_transforms
