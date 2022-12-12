import data.dataset as dataset
from data.data_aug.view_generator import ContrastiveLearningViewGenerator
from torchvision.transforms import transforms
from data.data_aug.gaussian_blur import GaussianBlur


def get_dataset(func):
    def wrapper(root_folder,name,transform,n_views,train,contrastive):
        args = func(root_folder,name,transform,n_views,train,contrastive)
        create_dataset_func = getattr(dataset,name)
        return create_dataset_func(*args, transform=ContrastiveLearningViewGenerator(transform,n_views) if contrastive else transform, download=True)
    return wrapper

@create_dataset
def dataset(root_folder,name,transform,n_views,train,contrastive):
    return root_folder, train

