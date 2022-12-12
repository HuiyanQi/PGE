import data.transform as trans


def choose_transform(func):
    def wrapper(train):
        args = func(train)
        transform =getattr(trans,func.__name__)
        transform = transform(*args)
        return transform
    return wrapper

@choose_transform
def cifar10_transform(train):
    size = 32
    padding = 4
    fill = 128
    length = 16
    n_holes = 1
    return size, padding, fill, length, n_holes, train

@choose_transform
def cifar100_transform(train):
    size = 32
    padding = 4
    fill = 128
    length = 16
    n_holes = 1
    return size, padding, fill, length, n_holes, train

@choose_transform
def stl10_transform(train):
    size = 96
    padding = 4
    fill = 128
    length = 16
    n_holes = 1
    return size, padding, fill, length, n_holes, train

@choose_transform
def mnist_transform(train):
    size = 28
    return size, train

@choose_transform
def fashionmnist_transform(train):
    size = 28
    return size, train

@choose_transform
def cub_transform(train):
    size = 512
    return size, train

@choose_transform
def mini_imagenet_transform(train):
    size = 84
    return size, train

@choose_transform
def contrastive_transform(train):
    size = 32
    return size, train

@choose_transform
def default_transform(train):
    size = 256
    return size, train
    # size = 96
    # padding = 4
    # fill = 128
    # length = 16
    # n_holes = 1
    # return size, padding, fill, length, n_holes, train