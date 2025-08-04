import os
from .GeoLocation_Kaggle import GeoLocation_Kaggle
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def get_dataloader(dataset_name, transform=None, loader_type = "test", transform_train=None, k_shot=None, **kwargs):
    print(f" Dataset: {dataset_name.upper()}.")
    print(f" Transformation test: {transform}")
    print(f" Transformation train: {transform_train}")
    print(f" Dataloader type: {loader_type}.")
    
    
    if dataset_name == "geolocation_kaggle":
        test = GeoLocation_Kaggle(root="../datasets/", 
            train=False, 
            transform=transform,
        )
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test

        else:
            raise ValueError(f" Wrong {loader_type} type. Geolocation_Kaggle doesn't support train")
    else:
        raise AttributeError(f"{dataset_name} is not currently supported.")