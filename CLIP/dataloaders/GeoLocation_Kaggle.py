import os
import pickle
import random
import json
from glob import glob
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
import warnings 
from torchvision.datasets.vision import VisionDataset



class GeoLocation_Kaggle(VisionDataset):
    def __init__(
        self,
        root = "/data/azfarm/siddhant/Geolocalization_UCF/",
        train = False,
        transform = None, 
        class_info = "/data/azfarm/siddhant/Geolocalization_UCF/GeoLocation_kaggle.txt", 
        ):

        self.root = root 
        self.train = train 
        self.transform = transform
        self.class_info = class_info 
        self.img_dir = os.path.join(self.root, "GeoLocation_Kaggle")
        self.split = os.path.join(self.root, "GeoLocation_kaggle.json")

        self.read_data()

        if self.train:
            raise ValueError(
                "This dataset has no train split"
            )
        
        self.images = self.test_imgs 
        self.labels = self.test_labels 

        warnings.warn('size mismatch is there')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        target = self.labels[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target 
    
    def read_data(self):
        with open(self.split, "r") as f:
            obj = json.load(f)
        
        self.test_imgs = [os.path.join(self.img_dir, i[0]) for i in obj]
        self.test_labels = [i[1] for i in obj]

        assert len(self.test_imgs) == 49997 


