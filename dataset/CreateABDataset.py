import os
import json
import random
from PIL import Image
import torch
import torchvision.transforms.functional as tfunction

# my custom dataset
class CreateABDataset:

    def __init__(self, data_root, load_size, crop_size, vh_flip, crop_random):
    
        # get the dataset options
        self.load_size = load_size
        self.crop_size = crop_size
        self.vh_flip = vh_flip
        self.crop_random = crop_random
        
        # assert on error
        config_file = os.path.join(data_root, "config.json")
        assert os.path.isfile(config_file), f"Config file not found at {config_file}"
    
        # try to read configuration file and set variables
        with open(config_file, 'r') as file:
            config = json.load(file)

            # get image paths
            self.APaths = self.GetFilePaths(os.path.join(data_root, config["ASubDirectory"]))
            self.BPaths = self.GetFilePaths(os.path.join(data_root, config["BSubDirectory"]))

            # assert if the size doesnt match
            assert len(self.APaths) == len(self.BPaths), "A and B images must be the same length"
            
            # add the necessary configuration options
            self.name = config["DatasetName"]
            self.load_on_memory = config["LoadOnMemory"]
            self.image_count = len(self.APaths)
            
            # load images on startup if dataset supports it
            if self.load_on_memory:
                self.AImages = [Image.open(path).convert('RGB') for path in self.APaths]
                self.BImages = [Image.open(path).convert('L') for path in self.BPaths]
            
            print(f"{self.name} dataset loaded successfully with {self.image_count} unique images")
        
    def __getitem__(self, index: int):
    
        # always make sure index is within the range of the images
        indexI = index % self.image_count

        # open images
        if self.load_on_memory:
            A = self.AImages[indexI]
            B = self.BImages[indexI]
        else:
            A = Image.open(self.APaths[indexI]).convert('RGB')
            B = Image.open(self.BPaths[indexI]).convert('L')

        # get the transformed images and noise
        tA, tB = self.GetTransformedAB(A, B)

        # return RGB and IR pairs
        return tA, tB
    
    def __len__(self):
        return self.image_count

    # return the images inside the given directory
    @staticmethod
    def GetFilePaths(folder):
        image_file_paths = []
        for root, dirs, filenames in os.walk(folder):
            filenames = sorted(filenames)
            for filename in filenames:
                input_path = os.path.abspath(root)
                file_path = os.path.join(input_path, filename)
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    image_file_paths.append(file_path)
        
            break  # prevent descending into subfolders
        return image_file_paths

    def GetTransformedAB(self, A, B):
    
        # convert everything into tensor
        A = tfunction.to_tensor(A)
        B = tfunction.to_tensor(B)

        # resize the both image
        A = tfunction.resize(A, [self.load_size, self.load_size], interpolation=tfunction.InterpolationMode.BILINEAR)
        B = tfunction.resize(B, [self.load_size, self.load_size], interpolation=tfunction.InterpolationMode.BILINEAR)
    
        # if crop_size given, random crop the both image
        if self.crop_size:
            if self.crop_random:
                # crop position for A and B
                x = random.randint(0, max(0, tfunction.get_image_size(A)[0] - self.crop_size))
                y = random.randint(0, max(0, tfunction.get_image_size(A)[1] - self.crop_size))
            else:
                # crop from the center
                x = max(0, (tfunction.get_image_size(A)[0] - self.crop_size) // 2)
                y = max(0, (tfunction.get_image_size(A)[1] - self.crop_size) // 2)
            
            # crop A, B and N
            A = tfunction.crop(A, y, x, self.crop_size, self.crop_size)
            B = tfunction.crop(B, y, x, self.crop_size, self.crop_size)

        # apply random horizontal and vertical flip
        if self.vh_flip and random.random() > 0.5:
            A = tfunction.hflip(A)
            B = tfunction.hflip(B)

        if self.vh_flip and random.random() > 0.5:
            A = tfunction.vflip(A)
            B = tfunction.vflip(B)
        
        # clamp to 0.0,1.0 range
        A = torch.clamp(A, min=0.0, max=1.0)
        B = torch.clamp(B, min=0.0, max=1.0)
        
        return A, B