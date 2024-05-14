import os
import random
import torch
from utility import FPNInsert
from .CreateABDataset import CreateABDataset
from .CreateFPNDataset import CreateFPNDataset

class NoisyDataLoader:
    def __init__(self, data_root: str, is_train_mode: bool):
    
        # dataset options
        self.load_size = 320                # scale images to this size if size > 0
        self.crop_size = 288                # crop to this size if size > 0
        self.noise_count = 10000             # total noise count
        self.random_seed = 12237505         # random seed for the tests
        self.train_test_ratio_img = [0.7, 0.3]  # train and test ratio for the image dataset
        self.train_test_ratio_fpn = [0.9, 0.1]  # train and test ratio for the fpn dataset
        self.aggregation_size = 6           # number of noisy image for each FPN pattern
        self.current = 0                    # set the current to zero
        
        # set the batch size and vertical/horizontal flip
        if is_train_mode:
            self.batch_size = 18 if os.getlogin() == 'ubuntu' else 2
            self.vh_flip = True
        else:
            self.batch_size = 1
            self.vh_flip = False

        # set the random seed for consistency
        random.seed(self.random_seed)
        
        # dataset should have __init__, __get_item__ and __len__ methods
        self.image_dataset = CreateABDataset(data_root, load_size=self.load_size, crop_size=self.crop_size, vh_flip=self.vh_flip, crop_random=is_train_mode)
        self.fpn_dataset   = CreateFPNDataset(crop_size=self.crop_size, noise_count=self.noise_count)
        
        # create shuffled image indices
        image_indices = list(range(len(self.image_dataset)))
        random.shuffle(image_indices)

        # create shuffled noise indices
        noise_indices = list(range(len(self.fpn_dataset)))
        random.shuffle(noise_indices)

        # get indices
        if is_train_mode:
            image_index = round(len(self.image_dataset) * self.train_test_ratio_img[0])
            noise_index = round(len(self.fpn_dataset) * self.train_test_ratio_fpn[0])
            self.image_indices = image_indices[:image_index]
            self.noise_indices = noise_indices[:noise_index]
        else:
            image_index = round(len(self.image_dataset) * self.train_test_ratio_img[1])
            noise_index = round(len(self.fpn_dataset) * self.train_test_ratio_fpn[1])
            self.image_indices = image_indices[-image_index:]
            self.noise_indices = noise_indices[-noise_index:]
        
        # print info
        print("The number of images in the {} set is {}".format("TRAIN" if is_train_mode else "TEST", len(self.image_indices)))
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= len(self.noise_indices):
            raise StopIteration
        
        # randomly sample #aggregation_size images and uniformly select #batch_size noise
        batch_noise_indices = self.noise_indices[self.current:self.current + self.batch_size]
        
        # create output list
        RGBList, IRCList, IRNList, FPNList = [], [], [], []
        
        # create outputs
        for fpn_idx in batch_noise_indices:
            FPN = self.fpn_dataset[fpn_idx]             # FPN:  2x256x256
            
            # create sub list
            sRGBList, sIRCList, sIRNList, sFPNList = [], [], [], []

            # get the image indices for the current noise pattern
            batch_image_indices = random.sample(self.image_indices, self.aggregation_size)
            for img_idx in batch_image_indices:
                RGB, IRC = self.image_dataset[img_idx]  # RGB:  3x256x256, IRC: 1x256x256
                IRN = FPNInsert(IRC, FPN)               # IRN:  1x256x256
                
                # aggregate images to return
                sRGBList.append(RGB)
                sIRCList.append(IRC)
                sIRNList.append(IRN)
                sFPNList.append(FPN)
            
            # append sublist as a tensor to the batch list
            RGBList.append(torch.stack(sRGBList))
            IRCList.append(torch.stack(sIRCList))
            IRNList.append(torch.stack(sIRNList))
            FPNList.append(torch.stack(sFPNList))
            
        # increment the current index
        self.current += self.batch_size
        
        # create a dictionary containing all elements
        # RGB: BATCH x AGGREGATED x 3 x CROP_SIZE x CROP_SIZE
        # IRC: BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        # IRN: BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        # FPN: BATCH x AGGREGATED x 2 x CROP_SIZE x CROP_SIZE
        return {'RGB': torch.stack(RGBList), 'IRC': torch.stack(IRCList), 'IRN': torch.stack(IRNList), 'FPN': torch.stack(FPNList)}
    
    def __len__(self):
        return len(self.noise_indices) // self.batch_size
    
    def GetDataLoaderOptions(self):
        return {"input_size": self.crop_size, "sample_size": self.aggregation_size}
