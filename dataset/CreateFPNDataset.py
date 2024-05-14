import random
import numpy as np
import torchvision.transforms.functional as tfunction

# custom FPN dataset
class CreateFPNDataset:

    def __init__(self, crop_size, noise_count):
    
        # get the copy of the options
        self.crop_size = crop_size
        self.noise_count = noise_count
        
        # noise parameters
        self.alpha_scale = 0.015
        self.beta_sigma  = 0.015
        
        # narcissism parameters
        self.n_sigma = [0.1, 0.2]
        self.n_square = 32
        self.n_probability = 1.0
        self.n_gain = [-0.1, 0.1]
        
        # create noise patterns
        self.noise_patterns = []
        for idx in range(self.noise_count):
            self.noise_patterns.append(self.GenerateNoise())

        print(f"FPN noise dataset loaded successfully with {self.noise_count} unique fpn")
        
    def __getitem__(self, index):
    
        # always make sure index is within the range of the images
        indexI = index % self.noise_count
        FPN = tfunction.to_tensor(self.noise_patterns[indexI])

        return FPN
    
    def __len__(self):
        return self.noise_count
    
    # generate FPN noise
    def GenerateNoise(self):
    
        # create pattern
        pattern = np.ndarray([self.crop_size, self.crop_size, 2], dtype=np.single)
        pattern[:, :, 0] = 1.0  # alpha
        pattern[:, :, 1] = 0.0  # beta

        # insert column noise
        for i in range(self.crop_size):
            pattern[:, i, 0] = random.uniform(1 - self.alpha_scale, 1 + self.alpha_scale)
            pattern[:, i, 1] = random.gauss(0, self.beta_sigma)
        
        # create narcissism
        if self.n_probability >= random.uniform(0, 1):
            cx = self.crop_size // 2 + random.randint(-self.n_square, self.n_square)
            cy = self.crop_size // 2 + random.randint(-self.n_square, self.n_square)
            s1 = self.crop_size * random.uniform(self.n_sigma[0], self.n_sigma[1])

            # update the beta term (gaussian output will be (0.0,0.1) range)
            gain = random.uniform(self.n_gain[0], self.n_gain[1])
            pattern[:, :, 1] += gain * self.gaus2d(self.crop_size, cx, cy, s1, s1)
        
        # return the stripe and radial noise
        return pattern

    # define un-normalized 2D gaussian
    @staticmethod
    def gaus2d(crop_size, mx=0, my=0, sx=1, sy=1):
        x, y = np.indices((crop_size, crop_size))
        return np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))
