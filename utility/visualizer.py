import os
import torch
import torchvision
from torchvision.utils import save_image
from .FPNUtility import FPNRemove, FPNNormalize

class Visualizer:
    
    def __init__(self, path, imageFolder, gpu_ids, save_logs=True):
        self.output_directory = os.path.join(path, imageFolder)
        self.log_name = os.path.join(path, "logs.txt")
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        
        # create output directory for the images and logs
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        # check the log file and delete if exist
        if save_logs and os.path.isfile(self.log_name):
            os.remove(self.log_name)
            
    def save_current_results(self, visuals, epoch, im_id=0):
        # save images to the disk
        for label, image in visuals.items():
            img_path = os.path.join(self.output_directory, 'ep%.3d_img%05d_%s.png' % (epoch, im_id, label))
            save_image(image, img_path)
            
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses):
        """print current losses on console; also save the losses to the disk
    
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = 'Losses[epoch: %03d, iters: %09d]: ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.8f ' % (k, v)
    
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        