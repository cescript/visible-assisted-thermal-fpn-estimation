import os
import torch

# main implementation of the algorithm
from .multiview.MULTIVIEWNetwork import MULTIVIEWNetwork

class MULTIVIEWModel:
    """
        Fixed Pattern Noise Removal For Multi-View Single-Sensor Infrared Camera (WACV 2024)
    """
    
    # static parameters of the model
    modelName  = 'MULTIVIEWModel'  # name of the model
    save_dir = ""                  # output directory

    # constructor for the MULTIVIEW-FPN model
    def __init__(self, options, dataOpt):
    
        # get default settings from the input arguments
        self.is_train_mode = options.is_train_mode
        self.save_dir = os.path.join(options.output_dir, self.modelName)
        self.gpu_ids = options.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # this model do not have alpha, just predicts the beta
        self.alpha = torch.ones(dataOpt['input_size'], dataOpt['input_size']).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # specify the models you want to save to the disk
        self.model_names = []
        
        # define the MULTIVIEW-FPN network models
        self.netM = MULTIVIEWNetwork(self.device)

    def set_input(self, input):
        self.rgb = input['RGB'].to(self.device)
        self.irn = input['IRN'].to(self.device)
        self.irc = input['IRC'].to(self.device)
        self.fpn = input['FPN'].to(self.device)
        
    def forward(self):
        
        # process each batch separately, self.irn is BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE, make it inp: AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        beta_est = torch.cat([self.netM(inp) for inp in self.irn], dim=0)
        
        # beta_est is BATCH x CROP_SIZE x CROP_SIZE, make it BATCH x 1 x 1 x CROP_SIZE x CROP_SIZE
        beta_est = beta_est.unsqueeze(1).unsqueeze(2)
 
        # find the network output BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        beta = -beta_est.repeat(1, self.irn.shape[1], 1, 1, 1)
        alpha = self.alpha.repeat(beta.shape[0], self.irn.shape[1], 1, 1, 1)

        # create BATCH x AGGREGATED x 2 x CROP_SIZE x CROP_SIZE
        self.fpn_est = torch.cat((alpha, beta), dim=2).to(self.device)
        
    def optimize_parameters(self):
        self.forward()

    def evaluate(self):
        """ regardless of train or test, we need gradients for multiview-fpn """
        self.forward()
            
    # generic implementation
    def update_learning_rate(self):
        """" model not applicable for training """
    
    def set_requires_grad(self, nets, requires_grad=False):
        """" model not applicable for training, never requires gradient """
                    
    def set_evalmode(self):
        """" model not applicable for training, always in eval mode """

    # get the estimated FPN
    def get_fpn(self):
        # fpn_est: BATCH x AGGREGATED x 2 x CROP_SIZE x CROP_SIZE
        return self.fpn_est
        
    # return the loss value as float
    def get_current_losses(self):
        return {'loss_S': float(0)}
    
    
    
                