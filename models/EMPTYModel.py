import os
import torch
from .common.WeightedMSELoss import WeightedMSELoss

class EMPTYModel:
    """
        EMPTY model, just returns the [alpha=1, beta=0]
    """
    
    # static parameters of the pix2pix model
    modelName  = 'EMPTYModel'     # name of the model
    loss_ab_weights = [0.5, 1.0]  # set the weights for alpha and beta channels at the output
    save_dir = ""                  # output directory

    # constructor for the SIMPLE model
    def __init__(self, options, dataOpt):
    
        # get default settings from the input arguments
        self.is_train_mode = options.is_train_mode
        self.save_dir = os.path.join(options.output_dir, self.modelName)
        self.gpu_ids = options.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        if self.is_train_mode:
            # define loss functions and move buffers to device
            self.criterionL = WeightedMSELoss(self.loss_ab_weights).to(self.device)
            
        # specify the models you want to save to the disk
        self.model_names = []

    def set_input(self, input):
        self.rgb = input['RGB'].to(self.device)
        self.irn = input['IRN'].to(self.device)
        self.irc = input['IRC'].to(self.device)
        self.fpn = input['FPN'].to(self.device)
        
    def forward(self):
    
        alpha = torch.ones(self.fpn.shape[3], self.fpn.shape[4])
        beta = torch.zeros(self.fpn.shape[3], self.fpn.shape[4])
        
        # create 1 x 1 x 2 x CROP_SIZE x CROP_SIZE array
        fpn = torch.stack((alpha, beta), dim=0).unsqueeze(0).unsqueeze(0).to(self.device)

        # reshape the FPN to original fpn size
        self.fpn_est = fpn.repeat(self.fpn.shape[0], self.fpn.shape[1], 1, 1, 1)
        
    def optimize_parameters(self):
        self.forward()

        # calculate loss
        self.loss_S = self.criterionL(torch.squeeze(self.fpn_est, dim=1), torch.squeeze(self.fpn, dim=1))

    # wrap forward with no_grad
    def evaluate(self):
        with torch.no_grad():
            self.forward()
            
    # generic implementation
    def update_learning_rate(self):
        """" empty """
    
    def set_requires_grad(self, nets, requires_grad=False):
        """" empty """
    
    def set_evalmode(self):
        """" empty """
    
    # get the estimated FPN
    def get_fpn(self):
        return self.fpn_est
        
    # return the loss value as float
    def get_current_losses(self):
        return {'loss_S': float(self.loss_S)}
    
    
    
                