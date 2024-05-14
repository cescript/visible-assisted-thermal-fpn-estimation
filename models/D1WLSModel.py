import os
import torch
import scipy.io as sio

# well known implementations
from .d1wls.D1WLSNetwork import D1WLSNetwork

class D1WLSModel:
    """
        1D-Weighted-Least-Square-Destriping-for-Uncooled-Infrared-Images
    """
    
    # static parameters of the model
    modelName  = 'D1WLSModel'      # name of the model
    alpha_so = [0.2, 1.0]          # scale and offset for the alpha channel of the output fpn
    beta_so = [0.2, 0.0]           # scale and offset for the beta channel of the output fpn
    save_dir = ""                  # output directory

    # constructor for the D1WLS model
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

        # define the D1WLS model using the D1WLSDESTRIPEDEMO parameters
        self.netD = D1WLSNetwork(lamda=40, titer=3)

        # no gradient will be required for this model
        for param in self.netD.parameters():
            param.requires_grad = False

        # always in evaluation mode
        self.netD.eval()

    def set_input(self, input):
        self.rgb = input['RGB'].to(self.device)
        self.irn = input['IRN'].to(self.device)
        self.irc = input['IRC'].to(self.device)
        self.fpn = input['FPN'].to(self.device)
        
    def forward(self):
        
        # create inputs by merging RGB and IR channels (inputs: BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE)
        beta_est = torch.cat([self.netD(input) for input in self.irn.view(-1, self.irn.shape[3], self.irn.shape[4])], dim=0)

        # find the network output BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        beta = -torch.reshape(beta_est, self.irn.shape)
        alpha = self.alpha.repeat(beta.shape[0], beta.shape[1], 1, 1, 1)

        # create BATCH x AGGREGATED x 2 x CROP_SIZE x CROP_SIZE
        self.fpn_est = torch.cat((alpha, beta), dim=2).to(self.device)

    def optimize_parameters(self):
        self.forward()
 
    # wrap forward with no_grad
    def evaluate(self):
        with torch.no_grad():
            self.forward()

    # evaluate the results
    def save_verification_data(self):
        sio.savemat("sample.mat", {'irn': self.irn.to('cpu'), 'result': self.fpn_est.to('cpu')})
        
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
    
    
    
                