import os
import torch
import scipy.io as sio

# import actual network
from .dlsnuc.DLSNUCNetwork import DLSNUCNetwork

class DLSNUCModel:
    """
        PyTorch implementation of DLS-NUC Model
        Model Parameters and utils folder fetched from https://github.com/hezw2016/DLS-NUC/tree/master
        verifyPyTorchOutputs.m function is used to verify the PyTorch model using the outputs of the Matlab model
        Total absolute difference between two implementation is about 0.005, which is 6x10^{-8} per pixel on average
    """
    
    # static parameters of the model
    modelName  = 'DLSNUCModel'     # name of the model
    loss_ab_weights = [0.5, 1.0]  # set the weights for alpha and beta channels at the output
    save_dir = ""                  # output directory

    # constructor for the DLS-NUC model
    def __init__(self, options, dataOpt):
    
        # get default settings from the input arguments
        self.is_train_mode = options.is_train_mode
        self.save_dir = os.path.join(options.output_dir, self.modelName)
        self.gpu_ids = options.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        # this model do not have alpha, just predicts the beta
        self.alpha = torch.ones(dataOpt['input_size'], dataOpt['input_size']).to(self.device).unsqueeze(0).unsqueeze(0)
        
        # specify the models you want to save to the disk
        self.model_names = []

        # define the DLS-NUC network models
        mcn = sio.loadmat("models/dlsnuc/original/model1.mat", squeeze_me=False)
        self.netD = DLSNUCNetwork(mcn['model']).to(self.device)
        
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
        
        # create inputs: BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        inputs = self.irn
        
        # concatanete alonf the second and third dimensions (inputs: BATCH x (AGGREGATED x 4) x CROP_SIZE x CROP_SIZE)
        inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))

        # find the network output (BATCH x AGGREGATED) x 1 x CROP_SIZE x CROP_SIZE
        beta  = -self.netD(inputs)
        alpha = self.alpha.repeat(beta.shape[0], 1, 1, 1)
        
        # create (BATCH x AGGREGATED) x 2 x CROP_SIZE x CROP_SIZE
        fpe = torch.cat((alpha, beta), dim=1).to(self.device)
        
        # split BATCH and AGGREGATED
        self.fpn_est = fpe.view(self.irn.shape[0], self.irn.shape[1], self.fpn.shape[2], self.irn.shape[3], self.irn.shape[4])
        
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
        return self.fpn_est
        
    # return the loss value as float
    def get_current_losses(self):
        return {'loss_S': float(0)}
    
    
    
                