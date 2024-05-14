import os
import torch
from torchvision.utils import save_image

# well known implementations
from .safta.SAFTANetwork import SAFTAGenerator, SAFTADenoiser
from .common.WeightedFrequencyLoss import WeightedFrequencyLoss
from .common.WeightedMSELoss import WeightedMSELoss
from .common.InitNetwork import InitNetwork
from .common.GetScheduler import get_scheduler
from utility.FPNUtility import FPNRemoveTensor

class SAFTAModel:
    """
        Self Attended Feature Temporal Aggregation Module for IR-FPA denoising
    """
    
    # static parameters of the model
    modelName  = 'SAFTAModel'      # name of the model
    ginput_nc  = 4                 # number of input channels (RGB + IR)
    goutput_nc = 1                 # output of the generator is IRe
    dinput_nc  = 2                 # input for the fpn estimator (IRn + IRe)
    doutput_nc = 2                 # output for the fpn estimator is alpha,beta
    lambda_im  = 0.15              # effect of image reconstruction loss over all loss
    alpha_so = [0.2, 1.0]          # scale and offset for the alpha channel of the output fpn
    beta_so = [0.2, 0.0]           # scale and offset for the beta channel of the output fpn
    loss_lh_weights = [0.6, 0.4]   # set the weights for Low and High Frequency Loss at the output
    loss_ab_weights = [0.5, 1.0]   # set the weights for alpha and beta channels at the output
    init_type = 'normal'           # network initialization [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02               # scaling factor for normal, xavier and orthogonal.
    save_dir = ""                  # output directory

    # constructor for the safta model
    def __init__(self, options, dataOpt):
    
        # get default settings from the input arguments
        self.is_train_mode = options.is_train_mode
        self.save_dir = os.path.join(options.output_dir, self.modelName)
        self.gpu_ids = options.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.agg_size = dataOpt['sample_size']
        # specify the models you want to save to the disk
        self.model_names = ['G', 'D']

        # define the SAFTA network models
        self.netG = SAFTAGenerator(self.ginput_nc, self.goutput_nc)
        self.netD = SAFTADenoiser(self.dinput_nc * self.agg_size, self.doutput_nc, self.alpha_so, self.beta_so)

        # init models
        self.netG = InitNetwork(self.netG, self.init_type, self.init_gain, self.gpu_ids)
        self.netD = InitNetwork(self.netD, self.init_type, self.init_gain, self.gpu_ids)
        
        if self.is_train_mode:
            # define loss functions and move buffers to device
            self.criterionF = WeightedMSELoss(self.loss_ab_weights).to(self.device)
            self.criterionI = WeightedFrequencyLoss(self.loss_lh_weights).to(self.device)
            
            # initialize optimizers using the all parameters from G and D networks
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=options.lr, betas=(options.beta1, 0.999))

            # create schedulers
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.schedulers = [get_scheduler(optimizer, options) for optimizer in self.optimizers]

    def set_input(self, input):
        self.rgb = input['RGB'].to(self.device)
        self.irn = input['IRN'].to(self.device)
        self.irc = input['IRC'].to(self.device)
        self.fpn = input['FPN'].to(self.device)
        
    def forward(self):
    
        # create inputs by merging RGB and IR channels (inputs: BATCH x AGGREGATED x 4 x CROP_SIZE x CROP_SIZE)
        inputs = torch.cat([self.rgb, self.irn], dim=2)
    
        # make the input AGGREGATED x BATCH x 4 x CROP_SIZE x CROP_SIZE for loop
        inputs = inputs.transpose(dim0=0, dim1=1)

        # get the estimated clean images (ir_est: BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE
        self.ire = torch.cat([self.netG(ins) for ins in inputs], dim=1).unsqueeze(dim=2)
        
        # debug purpose
        # save_image(self.ir_est[0].unsqueeze(dim=1), "batch1.png")
        # save_image(inputs[0].unsqueeze(dim=1), "input1.png")

        # create denoising network input (self.irn: BATCH x AGGREGATED x 1 x CROP_SIZE x CROP_SIZE)
        dinput = torch.cat((self.ire, self.irn), dim=1).squeeze(dim=2)

        # netD: BATCH x 2 x CROP_SIZE x CROP_SIZE --> BATCH x AGGREGATED x 2 x CROP_SIZE x CROP_SIZE
        self.fpn_est = self.netD(dinput).unsqueeze(dim=1).repeat(1, self.agg_size, 1, 1, 1)

    def calculate_image_loss(self, denoised):
        # ir_est is BATCH x AGG x 1 x HEIGHT x WIDTH, make it (BATCH x AGG) x 1 x HEIGHT x WIDTH
        ire_reshaped = denoised.reshape(-1, denoised.shape[2], denoised.shape[3], denoised.shape[4])
        irc_reshaped = self.irc.reshape(-1, self.irc.shape[2], self.irc.shape[3], self.irc.shape[4])
        
        return self.criterionI(ire_reshaped, irc_reshaped)

    def calculate_fpn_loss(self, fpnest):
        # fpn_est is BATCH x AGG x 2 x HEIGHT x WIDTH, make it (BATCH x AGG) x 2 x HEIGHT x WIDTH
        fpe_reshaped = fpnest.reshape(-1, fpnest.shape[2], fpnest.shape[3], fpnest.shape[4])
        fpn_reshaped = self.fpn.reshape(-1, self.fpn.shape[2], self.fpn.shape[3], self.fpn.shape[4])
    
        return self.criterionF(fpe_reshaped, fpn_reshaped)
    
    def optimize_parameters(self):
        self.forward()

        # calculate the fpn and ir estimation losses
        self.loss_I = self.calculate_image_loss(self.ire)
        self.loss_F = self.calculate_fpn_loss(self.fpn_est)
        self.loss_T = self.loss_F + self.lambda_im * self.loss_I
        
        # # set G and D's gradients to zero
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

        # calculate gradients for D and G
        self.loss_T.backward()

        # update the optimizer
        self.optimizer_D.step()
        self.optimizer_G.step()
        
    # wrap forward with no_grad
    def evaluate(self):
        with torch.no_grad():
            self.forward()
            
    # generic implementation
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('Learning rate %.7f -> %.7f' % (old_lr, lr))
    
    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def set_evalmode(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    # get the estimated FPN
    def get_fpn(self):
        return self.fpn_est

    # get the estimated IR image
    def get_ire(self):
        #  BATCH x AGG x 1 x CROP_SIZE x CROP_SIZE
        return self.ire
    
    # return the loss value as float
    def get_current_losses(self):
        return {'loss_F': float(self.loss_F), 'loss_I': float(self.loss_I), 'loss_T': float(self.loss_T)}
    
    
    
                
