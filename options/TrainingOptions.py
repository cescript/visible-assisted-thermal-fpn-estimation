import torch

# set the training parameters
class TrainingOptions:
    
    # general settings
    verbose = False             # print verbosity
    
    # saving options
    save_image_freq = 0         # frequency of saving images
    save_epoch_freq = 5         # frequency of saving checkpoints at the end of epochs
    load_iter = 0               # if > 0, load the weights in the given iter
    epoch_count = 1             # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

    # actual training parameters
    gpu_ids = [0]               # gpu ids
    is_train_mode = True   # train mode
    
    # folders
    name = "training_name"      # name of the training (possibly derived from parameters)
    output_dir = ".output"      # folder to store models and visuals

    def __init__(self, modelName):
        if len(self.gpu_ids) > 0:
            torch.cuda.empty_cache()
        
        # set the training options based on the selected model
        if modelName == 'SAFTAModel':
            self.n_epochs = 40          # number of epochs with the initial learning rate
            self.n_epochs_decay = 20    # number of epochs to linearly decay learning rate to zero
            self.beta1 = 0.9            # momentum term of adam
            self.lr = 0.0001            # initial learning rate for adam
            self.lr_policy = 'linear'   # learning rate policy. [linear | step | cosine]
            self.lr_decay_iters = 25    # multiply by a gamma every lr_decay_iters iterations (for step)
        else:
            assert False, "Unable to create training options for unknown model"
