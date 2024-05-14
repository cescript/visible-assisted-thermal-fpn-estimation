import torch

# set the training parameters
class TestOptions:
    
    # general settings
    verbose = False             # print verbosity
    
    # saving options
    save_limit = 100            # save the first # of the images during test, -1 for all
    load_iter = 'latest'        # if > 0, load the weights in the given iter

    # actual training parameters
    gpu_ids = [0]               # gpu ids
    is_train_mode = False       # test mode

    # folders
    output_dir = ".output"      # folder to store results

    def __init__(self):
        if len(self.gpu_ids) > 0:
            torch.cuda.empty_cache()