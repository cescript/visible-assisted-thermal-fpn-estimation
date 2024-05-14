import sys
import time
import models

from dataset.NoisyDataLoader import NoisyDataLoader
from options import TrainingOptions
from utility import Visualizer, FPNRemoveTensor
from models.common.NetworkIO import NetworkIO

# run training code
if __name__ == '__main__':
    
    # set the training model name
    trainingModelName = "SAFTAModel"

    # create training options
    opt = TrainingOptions(trainingModelName)

    # load the dataset
    dataLoader = NoisyDataLoader(data_root="dataset/cats", is_train_mode=opt.is_train_mode)

    # create a model given opt.model and other options
    model = models.GenerateModel(trainingModelName, opt, dataLoader.GetDataLoaderOptions())

    # check iterations exist or pretrained network
    if opt.load_iter > 0:
        print("Loading pre-trained model %s to train %s" % (opt.load_iter, model.modelName))
        NetworkIO.load_networks(model, opt.load_iter)
    
    # print it
    NetworkIO.print_networks(model, opt.verbose)

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(model.save_dir, "train", opt.gpu_ids)
    
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # timer for entire epoch
        epoch_start_time = time.time()

        for idx, data in enumerate(dataLoader):  # inner loop within one epoch
            # increase the total number of iters
            total_iters += dataLoader.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # get the visual and save it during training
            if opt.save_image_freq > 0 and total_iters % opt.save_image_freq == 0:
                # get the clean IR estimation
                ir_noisy = data['IRN']
                ir_clean = data['IRC']
                ir_mest = model.get_ire().to('cpu')
                visualizer.save_current_results({'irc': ir_clean[0], 'irn': ir_noisy[0], 'irm': ir_mest[0]}, epoch, idx)

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            NetworkIO.save_networks(model, 'latest')
            NetworkIO.save_networks(model, epoch)
        
        # print losses after each epoch
        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, total_iters, losses)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        # update learning rates at the end of every epoch
        model.update_learning_rate()
