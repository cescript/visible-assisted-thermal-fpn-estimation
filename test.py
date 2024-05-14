import models

from dataset.NoisyDataLoader import NoisyDataLoader
from options import TestOptions
from metrics import MetricEvaluator
from utility import Visualizer, FPNRemoveTensor
from models.common.NetworkIO import NetworkIO

if __name__ == '__main__':
    
    # create test options
    opt = TestOptions()
    
    # set the model names to be evaluated
    modelNames = ["EMPTYModel", "BESTModel", "DLSNUCModel", "D1WLSModel", "MULTIVIEWModel", "SAFTAModel"]
    
    # create a metric evaluator class
    metricEvaluator = MetricEvaluator(opt.output_dir)

    for mid, model in enumerate(modelNames):
        print("Evaluating %s model..." % model)
        
        # load the dataset
        dataLoader = NoisyDataLoader("dataset/cats", opt.is_train_mode)
        
        # start the evaluation of the model
        metricEvaluator.start(model)
        
        # create a model given opt.model and other options
        model = models.GenerateModel(model, opt, dataLoader.GetDataLoaderOptions())
    
        # check iterations exist or pretrained network
        NetworkIO.load_networks(model, opt.load_iter)
        
        # test with eval mode. This only affects layers like batchnorm and dropout.
        model.set_evalmode()

        # create a visualizer that display/save images and plots
        visualizer = Visualizer(model.save_dir, "results", opt.gpu_ids, save_logs=False)
        
        # start test
        for i, data in enumerate(dataLoader):

            # unpack data from data loader
            model.set_input(data)
            model.evaluate()
            
            # get the FPN estimation
            fpn_est = model.get_fpn().to('cpu')
            
            # get the clean IR estimation
            ir_noisy = data['IRN']
            ir_clean = data['IRC']
            ir_cest = FPNRemoveTensor(ir_noisy, fpn_est)
            
            # evaluate the results
            metricEvaluator.evaluate(ir_clean, ir_cest)
            
            # visualize for debug purposes
            if opt.save_limit < 0 or i < opt.save_limit:
                visualizer.save_current_results({'irc': ir_clean[0], 'irn': ir_noisy[0], 'ire': ir_cest[0]}, mid, i)
            
        # save the metric to given file
        metricEvaluator.save_metrics(reduction="mean")
