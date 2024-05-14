import os
import torch

class NetworkIO:
    """This class is responsible from saving and loading network model"""
    @staticmethod
    def save_networks(model, epoch):
        
        # check directory exist for the output
        output_path = os.path.join(model.save_dir, "checkpoints")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
      
        # check the models
        for name in model.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(output_path, save_filename)
                net = getattr(model, 'net' + name)

                if len(model.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(model.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    @staticmethod
    def load_networks(model, epoch):
        for name in model.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(model.save_dir, "checkpoints", load_filename)
                net = getattr(model, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('Loading the model from %s' % load_path)

                state_dict = torch.load(load_path, map_location=model.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    @staticmethod
    def print_networks(model, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture"""
        print('---------- Networks initialized -------------')
        for name in model.model_names:
            if isinstance(name, str):
                net = getattr(model, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        