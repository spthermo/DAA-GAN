import torch
from os import path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_network_state(generator, discriminator, dim, ndf, anatomy_out_channels, z_length, optimizerG , optimizerD, epoch , name , save_path):
    if not path.exists(save_path):
        raise ValueError("{} not a valid path to save model state".format(save_path))
    torch.save(
        {
            'epoch' : epoch,
            'width': dim,
            'height': dim,
            'ndf' : ndf,
            'anatomy_out_channels' : anatomy_out_channels,
            'z_length' : z_length,
            'generator_state_dict' : generator.state_dict(),
            'discriminator_state_dict' : discriminator.state_dict(),
            'optimizerG_state_dict' : optimizerG.state_dict(),
            'optimizerD_state_dict' : optimizerD.state_dict()
        }, path.join(save_path, name))
