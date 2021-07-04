from .generator import AdaINDecoder
from .noise_injection import NIGenerator
from .discriminators import LS_D, SNResDiscriminator
from .vgg import VGG

def get_model(opt):
    if opt.model_name == 'vgg':
        return VGG(opt)
    else:
        generator = NIGenerator(opt)
        if opt.dtype == 'spectral':
            discriminator = SNResDiscriminator(opt)
        else:
            discriminator = LS_D(opt)
        return generator, discriminator