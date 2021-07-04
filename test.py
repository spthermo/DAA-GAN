import torch
import sys
import datetime

from options.base_options import parse_arguments
from utils.label_mixing import check_label_mix, get_original_label
from utils.image_utils import save_multi_image, save_content_factors, save_gray_images
from utils.visualization import VisdomVisualizer
from loaders.dataloader import get_training_data
from models import get_model

if __name__ == "__main__":
    opt, uknown = parse_arguments(sys.argv)
    #create and init device
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))    
    gpus = [int(id) for id in opt.gpu.split(',') if int(id) >= 0]
    device = torch.device('cuda:{}' .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Test {0} on {1}'.format(opt.name, device))

    torch.manual_seed(667)
    if device.type == 'cuda':
        torch.cuda.manual_seed(667)

    #Load trained DAA-GAN
    loaded_params = torch.load(opt.load_weights_path, map_location=device)
    G, D = get_model(opt)
    G.load_state_dict(loaded_params['generator_state_dict'])
    G.to(device)
    loaded_vgg_params = torch.load(opt.load_vgg_weights_path, map_location=device)
    opt.model_name = 'vgg'
    VGG = get_model(opt)
    VGG.load_state_dict(loaded_vgg_params['model_state_dict'])
    VGG.to(device)

    #load data
    all_images, labels, anatomy_factors, modality_factors = get_training_data(opt)

    #auxiliary tensors init
    b_images = torch.zeros(1, 1, 224, 224)
    b_label = torch.zeros(1, 1)

    b_images2 = torch.zeros(1, 1, 224, 224)
    b_label2 = torch.zeros(1, 1)

    a_out = torch.zeros(1, opt.anatomy_out_channels, 224, 224)
    mu_out = torch.zeros(1, 8)
    a_out_2 = torch.zeros(1, opt.anatomy_out_channels, 224, 224)

    mixed = torch.zeros(1, 1)
    mixed_label = torch.zeros(1, 1)
    aggregated_noise_mask = torch.zeros(1, 1, 224, 224)
    aggregated_source_mask = torch.zeros(1, 1, 224, 224)
    zero_mask = torch.zeros(1, 224, 224)
    
    in_batch_iter = 0
    G.eval()
    VGG.eval()
    with torch.no_grad():
        idx2 = torch.randperm(all_images.shape[0])
        acc = 0
        for iteration in range(all_images.shape[0]):
            #init batch-wise losses
            b_images[0] = all_images[iteration]
            b_original_label = get_original_label(b_label[0])
            b_images2[0] = all_images[idx2[iteration]]
            a_out[0] = anatomy_factors[iteration]
            b_label[0] = labels[iteration]
            b_original_label = get_original_label(b_label[0])
            b_label2[0] = labels[idx2[iteration]]
            b_original_label2 = get_original_label(b_label2[0])
            a_out_2[0] = anatomy_factors[idx2[iteration]]
            mu_out[0] = modality_factors[iteration]

            augmented_a_out = a_out.clone()
            augmented_a_out, aggregated_noise_mask, _, mixed, _ = check_label_mix(2, 3, 4, \
                            augmented_a_out, a_out, a_out_2, aggregated_noise_mask, \
                            aggregated_source_mask, zero_mask, mixed, b_label, b_label2)
            if mixed[0] > 0:
                mixed_label[0] = b_label2[0]
            else:
                mixed_label[0] = b_label[0]
            gen, noisy_a_out = G(augmented_a_out.to(device), mu_out.to(device), aggregated_noise_mask.to(device))
            
            #predict pathology in the generated sample
            pred, _ = VGG(gen)
            _, indices = torch.max(pred, 1)
            pred_class = get_original_label(indices[0])
            
            #save input and generated images for FID computation
            if opt.save_images and mixed[0] > 0:
                save_gray_images(gen.squeeze(0).detach().cpu(), opt.generated_path, iteration)
