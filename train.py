import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import datetime

from supervision import anatomy_consistency_loss, masked_l1_loss
from utils.checkpoint import save_network_state
from utils.checkpoint import count_parameters
from utils.label_mixing import check_label_mix
from utils.visualization import VisdomVisualizer
from loaders.dataloader import get_training_data
from options.base_options import parse_arguments
from models import get_model

if __name__ == "__main__":
    opt, uknown = parse_arguments(sys.argv)
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))    
    gpus = [int(id) for id in opt.gpu.split(',') if int(id) >= 0]
    device = torch.device('cuda:{}' .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Training {0} for {1} epochs using a batch size of {2} on {3}'.format(opt.name, opt.epochs, opt.batch_size, device))

    torch.manual_seed(667)
    if device.type == 'cuda':
        torch.cuda.manual_seed(667)
    visualizer = VisdomVisualizer(opt.name, count=1)

    #load DAA-GAN noise injection+generator, discriminator modules
    G, D = get_model(opt)
    model_dict = G.state_dict()
    if opt.load_sdnet_decoder_weights_path != '':
        pretrained_model = torch.load(opt.load_sdnet_decoder_weights_path, map_location=device)
        pretrained_dict = pretrained_model['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        G.load_state_dict(model_dict)
    G.to(device)
    num_paramsG = count_parameters(G)
    D.to(device)
    num_paramsD = count_parameters(D)
    num_params = num_paramsD + num_paramsG

    #load pretrained pathology classifier weights
    if opt.load_vgg_weights_path != '':
        opt.model_name = 'vgg'
        VGG = get_model(opt)
        pretrained = torch.load(opt.load_vgg_weights_path, map_location=device)
        VGG.load_state_dict(pretrained['model_state_dict'])
        VGG.to(device)
    else:
        print("Could not find VGG weights in the specified path ({})".format(opt.load_vgg_weights_path), file=sys.stderr)
        sys.exit()
    print('Model Parameters: ', num_params)

    #load data
    all_images, labels, anatomy_factors, modality_factors = get_training_data(opt)

    #optimizer initialization
    optimizerG = optim.Adam(G.parameters(), betas=(0.0,0.999), lr=opt.learning_rate)
    optimizerD = optim.Adam(D.parameters(), betas=(0.0,0.999), lr=opt.learning_rate)

    #loss initialization
    nll_loss = nn.NLLLoss().to(device)

    #auxiliary tensors init
    b_images = torch.zeros(opt.batch_size, 1, opt.dim, opt.dim)
    b_labels = torch.zeros(opt.batch_size, 1)
    
    b_images2 = torch.zeros(opt.batch_size, 1, opt.dim, opt.dim)
    b_labels2 = torch.zeros(opt.batch_size, 1)
    
    a_out = torch.zeros(opt.batch_size, opt.anatomy_out_channels, opt.dim, opt.dim)
    mu_out = torch.zeros(opt.batch_size, 8)
    a_out_2 = torch.zeros(opt.batch_size, opt.anatomy_out_channels, opt.dim, opt.dim)
    
    mixed = torch.zeros(opt.batch_size, 1)
    mixed_real = torch.zeros(opt.batch_size, 1, opt.dim, opt.dim)
    mixed_labels = torch.zeros(opt.batch_size, 1)
    
    aggregated_noise_mask = torch.zeros(opt.batch_size, 1, opt.dim, opt.dim)
    aggregated_source_mask = torch.zeros(opt.batch_size, 1, opt.dim, opt.dim)
    zero_mask = torch.zeros(1, opt.dim, opt.dim)
    
    #create real/fake labels for Discriminator training
    real_labels = torch.ones(opt.batch_size).to(device)
    real_labels -= 0.1 # label smoothing
    fake_labels = torch.zeros(opt.batch_size).to(device)

    #train process
    total_batches = all_images.shape[0] // opt.batch_size
    global_iterations = 0
    for epoch in range(opt.epochs):
        idx = torch.randperm(all_images.shape[0])
        in_batch_iter = 0
        if opt.load_sdnet_decoder_weights_path != '':
            G.eval()
        else:
            G.train()
        D.train()
        VGG.eval()
        for iteration in range(all_images.shape[0]):
            idx2 = torch.randperm(all_images.shape[0])
            if (iteration + opt.batch_size) > all_images.shape[0]:
                break
            if in_batch_iter < opt.batch_size:
                #sample subject 1 and 2 images and the corresponding content and style factors
                b_images[in_batch_iter] = all_images[idx[iteration]]
                b_labels[in_batch_iter] = labels[idx[iteration]]
                b_images2[in_batch_iter] = all_images[idx2[iteration]]
                b_labels2[in_batch_iter] = labels[idx2[iteration]]
                a_out[in_batch_iter] = anatomy_factors[idx[iteration]]
                mu_out[in_batch_iter] = modality_factors[idx[iteration]]
                a_out_2[in_batch_iter] = anatomy_factors[idx2[iteration]]
                in_batch_iter += 1
            else:
                optimizerD.zero_grad()
                augmented_a_out = a_out.clone()
                #anatomy mixing based on pathology labels
                augmented_a_out, aggregated_noise_mask, aggregated_source_mask, mixed, _ = check_label_mix(2, 3, 4, \
                                            augmented_a_out, a_out, a_out_2, aggregated_noise_mask, aggregated_source_mask, \
                                            zero_mask, mixed, b_labels, b_labels2)
                for i in range(mixed.shape[0]):
                    if mixed[i] > 0:
                        mixed_real[i] = b_images2[i]
                        mixed_labels[i] = b_labels2[i]
                    else:
                        mixed_real[i] = b_images[i]
                        mixed_labels[i] = b_labels[i]
                real_input = aggregated_source_mask * mixed_real
                real_output = D(real_input.detach().to(device)).squeeze()

                #lsgan discriminator loss - real
                real_disc_loss = 0.5 * torch.mean((real_output-real_labels)**2)
                reco, noisy_a_out = G(augmented_a_out.to(device), mu_out.to(device), aggregated_noise_mask.to(device))
                consistency_loss = anatomy_consistency_loss(noisy_a_out, augmented_a_out.to(device), aggregated_noise_mask.to(device))
                fake_output = D(reco.detach()).squeeze()
                if opt.load_vgg_weights_path is not None:
                    pathology_output, _ = VGG(reco.detach())
                    pathology_loss = nll_loss(pathology_output, mixed_labels.squeeze(1).long().to(device))
                l1_masked_loss, l1_masked_loss_map = masked_l1_loss(reco, b_images.to(device), aggregated_source_mask.to(device))
                
                #lsgan discriminator loss - fake
                fake_disc_loss = 0.5 * torch.mean((fake_output-fake_labels)**2)
                batch_D_loss = real_disc_loss + fake_disc_loss
                batch_D_loss.backward()         
                optimizerD.step()
                D_x = real_output.mean()
                optimizerG.zero_grad()
                fake_input = aggregated_source_mask.to(device) * reco
                fake_output = D(fake_input).squeeze()
                
                #lsgan generator loss
                batch_fake_gen_loss = 0.5 * torch.mean((fake_output-real_labels)**2)
                generator_loss = batch_fake_gen_loss + pathology_loss + 10*(l1_masked_loss + consistency_loss)
                losses = {
                    'train_ADV': batch_fake_gen_loss.item(),
                    'train_PATH': pathology_loss.item(),
                    'train_BG': l1_masked_loss.item(),
                    'train_CONS': consistency_loss.item(),
                    'train_DISC': batch_D_loss.item()
                }
                
                #backprop and optimizer update
                generator_loss.backward()
                optimizerG.step()

                #visualizations
                if (iteration + 1) % opt.visdom_iters == 0:
                    visualizer.show_map(b_images.to(device), 'A_')
                    visualizer.show_map(b_images2.to(device), 'B_')
                    visualizer.show_map(reco, 'A_prime')
                    if opt.print_factors:
                        visualizer.show_anatomical_factors(augmented_a_out, 'Mixed')
                        if opt.model_name == 'noise':
                            visualizer.show_map(aggregated_noise_mask.to(device), 'Noise_patch_')
                            visualizer.show_anatomical_factors(noisy_a_out, 'Noisy')
                if (iteration + 1) % opt.disp_iters <= opt.batch_size:
                    visualizer.append_loss(epoch, global_iterations, generator_loss.item(), "train_COMBINED")
                    visualizer.append_loss(epoch, global_iterations, batch_fake_gen_loss.item(), "train_G_loss")
                    visualizer.append_loss(epoch, global_iterations, consistency_loss.item(), "train_cons_loss")
                    visualizer.append_loss(epoch, global_iterations, l1_masked_loss.item(), "train_bg_loss")
                    visualizer.append_loss(epoch, global_iterations, batch_D_loss.item(), "train_D_loss")
                    visualizer.append_loss(epoch, global_iterations, pathology_loss.item(), "train_path_loss")
                    visualizer.append_loss(epoch, global_iterations, D_x.item(), "D(x)")
                global_iterations += opt.batch_size
                in_batch_iter = 0

        print("Epoch {} checkpoint".format(epoch))
        if epoch == opt.epochs - 1 :
            current_dir = os.getcwd()
            final_dir = os.path.join(current_dir, opt.save_path)
            save_network_state(G, D, opt.dim, opt.ndf, \
                            opt.anatomy_out_channels, \
                            opt.z_length, optimizerG, optimizerD, \
                            epoch, opt.name + "_model_state_epoch_" + str(epoch), \
                            final_dir)

        

