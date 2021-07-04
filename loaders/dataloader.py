import os
import torch
import numpy as np
from loaders.acdc import ACDCLoader

def get_training_data(opt):
    #load training set
    loader = ACDCLoader(opt.data_path)
    #load labeled data
    data = loader.load_labelled_data(0, 'all', slice_num=3)
    limages = torch.from_numpy(data.images.astype(float))
    llabels = torch.from_numpy(data.labels.astype(float))

    #load unlabeled data
    udata = loader.load_unlabelled_data(0, 'training', slice_num=3)
    uimages = torch.from_numpy(udata.images.astype(float))
    ulabels = torch.from_numpy(udata.labels.astype(float))

    #merge labeled and part of unlabeled to create the full dataset
    all_images = torch.cat([limages, uimages], 0)
    labels = torch.cat([llabels, ulabels], 0)
    print(all_images.shape, labels.shape)
    #load the corresponding content and style factors
    anatomy_factors = torch.from_numpy(np.load(os.path.join(opt.load_factors_path, 'anatomy_factors.npz'))['arr_0'])
    modality_factors = torch.from_numpy(np.load(os.path.join(opt.load_factors_path, 'modality_factors.npz'))['arr_0'])

    #dim check
    assert all_images.shape[0] == anatomy_factors.shape[0], 'The number of images must match the number of factors.'
    return all_images, labels, anatomy_factors, modality_factors