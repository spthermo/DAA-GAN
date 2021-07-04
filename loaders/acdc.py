import os
import nibabel as nib
import numpy as np
import utils.data_utils

from loaders.base_loader import Loader
from skimage import transform
from loaders.data import Data

import logging

# sys.path.append('..')
# sys.path.append('./loaders')

class ACDCLoader(Loader):
    def __init__(self, data_path):
        super(ACDCLoader, self).__init__()
        self.num_masks = 3
        self.num_volumes = 100
        self.input_shape = (224, 224, 1)
        self.data_folder = data_path
        self.log = logging.getLogger('acdc')

    def splits(self):
        '''
        :return: an array of splits into validation, test and train indices
        '''

        splits = [
            {'validation': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             'test': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
             'training': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                          47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                          64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
             },
            {'validation': [85, 13, 9, 74, 73, 68, 59, 79, 47, 80, 14, 95, 25, 92, 87],
             'test': [54, 55, 99, 63, 91, 24, 51, 3, 64, 43, 61, 66, 96, 27, 76],
             'training': [46, 57, 49, 34, 17, 8, 19, 28, 97, 1, 90, 22, 88, 45, 12, 4, 5,
                          75, 53, 94, 62, 86, 35, 58, 82, 37, 84, 93, 6, 33, 15, 81, 23, 48,
                          71, 70, 11, 77, 36, 60, 31, 65, 32, 78, 98, 52, 100, 42, 38, 2, 20,
                          69, 26, 18, 40, 50, 16, 7, 41, 10, 83, 21, 39, 72, 56, 67, 44, 30, 89, 29]
             },
            {'validation': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
             'test': [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
             'training': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                          100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 62, 63,
                          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 79, 80,
                          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
             },
            {'validation': [20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 33, 34, 35, 36],
             'test': [38, 39, 40, 41, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55],
             'training': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                          21, 22, 32, 42, 44, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                          71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                          91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
             },
            {'validation': [11, 12, 14, 15, 70, 62, 21, 64, 69, 93, 94, 72, 75, 99, 100],
             'test': [16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30],
             'training': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                          47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63,
                          65, 66, 67, 68, 71, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 
                          85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13]
             }
        ]

        return splits

    def load_scanner_type(self):
        """
        Load a text file of volume to MR scanner type (1.5T or 3T), for example:
         1, 1.5
         2, 1.5
         3, 3
         4, 3
         etc
        :return: a map of volume to scanner type
        """
        folder_name = self.data_folder + '/acdc_t1_5_vs_T3.txt'
        if not os.path.exists(folder_name):
            self.log.warn('Scanner type text does not exist. Initialising with -1')
            vol_scanner = {v: -1 for v in self.volumes}
        else:
            vol_scanner = {}
            with open(folder_name, 'r') as f:
                for l in f:
                    pat, sc = l.split('\t')
                    vol_scanner[int(pat[-3:])] = float(sc.replace('\n', '').replace(',', '.'))

        return vol_scanner

    def load_labelled_data(self, split, split_type, modality='MR', normalise=False, value_crop=True, downsample=1, slice_num=None):
        if split < 0 or split > 4:
            raise ValueError('Invalid value for split: %d. Allowed values are 0, 1, 2.' % split)
        if split_type not in ['training', 'validation', 'test', 'all']:
            raise ValueError('Invalid value for split_type: %s. Allowed values are training, validation, test, all'
                             % split_type)

        npz_prefix = 'norm_' if normalise else 'unnorm_'

        # If numpy arrays are not saved, load and process raw data
        if not os.path.exists(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_images.npz')):
            if slice_num is not None:
                images, masks_lv, masks_rv, masks_myo, index, labels, phase = self.load_per_slice_raw_labelled_data(slice_num, normalise, value_crop)
            else:
                images, masks_lv, masks_rv, masks_myo, index = self.load_raw_labelled_data(normalise, value_crop)
            # save numpy arrays
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_images'), images)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_masks_lv'), masks_lv)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_masks_rv'), masks_rv)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_masks_myo'), masks_myo)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_index'), index)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_labels'), labels)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_phase'), phase)
        # Load data from saved numpy arrays
        else:
            images    = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_images.npz'))['arr_0']
            masks_lv  = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_masks_lv.npz'))['arr_0']
            masks_rv  = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_masks_rv.npz'))['arr_0']
            masks_myo = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_masks_myo.npz'))['arr_0']
            index     = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_index.npz'))['arr_0']
            labels    = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_labels.npz'))['arr_0']
            phase    = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_phase.npz'))['arr_0']

        assert images is not None and masks_myo is not None and masks_lv is not None and masks_rv is not None \
               and index is not None, 'Could not find saved data'

        assert images.max() == 1 and images.min() == -1, \
            'Images max=%.3f, min=%.3f' % (images.max(), images.min())

        self.log.debug('Loaded compressed acdc data of shape: ' + str(images.shape) + ' ' + str(index.shape))

        vol_scanner = self.load_scanner_type()

        # Case to load data from all splits.
        if split_type == 'all':
            masks = np.concatenate([masks_myo, masks_lv, masks_rv], axis=1)
            scanner = index.copy()
            for v in self.volumes:
                scanner[index == v] = vol_scanner[v]
            return Data(images, masks, index, labels, phase, scanner, downsample)

        # Select images belonging to the volumes of the split_type (training, validation, test)
        volumes = self.splits()[split][split_type]
        
        images = np.concatenate([images[index == v] for v in volumes])
        masks = np.concatenate([masks_myo, masks_lv, masks_rv], axis=-3)
        assert masks.max() == 1 and masks.min() == 0, 'Masks max=%.3f, min=%.3f' % (masks.max(), masks.min())
        masks = np.concatenate([masks[index == v] for v in volumes])
        assert images.shape[0] == masks.shape[0]
        
        # Select labels belonging to the volumes of the split_type (training, validation, test)
        labels = np.concatenate([labels[index == v] for v in volumes])
        assert images.shape[0] == labels.shape[0]

        # Select phase belonging to the volumes of the split_type (training, validation, test)
        phase = np.concatenate([phase[index == v] for v in volumes])
        assert images.shape[0] == phase.shape[0]

        # create a volume index
        index = np.concatenate([index[index == v] for v in volumes])
        assert images.shape[0] == index.shape[0]
        
        scanner = index.copy()
        for v in volumes:
            scanner[index == v] = vol_scanner[v]
        self.log.debug(split_type + ' set: ' + str(images.shape))
        return Data(images, masks, index, labels, phase, scanner, downsample)

    def load_unlabelled_data(self, split, split_type, modality='MR', normalise=True, value_crop=True, slice_num=None):
        if split < 0 or split > 4:
            raise ValueError('Invalid value for split: %d. Allowed values are 0, 1, 2.' % split)
        if split_type not in ['training', 'validation', 'test', 'all']:
            raise ValueError('Invalid value for split_type: %s. Allowed values are training, validation, test, all'
                             % split_type)
        
        npz_prefix = 'unlabeled_norm_' if normalise else 'unlabeled_unnorm_'

        # If numpy arrays are not saved, load and process raw data
        if not os.path.exists(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_images.npz')):
            if slice_num is not None:
                images, index, labels, phase = self.load_per_slice_raw_unlabelled_data(slice_num, True, normalise, value_crop)
                masks = np.zeros(images.shape)
            else:
                images, index = self.base_load_unlabelled_images('acdc', split, split_type, False, normalise, value_crop)
                masks = np.zeros(shape=(images.shape[:-1]) + (1,))
            # save numpy arrays
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_images'), images)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_index'), index)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_labels'), labels)
            np.savez_compressed(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_phase'), phase)
        # Load data from saved numpy arrays
        else:
            images    = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_images.npz'))['arr_0']
            index     = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_index.npz'))['arr_0']
            labels    = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_labels.npz'))['arr_0']
            phase    = np.load(os.path.join(self.data_folder, str(slice_num) + '_' + npz_prefix + 'acdc_phase.npz'))['arr_0']
            masks = np.zeros(images.shape)

        vol_scanner = self.load_scanner_type()
        scanner = index.copy()
        for v in self.volumes:
            scanner[index == v] = vol_scanner[v]
        return Data(images, masks, index, labels, phase, scanner)

    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        images, index = self.base_load_unlabelled_images('acdc', split, split_type, True, normalise, value_crop)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))

        vol_scanner = self.load_scanner_type()
        scanner = index.copy()
        for v in self.volumes:
            scanner[index == v] = vol_scanner[v]
        return Data(images, masks, index, scanner)

    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        self.log.debug('Loading acdc data from original location')
        images, masks_lv, masks_rv, masks_myo, index = [], [], [], [], []

        for patient_i in self.volumes:
            patient = 'patient%03d' % patient_i
            patient_folder = os.path.join(self.data_folder, patient)

            gt = [f for f in os.listdir(patient_folder) if 'gt' in f and f.startswith(patient + '_frame')]
            ims = [f.replace('_gt', '') for f in gt]

            for i in range(len(ims)):
                im = self.process_raw_image(ims[i], patient_folder, value_crop, normalise)
                im = np.expand_dims(im, axis=-1)

                m = self.resample_raw_image(gt[i], patient_folder, binary=True)
                m = np.expand_dims(m, axis=-1)

                images.append(im)

                # convert 3-dim mask array to 3 binary mask arrays for lv, rv, myo
                m_lv = m.copy()
                m_lv[m != 3] = 0
                m_lv[m == 3] = 1
                masks_lv.append(m_lv)

                m_rv = m.copy()
                m_rv[m != 1] = 0
                m_rv[m == 1] = 1
                masks_rv.append(m_rv)

                m_myo = m.copy()
                m_myo[m != 2] = 0
                m_myo[m == 2] = 1
                masks_myo.append(m_myo)

                index += [patient_i] * im.shape[2]

        assert len(images) == len(masks_myo)

        # move slice axis to the first position
        images    = [np.moveaxis(im, 2, 0) for im in images]
        masks_lv  = [np.moveaxis(m, 2, 0) for m in masks_lv]
        masks_rv  = [np.moveaxis(m, 2, 0) for m in masks_rv]
        masks_myo = [np.moveaxis(m, 2, 0) for m in masks_myo]

        # crop images and masks to the same pixel dimensions and concatenate all data
        images_cropped, masks_lv_cropped = utils.data_utils.crop_same(images, masks_lv, (224, 224))
        _, masks_rv_cropped = utils.data_utils.crop_same(images, masks_rv, (224, 224))
        _, masks_myo_cropped = utils.data_utils.crop_same(images, masks_myo, (224, 224))

        for i in range (len(images_cropped)):
            images_cropped[i] = utils.data_utils.normalise(images_cropped[i], -1, 1)

        images_cropped    = np.concatenate(images_cropped, axis=0)
        masks_lv_cropped  = np.concatenate(masks_lv_cropped, axis=0)
        masks_rv_cropped  = np.concatenate(masks_rv_cropped, axis=0)
        masks_myo_cropped = np.concatenate(masks_myo_cropped, axis=0)
        index             = np.array(index)

        self.log.debug(str(images[0].shape) + ', ' + str(masks_lv[0].shape))

        return images_cropped.transpose(0,3,1,2), masks_lv_cropped.transpose(0,3,1,2), masks_rv_cropped.transpose(0,3,1,2), masks_myo_cropped.transpose(0,3,1,2), index

    def resample_raw_image(self, mask_fname, patient_folder, binary=True, slice_num=None):
        """
        Load raw data (image/mask) and resample to fixed resolution.
        :param mask_fname:     filename of mask
        :param patient_folder: folder containing patient data
        :param binary:         boolean to define binary masks or not
        :return: the resampled image
        """
        m_nii_fname = os.path.join(patient_folder, mask_fname)
        new_res = (1.37, 1.37)
        print('Resampling %s at resolution %s to file %s' % (m_nii_fname, str(new_res), new_res))
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()

        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1

        result = []
        dims = im_data.shape
        if len(dims) > 3 and slice_num is not None:
            for i in range(im_data.shape[-1]):
                inner_im_data = im_data[..., i]
                im = inner_im_data[..., slice_num]
                rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
                # if rescaled.shape[0] < rescaled.shape[1]:
                rotated = transform.rotate(rescaled, 270.0)
                result.append(np.expand_dims(rotated, axis=-1))
                # else:
                # result.append(np.expand_dims(rescaled, axis=-1))
        else:
            for i in range(im_data.shape[-1]):
                im = im_data[..., i]
                rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
                rotated = transform.rotate(rescaled, 270.0)
                result.append(np.expand_dims(rotated, axis=-1))
        return np.concatenate(result, axis=-1)


    def process_raw_image(self, im_fname, patient_folder, value_crop, normalise, slice_num=None):
        """
        Normalise and crop extreme values of an image
        :param im_fname:        filename of the image
        :param patient_folder:  folder of patient data
        :param value_crop:      True/False to crop values between 5/95 percentiles
        :param normalise:       True/False normalise images
        :return:                a processed image
        """
        im = self.resample_raw_image(im_fname, patient_folder, binary=False, slice_num=slice_num)

        # crop to 5-95 percentile
        if value_crop:
            p5 = np.percentile(im.flatten(), 5)
            p95 = np.percentile(im.flatten(), 95)
            im = np.clip(im, p5, p95)

        # normalise to -1, 1
        if normalise:
            im = utils.data_utils.normalise(im, -1, 1)

        return im

    def load_raw_unlabelled_data(self, include_labelled=True, normalise=True, value_crop=True):
        self.log.debug('Loading unlabelled acdc data from original location')
        images, index = [], []

        for patient_i in self.volumes:
            patient = 'patient%03d' % patient_i
            self.log.debug('Loading patient %s' % patient)
            patient_folder = os.path.join(self.data_folder, patient)

            im_name = patient + '_4d.nii.gz'
            im = self.process_raw_image(im_name, patient_folder, value_crop, normalise)

            frames = range(im.shape[-1])
            if not include_labelled:
                gt = [f for f in os.listdir(patient_folder) if 'gt' in f and not f.startswith('._')]
                gt_ims = [f.replace('_gt', '') for f in gt if not f.startswith('._')]

                exclude_frames = [int(gt_im.split('.')[0].split('frame')[1]) for gt_im in gt_ims]
                frames = [f for f in range(im.shape[-1]) if f not in exclude_frames]

            for frame in frames:
                im_res = im[:, :, :, frame]
                if im_res.sum() == 0:
                    print('Skipping blank images')
                    continue
                images.append(im_res)
                index += [patient_i] * im_res.shape[-1]

        images = [np.expand_dims(np.moveaxis(im, 2, 0), axis=3) for im in images]
        zeros = [np.zeros(im.shape) for im in images]
        images_cropped, _ = utils.data_utils.crop_same(images, zeros, (224, 224))
        images_cropped = np.concatenate(images_cropped, axis=0)[..., 0]
        index = np.array(index)

        return images_cropped.transpose(0,3,1,2), index


    def load_per_slice_raw_labelled_data(self, slice_num, normalise=True, value_crop=True):
        self.log.debug('Loading acdc data from original location')
        images, masks_lv, masks_rv, masks_myo, index, labels, phase = [], [], [], [], [], [], []

        for patient_i in self.volumes:
            patient = 'patient%03d' % patient_i
            patient_folder = os.path.join(self.data_folder, patient)

            #retrieve pathology label from patient's Info.cfg file
            cfg = [f for f in os.listdir(patient_folder) if 'cfg' in f and f.startswith('Info')]
            label_file = open(os.path.join(patient_folder, cfg[0]), mode = 'r')
            lines = label_file.readlines()
            label_file.close()
            label_char = ''
            for line in lines:
                line = line.split(' ')
                if line[0] == 'Group:':
                    label_char = line[1]

            if label_char == 'NOR\n':
                label = 0
            elif label_char == 'MINF\n':
                label = 1
            elif label_char == 'DCM\n':
                label = 2
            elif label_char == 'HCM\n':
                label = 3
            else: #RV
                label = 4

            gt = [f for f in os.listdir(patient_folder) if 'gt' in f and f.startswith(patient + '_frame')]
            ims = [f.replace('_gt', '') for f in gt]

            for i in range(len(ims)):
                im = self.process_raw_image(ims[i], patient_folder, value_crop, normalise)
                im = np.expand_dims(im, axis=-1)

                m = self.resample_raw_image(gt[i], patient_folder, binary=True)
                m = np.expand_dims(m, axis=-1)

                # sample the predefined slice
                im = np.transpose(im, (2,0,1,3))
                im = np.transpose(np.expand_dims(im[slice_num], axis=0), (1,2,0,3))
                m = np.transpose(m, (2,0,1,3))
                m = np.transpose(np.expand_dims(m[slice_num], axis=0), (1,2,0,3))

                images.append(im)

                # convert 3-dim mask array to 3 binary mask arrays for lv, rv, myo
                m_lv = m.copy()
                m_lv[m != 3] = 0
                m_lv[m == 3] = 1
                masks_lv.append(m_lv)

                m_rv = m.copy()
                m_rv[m != 1] = 0
                m_rv[m == 1] = 1
                masks_rv.append(m_rv)

                m_myo = m.copy()
                m_myo[m != 2] = 0
                m_myo[m == 2] = 1
                masks_myo.append(m_myo)

                index += [patient_i] * im.shape[2]

                # systole/diastole phase
                if i == 0:
                    phase.append(0)
                else:
                    phase.append(1)

                labels.append(label)

        assert len(images) == len(masks_myo)

        # move slice axis to the first position
        images    = [np.moveaxis(im, 2, 0) for im in images]
        masks_lv  = [np.moveaxis(m, 2, 0) for m in masks_lv]
        masks_rv  = [np.moveaxis(m, 2, 0) for m in masks_rv]
        masks_myo = [np.moveaxis(m, 2, 0) for m in masks_myo]

        # crop images and masks to the same pixel dimensions and concatenate all data
        images_cropped, masks_lv_cropped = utils.data_utils.crop_same(images, masks_lv, (224, 224))
        _, masks_rv_cropped = utils.data_utils.crop_same(images, masks_rv, (224, 224))
        _, masks_myo_cropped = utils.data_utils.crop_same(images, masks_myo, (224, 224))

        for i in range (len(images_cropped)):
            images_cropped[i] = utils.data_utils.normalise(images_cropped[i], -1, 1)

        images_cropped    = np.concatenate(images_cropped, axis=0)
        masks_lv_cropped  = np.concatenate(masks_lv_cropped, axis=0)
        masks_rv_cropped  = np.concatenate(masks_rv_cropped, axis=0)
        masks_myo_cropped = np.concatenate(masks_myo_cropped, axis=0)
        index             = np.array(index)
        labels            = np.array(labels)
        labels            = np.expand_dims(labels, axis=1)
        phase             = np.array(phase)
        phase             = np.expand_dims(phase, axis=1)

        self.log.debug(str(images[0].shape) + ', ' + str(masks_lv[0].shape))

        return images_cropped.transpose(0,3,1,2), masks_lv_cropped.transpose(0,3,1,2), masks_rv_cropped.transpose(0,3,1,2), masks_myo_cropped.transpose(0,3,1,2), index, labels, phase


    def load_per_slice_raw_unlabelled_data(self, slice_num, include_labelled=True, normalise=True, value_crop=True):
        self.log.debug('Loading unlabelled acdc data from original location')
        images, index, labels, phase = [], [], [], []

        for patient_i in self.volumes:
            patient = 'patient%03d' % patient_i
            self.log.debug('Loading patient %s' % patient)
            patient_folder = os.path.join(self.data_folder, patient)

            #retrieve pathology label from patient's Info.cfg file
            cfg = [f for f in os.listdir(patient_folder) if 'cfg' in f and f.startswith('Info')]
            label_file = open(os.path.join(patient_folder, cfg[0]), mode = 'r')
            lines = label_file.readlines()
            label_file.close()
            label_char = ''
            for line in lines:
                line = line.split(' ')
                if line[0] == 'Group:':
                    label_char = line[1]

            if label_char == 'NOR\n':
                label = 0
            elif label_char == 'MINF\n':
                label = 1
            elif label_char == 'DCM\n':
                label = 2
            elif label_char == 'HCM\n':
                label = 3
            else: #RV
                label = 4

            im_name = patient + '_4d.nii.gz'
            im = self.process_raw_image(im_name, patient_folder, value_crop, normalise, slice_num)

            frames = range(im.shape[-1])
            if not include_labelled:
                gt = [f for f in os.listdir(patient_folder) if 'gt' in f and not f.startswith('._')]
                gt_ims = [f.replace('_gt', '') for f in gt if not f.startswith('._')]

                exclude_frames = [int(gt_im.split('.')[0].split('frame')[1]) for gt_im in gt_ims]
                frames = [f for f in range(im.shape[-1]) if f not in exclude_frames and f > exclude_frames[0] and f < exclude_frames[1]]
            else:
                gt = [f for f in os.listdir(patient_folder) if 'gt' in f and not f.startswith('._')]
                gt_ims = [f.replace('_gt', '') for f in gt if not f.startswith('._')]
                exclude_frames = [int(gt_im.split('.')[0].split('frame')[1]) for gt_im in gt_ims]
                frames = [f for f in range(im.shape[-1]) if (f > exclude_frames[0] and f < exclude_frames[1]) or f == exclude_frames[0] or f == exclude_frames[1]]

            im = np.expand_dims(im, axis=2)
            temp = np.zeros(len(frames))
            for i in range(len(frames)):
                temp[i] = frames[i]
            phase_weight_scaled = np.zeros(len(frames))
            for i in range (temp.shape[0]):
                phase_weight_scaled[i] = (temp[i] - np.min(temp)) / (np.max(temp) - np.min(temp))

            parallel_phase_cnt = 0
            for frame in frames:
                im_res = im[:, :, :, frame]
                if im_res.sum() == 0:
                    print('Skipping blank images')
                    continue
                im_res = np.expand_dims(im_res, axis=2)
                images.append(im_res)
                index += [patient_i] * im_res.shape[-1]
                labels.append(label)
                phase.append(phase_weight_scaled[parallel_phase_cnt])
                parallel_phase_cnt += 1

        images            = [np.moveaxis(im, 2, 0) for im in images]
        zeros             = [np.zeros(im.shape) for im in images]
        images_cropped, _ = utils.data_utils.crop_same(images, zeros, (224, 224))
        images_cropped    = np.concatenate(images_cropped, axis=0)#[..., 0]
        index             = np.array(index)
        labels            = np.array(labels)
        labels            = np.expand_dims(labels, axis=1)
        phase             = np.array(phase)
        phase             = np.expand_dims(phase, axis=1)

        return images_cropped.transpose(0,3,1,2), index, labels, phase