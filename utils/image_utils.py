import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def generate_noise_patch(img):
    blurred_img = cv2.blur(img.numpy(), (10,10))
    noise_code = np.random.normal(0, 1, (img.shape[0], img.shape[1])).astype(np.float32)
    masked_blurred = blurred_img * noise_code
    return torch.from_numpy(masked_blurred).type(dtype=torch.float32)

def generate_source_mask(img, kernel=10):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    masked = cv2.erode(img.numpy(), element, iterations = 1)
    masked = cv2.dilate(masked, element, iterations = 1)
    masked = cv2.erode(masked, element)
    source_mask = cv2.blur(masked, (kernel,kernel))
    source_mask[source_mask > 0] = 1
    return torch.from_numpy(source_mask).type(dtype=torch.float32)

def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def spatial_registration(img1, img2):
    img1 = generate_source_mask(img1, 1)
    img2 = generate_source_mask(img2, 1)
    img1_numpy = img1.numpy()
    img2_numpy = img2.numpy()
    # plt.imshow(img1_numpy*255, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.imshow(img2_numpy*255, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    cnts1, _ = cv2.findContours(np.array(img1_numpy * 255, dtype = np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(np.array(img2_numpy * 255, dtype = np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dX = 0
    dY = 0
    if len(cnts1) > 0 and len(cnts2) > 0:
        M = cv2.moments(cnts1[0])
        check_zero = 0
        for key in M.keys():
            check_zero += M[key]
        if check_zero > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            M2 = cv2.moments(cnts2[0])
            check_zero = 0
            for key in M2.keys():
                check_zero += M2[key]
            if check_zero > 0:
                cX2 = int(M2["m10"] / M2["m00"])
                cY2 = int(M2["m01"] / M2["m00"])
                dX = cX - cX2
                dY = cY - cY2
    new_img2 = shift_image(img2_numpy, dX, dY)
    if np.sum(new_img2) > 0:
        return torch.from_numpy(new_img2).type(dtype=torch.float32)
    else:
        return torch.from_numpy(img1_numpy).type(dtype=torch.float32)

def dilate_content_factor(tensor, kernel_size=0):
    img = tensor.squeeze().numpy()
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    masked = cv2.erode(img, element, iterations = 1)
    masked = cv2.dilate(masked, element, iterations = 1)
    masked = cv2.erode(masked, element)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    processing_img = cv2.dilate(masked,kernel,iterations=1)
    return torch.from_numpy(processing_img).type(dtype=torch.float32)

def erode_content_factor(tensor, kernel_size=0):
    img = tensor.squeeze().numpy()
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    masked = cv2.erode(img, element, iterations = 1)
    masked = cv2.dilate(masked, element, iterations = 1)
    masked = cv2.erode(masked, element)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    processing_img = cv2.erode(masked,kernel,iterations=1)
    return torch.from_numpy(processing_img).type(dtype=torch.float32)

def save_gray_images(y, folder, idx):
    plt.imsave(folder + '/sample' + str(idx) + '.jpg', y[0], cmap='gray')

def save_content_factors(folder, x, x_noise, idx):
    img1 = x[0]
    for j in range(x.shape[0]):
        if j > 0:
            img1 = np.concatenate((img1, x[j]), axis=1)
    img2 = x_noise[0]
    for j in range(x_noise.shape[0]):
        if j > 0:
            img2 = np.concatenate((img2, x_noise[j]), axis=1)
    plt.imsave(folder + '/' + str(idx) + '_mixed_factors_' +'.jpg', img1, cmap='gray')
    plt.imsave(folder + '/' + str(idx) + '_noisy_factors_' +'.jpg', img2, cmap='gray')

def save_multi_image(folder, gen, s1, s2, a_out, augmented_a_out, pred_class, s1_class, s2_class, idx):
    x = gen[0]
    y = a_out[0,:,:]
    x = np.concatenate((x, s1[0], s2[0]), axis=1)
    y = np.concatenate((y,\
                    a_out[1,:,:], a_out[2,:,:], a_out[3,:,:], a_out[4,:,:], a_out[5,:,:], a_out[6,:,:], a_out[7,:,:],\
                    augmented_a_out[0,:,:], augmented_a_out[1,:,:], augmented_a_out[2,:,:], augmented_a_out[3,:,:], augmented_a_out[4,:,:], augmented_a_out[5,:,:], augmented_a_out[6,:,:], augmented_a_out[7,:,:]), axis=1)
    plt.imsave(folder + '/' + str(idx) + s1_class + '_mixed_with_' + s2_class + '_generated_' + pred_class + '_images_' + '.jpg', x, cmap='gray')
    plt.imsave(folder + '/' + str(idx) + s1_class + '_mixed_with_' + s2_class + '_generated_' + pred_class + '_factors_' +'.jpg', y, cmap='gray')
