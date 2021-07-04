from utils.image_utils import generate_noise_patch, generate_source_mask, spatial_registration

def check_label_mix(
    rv_idx, 
    lv_idx, 
    myo_idx,
    augmented_a_out,
    a_out,
    a_out_2,
    aggregated_noise_mask,
    aggregated_source_mask,
    zero_mask,
    mixed,
    b_labels,
    b_labels2
    ):
    swapped_factors = 0
    for i in range(augmented_a_out.shape[0]):
        if b_labels[i] == 0: #NOR
            if b_labels2[i] == 0: #NOR
                augmented_a_out[i,rv_idx] = spatial_registration(a_out[i,rv_idx], a_out_2[i,rv_idx])
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,rv_idx] + a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 3
            elif b_labels2[i] == 1: #MINF
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 2
            elif b_labels2[i] == 2: #DCM
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 2
            elif b_labels2[i] == 3: #HCM
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 2
            else: #ARV
                augmented_a_out[i,rv_idx] = spatial_registration(a_out[i,rv_idx], a_out_2[i,rv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,rv_idx])
                swapped_factors = 1
            mixed[i] = 1
        else:
            if b_labels2[i] == 0: #NOR
                augmented_a_out[i,rv_idx] = spatial_registration(a_out[i,rv_idx], a_out_2[i,rv_idx])
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,rv_idx] + a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 3
                mixed[i] = 1
            elif b_labels2[i] == 1 and b_labels[i] == 1: #MINF
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 2
                mixed[i] = 1
            elif b_labels2[i] == 2 and b_labels[i] == 2: #DCM
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 2
                mixed[i] = 1
            elif b_labels2[i] == 3 and b_labels[i] == 3: #HCM
                augmented_a_out[i,myo_idx] = spatial_registration(a_out[i,myo_idx], a_out_2[i,myo_idx])
                augmented_a_out[i,lv_idx] = spatial_registration(a_out[i,lv_idx], a_out_2[i,lv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,myo_idx] + a_out[i,lv_idx])
                swapped_factors = 2
                mixed[i] = 1
            elif b_labels2[i] == 4 and b_labels[i] == 4: #ARV
                augmented_a_out[i,rv_idx] = spatial_registration(a_out[i,rv_idx], a_out_2[i,rv_idx])
                aggregated_noise_mask[i] = generate_noise_patch(a_out[i,rv_idx])
                swapped_factors = 1
                mixed[i] = 1
            else:
                aggregated_noise_mask[i] = zero_mask
                mixed[i] = 0
        aggregated_source_mask[i] = generate_source_mask(a_out[i,rv_idx] + a_out[i,myo_idx] + a_out[i,lv_idx])
    return augmented_a_out, aggregated_noise_mask, aggregated_source_mask, mixed, swapped_factors

def get_original_label(idx):
    if idx == 0:
        label = 'NOR'
    elif idx == 1:
        label = 'MINF'
    elif idx == 2:
        label = 'DCM'
    elif idx == 3:
        label = 'HCM'
    else:
        label = 'RV'

    return label