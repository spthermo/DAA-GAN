import argparse

def parse_arguments(args):
    usage_text = (
        "DAA-GAN Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=150, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type= int, default=1, help='Number of inputs per batch')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='noise', help='The name of the model (simple GAN or with Noise Injection).')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    parser.add_argument("--data_path", type=str, default='', help='Path to ACDC dataset')
    parser.add_argument('--load_sdnet_decoder_weights_path', type=str, default='', help= 'Path to pretrained decoder weights')
    parser.add_argument('--load_vgg_weights_path', type=str, default='', help= 'Path to pretrained VGG weights')
    parser.add_argument('--load_weights_path', type=str, default='', help= 'Path to pretrained DAA-GAN weights')
    parser.add_argument('--load_factors_path', type=str, default='factors_npz', help= 'Path to compressed anatomy factors')
    parser.add_argument('--save_images', type=int, default=0, help='Set to 1 to save generated images and mixed factors durint inference')
    #test details
    parser.add_argument('--generated_path', type=str, default='reporting/generated_images', help= 'Path to compressed anatomy factors')
    parser.add_argument('--combo_image_path', type=str, default='reporting/combo_images', help= 'Path to compressed anatomy factors')
    parser.add_argument('--image_path', type=str, default='reporting/input_images', help= 'Path to compressed anatomy factors')
    #model params
    parser.add_argument('--dim', type= int, default=224, help='Input image dimension (dim x dim)')
    parser.add_argument('--ndf', type= int, default=64, help='Number of filters')
    parser.add_argument('--anatomy_out_channels', type= int, default=12, help='Number of anatomy factors')
    parser.add_argument('--z_length', type= int, default=8, help='Number of imaging factors')
    parser.add_argument('--dtype', type=str, default='ls_d', help= 'Type of discriminator module')
    parser.add_argument('--num_classes', type= int, default=5, help='Number of pathology classes')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')
    #visdom params
    parser.add_argument('-d','--disp_iters', type=int, default=1, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--visdom_iters', type=int, default=10, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument('--print_factors', type=int, default=0, help='Set to 1 to visualize the anatomy factors in Visdom')
    parser.add_argument('--batch_percentage', type=float, default=1.0, help='The percentage of the current batch to be visualized.')


    return parser.parse_known_args(args)