import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse

import utils

import torch
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='IMAGE_DERAIN')
parser.add_argument('--path_to_rainy_image', type=str, metavar='str')
parser.add_argument('--in_size', type=int, default=512, metavar='N',
                    help='size of input image during eval')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_512', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_512)')
parser.add_argument('--save_output', action='store_true', default=False,
                    help='to save the output images')
parser.add_argument('--output_dir', type=str, default='./eval_output', metavar='str',
                    help='evaluation output dir (default: ./eval_output)')
args = parser.parse_args()


def load_model(net_G, ckptdir='did_mdn.pt'):

    net_G = cycnet.define_G(
                input_nc=3, output_nc=6, ngf=64, netG=net_G, use_dropout=False, norm='none').to(device)
    print('loading the best checkpoint...')
    checkpoint = torch.load(os.path.join(ckptdir))
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.to(device)
    net_G.eval()

    return net_G



def run_eval(net_G, save_output=True, output_dir='', path_to_rainy_image='', in_size=512):

    print('running evaluation...')

    if save_output:
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

    img = cv2.imread(path_to_rainy_image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # we recommend to use TF.resize since it was also used during trainig
    # You may also try cv2.resize, but it will produce slightly different results
    img = TF.resize(TF.to_pil_image(img), [in_size, in_size])
    img = TF.to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        G_pred = net_G(img.to(device))[:, 0:3, :, :]

    G_pred = np.array(G_pred.cpu().detach())
    G_pred = G_pred[0, :].transpose([1, 2, 0])
    img = np.array(img.cpu().detach())
    img = img[0, :].transpose([1, 2, 0])

    G_pred[G_pred > 1] = 1
    G_pred[G_pred < 0] = 0

    psnr = utils.cpt_rgb_psnr(G_pred, img, PIXEL_MAX=1.0)
    ssim = utils.cpt_rgb_ssim(G_pred, img)

    if save_output:
        fname = path_to_rainy_image.split('/')[-1]
        plt.imsave(os.path.join(output_dir, fname[:-4] + '_input.png'), img)
        plt.imsave(os.path.join(output_dir, fname[:-4] + '_output.png'), G_pred)

    print('Image: %s, psnr: %.4f, ssim: %.4f'
          % (path_to_rainy_image, np.mean(psnr), np.mean(ssim)))



if __name__ == '__main__':

    # args.dataset = 'rain100h'
    # args.dataset = 'rain100l'
    # args.dataset = 'rain800'
    # args.dataset = 'rain800-real'
    # args.dataset = 'did-mdn-test1'
    # args.dataset = 'did-mdn-test2'

    # args.net_G = 'unet_512'
    # args.ckptdir = 'checkpoints'
    # args.save_output = True

    net_G = load_model(args)
    run_eval(args)