# python inference.py -i data/aligned -o results/RF++/aligned -v RestoreFormer++ -s 2 --aligned --save
# python inference.py -i data/raw -o results/RF++/raw -v RestoreFormer++ -s 2 --save
# python inference.py -i data/aligned -o results/RF/aligned -v RestoreFormer -s 2 --aligned --save
# python inference.py -i data/raw -o results/RF/raw -v RestoreFormer -s 2 --save

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from basicsr.utils import imwrite

from RestoreFormer.download_util import load_file_from_url
from RestoreFormer.RestoreFormer import RestoreFormer
from tqdm import tqdm
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Inference demo for RestoreFormer (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='naming to separate processes')
    parser.add_argument('--videos_path', type=str, default='output_multi', help='video naming to look for')

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='RestoreFormer++/extracted_images',
        help='Input image or folder. Default: data/test')
    parser.add_argument('-o', '--output', type=str, default='RestoreFormer++/restored_images', help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-v', '--version', type=str, default='RestoreFormer++', help='RestoreFormer model version. Option: RestoreFormer| RestoreFormer++')
    parser.add_argument(
        '-s', '--upscale', type=int, default=1, help='The final upsampling scale of the image. Default: 2')

    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')

    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument("--single", action='store_true')


    args = parser.parse_args()

    args.input=args.input+'_'+args.name
    args.output=args.output+'_'+args.name


    # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            realesrgan_model_path = os.path.join('experiments/weights', 'RealESRGAN_x2plus.pth')
            if not os.path.isfile(realesrgan_model_path):
                # download pre-trained models from url
                realesrgan_model_path = load_file_from_url(
                    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model_dir=os.path.join(ROOT_DIR, 'experiments/weights'), 
                    progress=True, 
                    file_name=None)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=realesrgan_model_path,
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    
    if args.version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt'
    elif args.version == 'RestoreFormer++':
        arch = 'RestoreFormer++'
        channel_multiplier = 2
        model_name = 'RestoreFormer++'
        url = 'https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.ckpt')
    if not os.path.isfile(model_path):
        model_path = os.path.join('experiments/weights', model_name + '.ckpt')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = RestoreFormer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        bg_upsampler=bg_upsampler)

    os.makedirs(args.input, exist_ok=True)

    video_list= glob.glob(args.videos_path+"/**/Dimitra_output.mp4", recursive=True)

    if args.input.endswith('/'):
        args.input = args.input[:-1]

    for video in tqdm(video_list):
        restored_path = video[:-4]+'_cleaned.mp4'
        silent_path = video[:-4]+'_silent.mp4'
        if os.path.isfile(restored_path) and not args.single:
            continue
        if os.path.isfile(args.input+'/00001.png'):
            shutil.rmtree(args.input)
            os.makedirs(args.input)
        if os.path.isfile(args.output + '/restored_imgs/00001.png'):
            shutil.rmtree(args.output+ '/restored_imgs')
            os.makedirs(args.output+ '/restored_imgs')

        #extract images from video:
        os.system(f"ffmpeg -i {video} -vf \"fps=25\" {args.input}/%05d.png  -hide_banner -loglevel panic")


        # ------------------------ input & output ------------------------

        if os.path.isfile(args.input):
            img_list = [args.input]
        else:
            img_list = sorted(glob.glob(os.path.join(args.input, '*')))

        #print(args.input)
        #print(img_list)
        os.makedirs(args.output, exist_ok=True)

        # ------------------------ restore ------------------------
        for img_path in img_list:
            # read image
            img_name = os.path.basename(img_path)
            #print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True)
            
            # save restored img
            if restored_img is not None:
                if args.ext == 'auto':
                    extension = ext[1:]
                else:
                    extension = args.ext

                if args.suffix is not None:
                    save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
                else:
                    save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
                imwrite(restored_img, save_restore_path)

        os.system(f"ffmpeg -y -framerate 25 -i {args.output}/restored_imgs/%05d.png -c:v libx264 -pix_fmt yuv420p {silent_path} -hide_banner -loglevel panic")
        os.system(f"ffmpeg -y -i {silent_path} -i {video} -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy {restored_path} -hide_banner -loglevel panic")
        os.system(f'rm {silent_path}')

    #print(f'Results are in the [{args.output}] folder.')

if __name__ == '__main__':
    main()
