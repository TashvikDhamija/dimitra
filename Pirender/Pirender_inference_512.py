import argparse
import cv2
import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat

from core.utils import obtain_seq_index
from configs.default import get_cfg_defaults
from generators.face_model import FaceGenerator
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@torch.no_grad()
def render_video(net_G, src_img_path, exp_path, wav_path, output_path, silent=False, semantic_radius=13, fps=30, split_size=64):
    target_exp_seq = np.load(exp_path)
    src_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])(Image.fromarray(cv2.cvtColor(cv2.imread(src_img_path), cv2.COLOR_BGR2RGB)))

    target_win_exps = [torch.tensor(target_exp_seq[obtain_seq_index(i, len(target_exp_seq), semantic_radius)]).permute(1, 0) 
                       for i in range(len(target_exp_seq))]
    target_exp_concat = torch.stack(target_win_exps, dim=0)
    
    output_imgs = []
    for win_exp in torch.split(target_exp_concat, split_size, dim=0):
        win_exp = win_exp.cuda()
        cur_src_img = src_img.expand(win_exp.shape[0], -1, -1, -1).cuda()
        output_imgs.append(net_G(cur_src_img, win_exp)["fake_image"].cpu().clamp_(-1, 1))

    transformed_imgs = ((torch.cat(output_imgs, 0) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)
    if silent:
        torchvision.io.write_video(output_path, transformed_imgs.cpu(), fps)
    else:
        silent_video_path = os.path.join(os.path.dirname(output_path), "silent.mp4")
        torchvision.io.write_video(silent_video_path, transformed_imgs.cpu(), fps)
        os.system(f"ffmpeg -loglevel error -y -i {silent_video_path} -i {wav_path} -shortest {output_path}")
        try:
            os.remove(silent_video_path)
        except:
            print(f'Failed to remove {silent_video_path}')

@torch.no_grad()
def get_netG(checkpoint_path):
    with open("Pirender/configs/renderer_conf_512.yaml", "r") as f:
        renderer_config = yaml.safe_load(f)

    renderer = FaceGenerator(**renderer_config).to(torch.cuda.current_device())
    renderer.load_state_dict(torch.load(checkpoint_path, map_location='cuda')["net_G_ema"], strict=False)
    return renderer.eval()

@torch.no_grad()
def generate_expression_params(pose_path, output_path):
    data = loadmat(pose_path)
    data_face = data["coeff"]
    data_exp = data_face[:, 80:144]
    angles = data_face[:, 224:227]
    translations = data_face[:, 254:257]
    # This loop prevent head rotation too far from the original pose to reduce artifacts. It can be commented to obtain the real generated head poses
    for j in range(angles.shape[1]):
        max_angles = angles[0, j] +0.2
        min_angles = angles[0, j] -0.2
        angles[angles[:, j] > max_angles, j] = max_angles
        angles[angles[:, j] < min_angles, j] = min_angles

    if int(data["transform_params"][0,0])!=512:
        ratio = 512/data["transform_params"][0,0]
        ratio_np = np.asarray([1/ratio, ratio, ratio],dtype=np.float32)
        crop = data["transform_params"][:, -3:]*ratio_np
    else:
        crop = data["transform_params"][:, -3:]
    pose_params = np.concatenate((angles, translations, crop), axis=1)
    gen_exp_pose = np.concatenate((data_exp, pose_params), axis=1)
    np.save(output_path, gen_exp_pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for demo")
    parser.add_argument("--renderer_checkpoint", type=str, default="Pirender/checkpoints/renderer_checkpoint_512.pt")
    parser.add_argument("--pose_path", type=str, required=True)
    parser.add_argument("--src_img_path", type=str, required=True)
    parser.add_argument("--wav_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.freeze()

    with torch.no_grad():
        exp_param_path = f"{args.output_path[:-4]}.npy"
        generate_expression_params(args.pose_path, exp_param_path)

        image_renderer = get_netG(args.renderer_checkpoint)
        render_video(image_renderer, args.src_img_path, exp_param_path, args.wav_path, args.output_path, split_size=4, fps=25)
