import os
import argparse
import torch
import numpy as np
from scipy.io import loadmat, savemat
import math

import gaussian_diffusion as gf
import CMDT as tr_model_wlm
from spaced_diffusion import SpacedDiffusion, space_timesteps
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_data(max_len, dmm_path, wlm_path):
    wlm = np.load(wlm_path)
    wlm = np.reshape(wlm, (wlm.shape[0], wlm.shape[1] * wlm.shape[2]))

    dmm = loadmat(dmm_path)
    dmm = dmm['coeff']
    nb_frames = wlm.shape[0]

    nb_gen = math.ceil(nb_frames / max_len)
    pose = np.transpose(dmm)

    pose_tensor = torch.tensor(pose, dtype=torch.float32)
    wlm_tensor = torch.tensor(wlm, dtype=torch.float32)

    lengths_audio = wlm_tensor.shape[-1]
    lengths_audio_t = torch.tensor(lengths_audio)
    lengths_pose = pose_tensor.shape[-1]
    lengths_pose_t = torch.tensor(lengths_pose)

    lengths_audio = [lengths_audio, lengths_audio, lengths_audio]

    audio_pad = torch.nn.functional.pad(wlm_tensor, (0, torch.max(lengths_audio_t) - wlm_tensor.shape[-1]))
    pose_pad = torch.nn.functional.pad(pose_tensor, (0, torch.max(lengths_pose_t) - pose_tensor.shape[-1]))

    audio_pad = torch.stack([audio_pad, audio_pad, audio_pad]).to('cuda')
    pose_pad = torch.stack([pose_pad, pose_pad, pose_pad])
    pose_pad = pose_pad.permute((0, 2, 1)).to('cuda')

    return audio_pad, pose_pad, lengths_audio, dmm_path, nb_gen

def save_result(dmm_path, generated_pose):
    save_path = "output_single/"+dmm_path.split('/')[-1]
    dmm = loadmat(dmm_path)
    coeff = dmm['coeff']
    nb_frames = generated_pose.shape[1]
    coeff = np.repeat(np.expand_dims(coeff[0, :], 0), nb_frames, axis=0)
    coeff = coeff[:nb_frames, :]
    coeff[:, 80:144] = generated_pose[0, :nb_frames, :64]
    coeff[:, 224:227] = generated_pose[0, :nb_frames, 64:67]
    coeff[:, 254:257] = generated_pose[0, :nb_frames, 67:]
    transform = dmm['transform_params']
    transform = transform[:nb_frames, :]
    transform = np.repeat(np.expand_dims(transform[0, :], 0), nb_frames, axis=0)

    savemat(
        save_path,
        {'coeff': coeff, 'transform_params': transform}
    )

def Infer(data, model_face, model_lips, model_head, diffusion_face, diffusion_lips, diffusion_head, opt):
    model_face.load_state_dict(torch.load(opt.face_weights_path), strict=True)
    model_face.eval()
    model_lips.load_state_dict(torch.load(opt.lips_weights_path), strict=True)
    model_lips.eval()
    model_head.load_state_dict(torch.load(opt.head_weights_path), strict=True)
    model_head.eval()

    wlm, pose, len_pose, dmm_path, nb_gen = data
    pose_0 = torch.cat((pose[:, 0, 80:144], pose[:, 0, 224:227], pose[:, 0, 254:257]), axis=-1)
    m_idx = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]
    upper_face3d_indices = [6, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    lower_face3d_indices = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]
    pose_face3d_indices = [64, 65, 66, 67, 68, 69]
    m_0 = pose_0[:, m_idx]
    generated_seqs = []

    for i in range(nb_gen):
        if i != nb_gen - 1:
            wlm_cur = wlm[:, i * opt.audio_len:(i + 1) * opt.audio_len, :]
        else:
            wlm_cur = wlm[:, i * opt.audio_len:, :]
        B, T = wlm_cur.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in len_pose])

        pose_0_f = pose_0[:, upper_face3d_indices]
        xf_proj_pose_f, xf_out_pose_f = model_face.encode_pose(pose_0_f, T)
        xf_proj_mfcc_f, xf_out_mfcc_f = model_face.encode_mfcc(wlm_cur)
        head_poses_face = diffusion_face.ddim_sample_loop(
            model_face,
            (B, T, opt.pose_size - 19),
            clip_denoised=False,
            progress=False,
            model_kwargs={'xf_proj_pose': xf_proj_pose_f, 'xf_out_pose': xf_out_pose_f, 'xf_proj_wlm': xf_proj_mfcc_f, 'xf_out_wlm': xf_out_mfcc_f, 'length': cur_len}
        )

        pose_0_l = pose_0[:, lower_face3d_indices]
        xf_proj_pose_l, xf_out_pose_l = model_lips.encode_pose(pose_0_l, T)
        xf_proj_mfcc_l, xf_out_mfcc_l = model_lips.encode_mfcc(wlm_cur)

        head_poses_lips = diffusion_lips.ddim_sample_loop(
            model_lips,
            (B, T, 13),
            clip_denoised=False,
            progress=False,
            model_kwargs={'xf_proj_pose': xf_proj_pose_l, 'xf_out_pose': xf_out_pose_l, 'xf_proj_wlm': xf_proj_mfcc_l, 'xf_out_wlm': xf_out_mfcc_l, 'length': cur_len}
        )

        pose_0_h = pose_0[:, pose_face3d_indices]

        xf_proj_pose_h, xf_out_pose_h = model_head.encode_pose(pose_0_h, T)
        xf_proj_mfcc_h, xf_out_mfcc_h = model_head.encode_mfcc(wlm_cur)
        head_poses_head = diffusion_head.ddim_sample_loop(
            model_head,
            (B, T, 6),
            clip_denoised=False,
            progress=False,
            model_kwargs={'xf_proj_pose': xf_proj_pose_h, 'xf_out_pose': xf_out_pose_h, 'xf_proj_wlm': xf_proj_mfcc_h, 'xf_out_wlm': xf_out_mfcc_h, 'length': cur_len}
        )

        head_poses = torch.unsqueeze(pose_0, 1)
        head_poses = head_poses.repeat(1, T, 1)
        head_poses[:, :, upper_face3d_indices] = head_poses_face
        head_poses[:, :, lower_face3d_indices] = head_poses_lips
        head_poses[:, :, pose_face3d_indices] = head_poses_head

        pose_0 = head_poses[:, -1, :]
        pose_0[:, m_idx] = m_0
        torch.cuda.empty_cache()
        generated_seqs.append(head_poses.cpu().detach().numpy())
    head_poses_np = np.concatenate(generated_seqs, axis=1)
    save_result(dmm_path, head_poses_np)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-audio_len', type=int, default=100)
    parser.add_argument('-pose_size', type=int, default=70)
    parser.add_argument('-face_weights_path', required=True, help='Path to face model weights')
    parser.add_argument('-lips_weights_path', required=True, help='Path to lips model weights')
    parser.add_argument('-head_weights_path', required=True, help='Path to head model weights')
    parser.add_argument('-diffusion_steps', type=int, default=500)
    parser.add_argument('-dmm_path', required=True, help='Path to 3dmm.mat file')
    parser.add_argument('-wlm_path', required=True, help='Path to wlm.npy file')

    opt = parser.parse_args()

    model_face = tr_model_wlm.MotionTransformer(opt.pose_size - 19, num_frames=opt.audio_len)
    model_face = model_face.cuda()
    model_lips = tr_model_wlm.MotionTransformer(13, num_frames=opt.audio_len)
    model_lips = model_lips.cuda()
    model_head = tr_model_wlm.MotionTransformer(6, num_frames=opt.audio_len)
    model_head = model_head.cuda()

    data = get_data(opt.audio_len, opt.dmm_path, opt.wlm_path)

    beta_scheduler = 'linear'
    betas = gf.get_named_beta_schedule(beta_scheduler, opt.diffusion_steps)
    diffusion_face = SpacedDiffusion(
        use_timesteps=space_timesteps(opt.diffusion_steps, "ddim25"),
        betas=betas,
        model_mean_type=gf.ModelMeanType.START_X,
        model_var_type=gf.ModelVarType.FIXED_SMALL,
        loss_type=gf.LossType.MSE,
        rescale_timesteps=False,
    )
    diffusion_lips = SpacedDiffusion(
        use_timesteps=space_timesteps(opt.diffusion_steps, "ddim25"),
        betas=betas,
        model_mean_type=gf.ModelMeanType.START_X,
        model_var_type=gf.ModelVarType.FIXED_SMALL,
        loss_type=gf.LossType.MSE,
        rescale_timesteps=False,
    )
    diffusion_head = SpacedDiffusion(
        use_timesteps=space_timesteps(opt.diffusion_steps, "ddim25"),
        betas=betas,
        model_mean_type=gf.ModelMeanType.START_X,
        model_var_type=gf.ModelVarType.FIXED_SMALL,
        loss_type=gf.LossType.MSE,
        rescale_timesteps=False,
    )

    Infer(data, model_face, model_lips, model_head, diffusion_face, diffusion_lips, diffusion_head, opt)

if __name__ == "__main__":
    main()

# python Inference_vox_3_wlm.py -face_weights_path /path/to/face_weights.pth -lips_weights_path /path/to/lips_weights.pth -head_weights_path /path/to/head_weights.pth -dmm_path /path/to/3dmm.mat -wlm_path /path/to/wlm.npy -save_folder /path/to/save/folder -save_name result_subfolder
