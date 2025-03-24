import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import gaussian_diffusion as gf
import CMDT as tr_model_wlm
from Dataset_test import AudioPoseDataset
from pad_sequence_inference_wlm import collate_fn_padd
from spaced_diffusion import SpacedDiffusion, space_timesteps
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def save_result(dmm_paths, generated_poses, nb_framesS, opt):
    for dmm_path, generated_pose, nb_frames in zip(dmm_paths, generated_poses, nb_framesS):
        save_path = os.path.dirname(dmm_path)

        dmm = loadmat(dmm_path)
        coeff = np.repeat(np.expand_dims(dmm['coeff'][0,:], 0), nb_frames, axis=0)
        transform = np.repeat(np.expand_dims(dmm['transform_params'][:nb_frames,:][0,:], 0), nb_frames, axis=0)
        coeff[:nb_frames, 80:144] = generated_pose[:nb_frames, :64]
        coeff[:nb_frames, 224:227] = generated_pose[:nb_frames, 64:67]
        coeff[:nb_frames, 254:257] = generated_pose[:nb_frames, 67:]
        
        savemat(os.path.join(save_path, 'generated.mat'), {'coeff': coeff, 'transform_params': transform})

def Infer(dataloader, model_face, model_lips, model_head, diffusion_face, diffusion_lips, diffusion_head, opt):

    model_face.load_state_dict(torch.load(opt.load_face_weights), strict=True)
    model_face.eval()
    model_lips.load_state_dict(torch.load(opt.load_lips_weights), strict=True)
    model_lips.eval()
    model_head.load_state_dict(torch.load(opt.load_head_weights), strict=True)
    model_head.eval()
    m_idx = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]
    upper_face3d_indices = [6, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                            57, 58, 59, 60, 61, 62, 63]
    lower_face3d_indices = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]
    pose_face3d_indices = [64, 65, 66, 67, 68, 69]
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        pose, wlm, dmm_path, nb_gens, nb_frames = batch
        pose_0 = torch.cat((pose[:, 0, 80:144], pose[:, 0, 224:227], pose[:, 0, 254:257]), axis=-1)
        m_0 = pose_0[:, m_idx]
        generated_seqs = []
        nb_gen = max(nb_gens)

        for i in range(nb_gen):
            if i !=nb_gen-1:
                wlm_cur = wlm[:, i * opt.audio_len:(i + 1) * opt.audio_len,:]

            else:
                wlm_cur = wlm[:, i * opt.audio_len:,:]
            B, T = wlm_cur.shape[:2]
            cur_len = torch.LongTensor([min(T, opt.audio_len) for _ in range(B)])

            pose_0_f=pose_0[:,upper_face3d_indices]
            xf_proj_pose_f, xf_out_pose_f = model_face.encode_pose(pose_0_f,T)
            xf_proj_mfcc_f, xf_out_mfcc_f = model_face.encode_mfcc(wlm_cur)
            head_poses_face = diffusion_face.p_sample_loop(
                model_face,
                (B, T, opt.pose_size-19),
                clip_denoised=False,
                progress=False,
                model_kwargs={'xf_proj_pose':xf_proj_pose_f,'xf_out_pose':xf_out_pose_f,'xf_proj_wlm':xf_proj_mfcc_f,'xf_out_wlm':xf_out_mfcc_f, 'length': cur_len})

            pose_0_l=pose_0[:,lower_face3d_indices]
            xf_proj_pose_l, xf_out_pose_l = model_lips.encode_pose(pose_0_l,T)
            xf_proj_mfcc_l, xf_out_mfcc_l = model_lips.encode_mfcc(wlm_cur)

            head_poses_lips = diffusion_lips.p_sample_loop(
                model_lips,
                (B, T, 13),
                clip_denoised=False,
                progress=False,
                model_kwargs={'xf_proj_pose':xf_proj_pose_l,'xf_out_pose':xf_out_pose_l,'xf_proj_wlm':xf_proj_mfcc_l,'xf_out_wlm':xf_out_mfcc_l, 'length': cur_len})

            pose_0_h=pose_0[:,pose_face3d_indices]
            xf_proj_pose_h, xf_out_pose_h = model_head.encode_pose(pose_0_h,T)
            xf_proj_mfcc_h, xf_out_mfcc_h = model_head.encode_mfcc(wlm_cur)
            head_poses_head = diffusion_head.p_sample_loop(
                model_head,
                (B, T, 6),
                clip_denoised=False,
                progress=False,
                model_kwargs={'xf_proj_pose':xf_proj_pose_h,'xf_out_pose':xf_out_pose_h,'xf_proj_wlm':xf_proj_mfcc_h,'xf_out_wlm':xf_out_mfcc_h, 'length': cur_len})

            head_poses = torch.unsqueeze(pose_0,1)#torch.zeros(B,T,70).cuda()
            head_poses = head_poses.repeat(1,T,1)
            head_poses[:,:,upper_face3d_indices]=head_poses_face
            head_poses[:, :, lower_face3d_indices] = head_poses_lips
            head_poses[:, :, pose_face3d_indices] = head_poses_head

            pose_0=head_poses[:,-1,:]
            pose_0[:,m_idx]=m_0
            torch.cuda.empty_cache()
            generated_seqs.append(head_poses.cpu().detach().numpy())
        
        head_poses_np = np.concatenate(generated_seqs, axis=1)
        save_result(dmm_path, head_poses_np, nb_frames, opt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-audio_len', type=int, default=100)
    parser.add_argument('-pose_size', type=int, default=70)
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-num_worker', type=int, default=0)
    parser.add_argument('-load_face_weights', required=True)
    parser.add_argument('-load_lips_weights', required=True)
    parser.add_argument('-load_head_weights', required=True)
    parser.add_argument('-diffusion_steps', type=int, default=500)
    parser.add_argument('-root_dir', type=str, default='output_multi')

    opt = parser.parse_args()

    model_face = tr_model_wlm.MotionTransformer(opt.pose_size-19, num_frames=opt.audio_len)
    model_face = model_face.cuda()
    model_lips = tr_model_wlm.MotionTransformer(13, num_frames=opt.audio_len)
    model_lips = model_lips.cuda()
    model_head = tr_model_wlm.MotionTransformer(6, num_frames=opt.audio_len)
    model_head = model_head.cuda()

    pose_dataset = AudioPoseDataset(root_dir=opt.root_dir,max_frames=opt.audio_len)
    dataloader = DataLoader(pose_dataset, batch_size=opt.batch_size, shuffle=False, 
                            collate_fn=collate_fn_padd, num_workers=opt.num_worker)

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

    Infer(dataloader, model_face, model_lips, model_head, diffusion_face, diffusion_lips, diffusion_head, opt)

if __name__ == "__main__":
    main()
