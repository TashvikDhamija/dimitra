import torch


def collate_fn_padd(batch):
    '''
    Pads batch of variable length
    '''

    pose = []
    wlm = []
    dmm_pth=[]
    nb_gen=[]
    nb_frames = []
    for k in range(len(batch)):
        pose.append(batch[k][0])
        wlm.append(batch[k][1])
        dmm_pth.append(batch[k][2])
        nb_gen.append(batch[k][3])
        nb_frames.append(batch[k][4])

    ## get sequence lengths
    lengths_pose = [t.shape[-1] for t in pose]
    lengths_pose_t = torch.tensor(lengths_pose)

    ## padd
    pose = [torch.Tensor(t) for t in pose]
    wlm = [torch.Tensor(t).permute((1,0)) for t in wlm]
    lengths_wlm = torch.tensor([t.shape[-1] for t in wlm])
    pose_pad = [torch.nn.functional.pad(seq, (0, torch.max(lengths_pose_t)-seq.shape[-1])) for seq in pose]
    wlm_pad = [torch.nn.functional.pad(seq, (0, torch.max(lengths_wlm) - seq.shape[-1])) for seq in wlm]

    ## compute mask
    pose_pad=torch.stack(pose_pad)
    pose_pad=pose_pad.permute((0,2,1)).to('cuda')

    wlm_pad = torch.stack(wlm_pad)
    wlm_pad=wlm_pad.permute((0,2,1)).to('cuda')

    return pose_pad, wlm_pad, dmm_pth, nb_gen, nb_frames
