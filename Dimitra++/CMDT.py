# Code based on https://github.com/mingyuan-zhang/MotionDiffuse/blob/main/text2motion/models/transformer.py

# from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np

import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()

        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class FFN_in(nn.Module):

    def __init__(self, latent_dim, ffn_dim, out_dim):
        super().__init__()
        self.linear_in = nn.Linear(latent_dim, ffn_dim)
        self.linear_in_2 = zero_module(nn.Linear(ffn_dim, out_dim))
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.linear_in_2(self.activation(self.linear_in(x)))
        return x.permute((0, 2, 1))


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=512,
                 condition_latent_dim=256,
                 text_latent_dim=256,
                 pose_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=512,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.sa_block = LinearTemporalSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block_pose = LinearTemporalCrossAttention(seq_len, latent_dim, pose_latent_dim, num_head, dropout,
                                                          time_embed_dim)
        self.linear_in = nn.Linear(latent_dim * 2, latent_dim)

        self.norm = Norm(latent_dim)
        self.norm_text = Norm(latent_dim)
        self.norm_pose = Norm(latent_dim)
        self.norm_ffn = Norm(latent_dim)

        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf_pose, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x1 = self.ca_block_pose(x, xf_pose, emb)

        x = torch.cat((x, x1), dim=-1)
        x = self.linear_in(x)
        x = self.ffn(self.norm_ffn(x), emb)  # [:,:self.seq_len,:]

        return x


class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 num_audio_layers=4,
                 text_latent_dim=256,
                 audio_latent_dim=512,
                 pose_dim=70,
                 pose_latent_dim=512,
                 text_ff_size=2048,
                 audio_ff_size=2048,
                 text_num_heads=4,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))



        self.mfcc_emb = nn.Linear(9216, 2048)  # MonoPhone+pad
        self.mfcc_emb_2 = nn.Linear(2048, 512)  # MonoPhone+pad

        self.mfcc_pos_enc = PositionalEncoding(512)
        self.mfcc_pre_proj = nn.Linear(audio_latent_dim, audio_latent_dim)
        self.mfcc_proj = nn.Linear(audio_latent_dim, self.time_embed_dim)
        MFCCTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=audio_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=audio_ff_size,
            dropout=dropout,
            activation=activation)
        self.MFCCTransEncoder = nn.TransformerEncoder(
            MFCCTransEncoderLayer,
            num_layers=num_audio_layers)
        self.mfcc_ln = nn.LayerNorm(audio_latent_dim)

        self.pose_pre_proj = nn.Linear(self.input_feats, pose_latent_dim)
        self.pose_proj = nn.Linear(pose_latent_dim, self.time_embed_dim)
        self.pose_ln = nn.LayerNorm(pose_latent_dim)

        # Input Embedding
        # print(self.input_feats)
        self.joint_embed = nn.Linear(int(self.input_feats), self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                LinearTemporalDiffusionTransformerDecoderLayer(
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    condition_latent_dim=audio_latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        # self.ffn_in = FFN_in(512, 512)

    def encode_pose(self, fpose, T):
        # T, B, D
        # print(fpose.shape)
        x = self.pose_pre_proj(fpose)
        xf_out = self.pose_ln(x)
        xf_proj = self.pose_proj(xf_out)
        # B, T, D
        xf_out = xf_out.unsqueeze(1).repeat(1, T, 1)
        return xf_proj, xf_out

    def encode_mfcc(self, audio):
        # T, B, D
        x = audio.permute(1, 0, 2)
        x = self.mfcc_emb_2(self.mfcc_emb(x))
        x = self.mfcc_pos_enc(x)
        x = self.mfcc_pre_proj(x)
        xf_out = self.MFCCTransEncoder(x)
        xf_out = self.mfcc_ln(xf_out)
        tmp = audio.argmax(dim=-1)
        tmp2 = tmp.argmax(dim=-1)
        xf_proj = self.mfcc_proj(xf_out[tmp2, torch.arange(xf_out.shape[1])])
        ########xf_proj = self.audio_proj(xf_out[audio.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, x, timesteps, length=None, wlm=None, fpose=None, xf_proj_pose=None, xf_out_pose=None,
                xf_proj_wlm=None, xf_out_wlm=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        if xf_proj_wlm is None or xf_out_wlm is None:
            xf_proj_wlm, xf_out_wlm = self.encode_mfcc(wlm)

        if xf_proj_pose is None or xf_out_pose is None:
            xf_proj_pose, xf_out_pose = self.encode_pose(fpose, T)

        emb = self.time_embed(
            timestep_embedding(timesteps, self.latent_dim))  + xf_proj_wlm +xf_proj_pose
        # B, T, latent_dim
        h = self.joint_embed(x)


        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)
        # print(src_mask.shape)
        for module in self.temporal_decoder_blocks:
            h = torch.add(h, xf_out_wlm)
            # h = torch.cat((h, xf_out_mfcc), dim=-1)
            h = module(h, xf_out_pose, emb, src_mask)

        output = self.out(h).view(B, T, -1).contiguous()

        return output
