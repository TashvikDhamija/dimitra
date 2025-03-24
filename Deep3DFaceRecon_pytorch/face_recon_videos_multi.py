import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat

import torch

from Deep3DFaceRecon_pytorch.models import create_model
from Deep3DFaceRecon_pytorch.options.inference_options import InferenceOptions
from Deep3DFaceRecon_pytorch.util.preprocess import align_img
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
from Deep3DFaceRecon_pytorch.util.util import mkdirs, tensor2im, save_image
import warnings

# Suppress only the DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_data_path(keypoint_root):
    filenames = list()
    keypoint_filenames_1 = list()

    VIDEO_EXTENSIONS_LOWERCASE = {'txt'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS

    for ext in extensions:
        keypoint_filenames_1 += glob.glob(f'{keypoint_root}/**/*.{ext}', recursive=True)
    keypoint_filenames = []
    for key_file in keypoint_filenames_1:
        mat_file = key_file.replace('.txt', '.mat')
        if not os.path.isfile(mat_file):
            keypoint_filenames.append(key_file)
    keypoint_filenames = sorted(keypoint_filenames)

    for file_vid in keypoint_filenames:
        file_vid = file_vid.replace('.txt', '.png')
        filenames.append(file_vid)

    assert len(filenames) == len(keypoint_filenames)

    return filenames, keypoint_filenames


class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = filenames
        self.txt_filenames = txt_filenames
        self.lm3d_std = load_lm3d(bfm_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        txt_filename = self.txt_filenames[index]

        frames = self.read_image(filename)
        lm = np.loadtxt(txt_filename).astype(np.float32)
        lm = lm.reshape([len(frames), -1, 2])
        out_images, out_trans_params = list(), list()
        for i in range(len(frames)):
            dat = self.image_transform(frames[i], lm[i])
            if dat is None:
                return {'imgs': 'None', 'trans_param': 'None', 'filename': filename}
            out_img, _, out_trans_param = dat
            out_images.append(out_img[None])
            out_trans_params.append(out_trans_param[None])
        return {
            'imgs': torch.cat(out_images, 0),
            'trans_param': torch.cat(out_trans_params, 0),
            'filename': filename
        }

    def read_image(self, filename):
        frames = []
        cap = cv2.imread(filename)
        frame = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
        frames.append(frame)
        return frames

    def read_video(self, filename):
        frames = list()
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def image_transform(self, images, lm):
        W, H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2] + 1) / 2.
            lm = np.concatenate(
                [lm[:, :1] * W, lm[:, 1:2] * H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        data = align_img(images, lm, self.lm3d_std)
        if data is None:
            return None
        trans_params, img, lm, _ = data
        img = torch.tensor(np.array(img) / 255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params


def main(opt, model):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    filenames, keypoint_filenames = get_data_path(opt.keypoint_dir)
    dataset = VideoPathDataset(filenames, keypoint_filenames, opt.bfm_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # can noly set to one here!
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batch_size = opt.inference_batch_size
    for data in tqdm(dataloader):
        try:

            if isinstance(data['imgs'], list):
                print(data['filename'])
                print('error skip')
                continue
            num_batch = data['imgs'][0].shape[0] // batch_size + 1
            name_t = data['filename'][0]
            name_t = name_t.replace('.png', '.mat')
            if os.path.isfile(name_t):
                print('skipped')
                continue
            pred_coeffs = list()
            for index in range(num_batch):
                data_input = {
                    'imgs': data['imgs'][0, index * batch_size:(index + 1) * batch_size],
                }
                if data_input['imgs'].nelement() == 0:
                    continue
                model.set_input(data_input)

                model.test()
                pred_coeff = {key: model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
                pred_coeff = np.concatenate([
                    pred_coeff['id'],
                    pred_coeff['exp'],
                    pred_coeff['tex'],
                    pred_coeff['angle'],
                    pred_coeff['gamma'],
                    pred_coeff['trans']], 1)
                pred_coeffs.append(pred_coeff)
                visuals = model.get_current_visuals()  # get image results
                if False:  # debug
                    for name in visuals:
                        images = visuals[name]
                        for i in range(images.shape[0]):
                            image_numpy = tensor2im(images[i])
                            save_image(
                                image_numpy,
                                os.path.join(
                                    opt.output_dir,
                                    os.path.basename(data['filename'][0]) + str(i).zfill(5) + '.jpg')
                            )
                    exit()

            pred_coeffs = np.concatenate(pred_coeffs, 0)
            pred_trans_params = data['trans_param'][0].cpu().numpy()
            name = data['filename'][0]

            name = name.replace('.png', '.mat')
            dir = name.split('/')
            dir = '/'.join(dir[0:-1])
            os.makedirs(dir, exist_ok=True)
            savemat(
                name,
                {'coeff': pred_coeffs, 'transform_params': pred_trans_params}
            )
        except:
            print("issue processing file skipping : "+ data['filename'][0])

if __name__ == '__main__':
    opt = InferenceOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    model.device = 'cuda:0'
    model.parallelize()
    model.eval()
    main(opt, model)
