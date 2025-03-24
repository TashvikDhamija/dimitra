from transformers import   WavLMModel,  Wav2Vec2FeatureExtractor
import soundfile as sf
import numpy as np
import torch
import torchaudio
import glob
import os
from tqdm import tqdm
from scipy.io import loadmat
import argparse
import wave

# Define model options
MODEL_OPTIONS = {
    'wavlm': ('microsoft/wavlm-large', WavLMModel, Wav2Vec2FeatureExtractor)
}

def load_model(model_name, device):
    #print(f"Loading the {model_name.capitalize()} Processor and Model...")
    model_path, model_class, processor_class = MODEL_OPTIONS[model_name]
    
    # Define local checkpoint directory
    local_checkpoint_dir = f"extract_audio/checkpoints/{model_name}/"
    
    # Check if the local checkpoint exists
    if os.path.exists(local_checkpoint_dir):
        #print(f"Loading model from local checkpoint: {local_checkpoint_dir}")
        processor = processor_class.from_pretrained(local_checkpoint_dir)
        model = model_class.from_pretrained(local_checkpoint_dir)
    else:
        # If no local checkpoint, download from the online source
        print(f"Local checkpoint not found. Downloading from {model_path}...")
        processor = processor_class.from_pretrained(model_path)
        model = model_class.from_pretrained(model_path)
        # Save the downloaded model for future use
        if not os.path.exists(local_checkpoint_dir):
            os.makedirs(local_checkpoint_dir)
        processor.save_pretrained(local_checkpoint_dir)
        model.save_pretrained(local_checkpoint_dir)
    
    model = model.to(device)
    return processor, model

@torch.no_grad()
def get_embeddings_from_16k_speech(speech, processor, model, device="cuda:0"):
    model = model.to(device)
    if speech.ndim == 2:
        if speech.shape[1]==2:
            speech = speech[:, 0]  # [T, 2] ==> [T,]
        elif speech.shape[0]==2:
            speech = speech[0, :]
        else:
            print("ERROR, something wrong with the number of channel in audio")
    
    input_values_all = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    input_values_all = input_values_all.to(device)

    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []

    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])

    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all

    if input_values.shape[1] >= kernel:
        hidden_states = model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])

    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret


def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def process_audio_files(root_path, model_name, window_size):
    audio_files = glob.glob(root_path, recursive=True)
    processor, model = load_model(model_name,"cuda")
    
    for audio_file in tqdm(audio_files, desc=f"Processing audio files with {model_name}"):
        out_name = audio_file.replace('.wav', '.npy')

        if os.path.exists(out_name):
            continue

        num_frames = int(get_wav_duration(audio_file)*25)
        if num_frames == 0:
            print(f"Skipping {audio_file}: No corresponding video frames found.")
            continue

        speech, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            speech = resampler(speech)

        speech_16k = speech.squeeze().numpy()
        embeddings = get_embeddings_from_16k_speech(speech_16k, processor, model)

        audio_window_list = []
        center_idx_list = [2 * idx for idx in range(0, num_frames)]
        padding = np.zeros(embeddings.shape[1], dtype=np.float32)
        for center_idx in center_idx_list:
            cur_audio_window = []
            for i in range(center_idx - window_size, center_idx + window_size + 1):
                if i < 0:
                    cur_audio_window.append(padding)
                elif i >= len(embeddings):
                    cur_audio_window.append(padding)
                else:
                    cur_audio_window.append(embeddings[i])

            cur_audio_win_array = np.stack(cur_audio_window, axis=0)
            audio_window_list.append(cur_audio_win_array)
        audio_window_array = np.stack(audio_window_list, axis=0)

        np.save(out_name, audio_window_array)
        #except:
        #    print("issue processing file skipping : "+video_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio files with specified model.')
    parser.add_argument('--model_name', type=str, choices=['wavlm'], 
                        help='Name of the model to use for processing')
    parser.add_argument('--dataroot', type=str, default='output_multi/**/*.wav', help='Root path for audio files')
    parser.add_argument('--window_size', type=int, default=4, help='Window size for audio feature extraction')

    args = parser.parse_args()
    
    process_audio_files(args.dataroot, args.model_name, args.window_size)
