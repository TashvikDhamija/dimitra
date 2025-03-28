import os
import argparse
import time
import glob
import os
from tqdm import tqdm

def get_file_counts(directory):
    """
    Counts the number of .png, .mp4, and .wav files in the given directory.
    """
    png_count = 0
    mp4_count = 0
    wav_count = 0

    # List only files in the directory (exclude subdirectories)
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if item.endswith(".png"):
                png_count += 1
            elif item.endswith(".mp4"):
                mp4_count += 1
            elif item.endswith(".wav"):
                wav_count += 1

    return png_count, mp4_count, wav_count


def check_directory_structure(base_directory):
    """
    Recursively checks subdirectories to ensure they contain the same file counts.
    """
    # Store the reference file counts for comparison
    reference_counts = None

    for root, dirs, files in os.walk(base_directory):
        if not dirs and files:
            png_count, mp4_count, wav_count = get_file_counts(base_directory)
            if png_count!=0 or mp4_count!=0 or wav_count!=0:
                if mp4_count>2 and png_count==0 and wav_count==0:
                    gen_type = "recon_full_dataset"
                elif mp4_count>1 and png_count==1 and wav_count==0:
                    gen_type = "multi_voice_1"
                elif mp4_count==0 and png_count==1 and wav_count>1:
                    gen_type = "multi_voice_2"
                elif mp4_count==1 and png_count==0 and wav_count>1:
                    gen_type = "multi_voice_3"
                elif mp4_count>1 and png_count==0 and wav_count==1:
                    gen_type = "multi_id_1"
                elif mp4_count==0 and png_count>1 and wav_count==1:
                    gen_type = "multi_id_2"
                elif mp4_count==1 and png_count>1 and wav_count==0:
                    gen_type = "multi_id_3"
                else:
                    return None,None
                dir_type="direct"
                break

        else:
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)

                # Get file counts for the current subdirectory
                png_count, mp4_count, wav_count = get_file_counts(subdir_path)

                # Set the reference counts if not already set
                if png_count!=0 or mp4_count!=0 or wav_count!=0:
                    if mp4_count==1 and png_count==0 and wav_count==0:
                        gen_type = "recon"
                    elif mp4_count==0 and png_count==1 and wav_count==1:
                        gen_type = "GEN_1"
                        break
                    elif mp4_count==1 and png_count==0 and wav_count==1:
                        gen_type = "GEN_2"
                        break
                    elif mp4_count==1 and png_count==1 and wav_count==0:
                        gen_type = "GEN_3"
                        break
                    elif mp4_count==2 and png_count==0 and wav_count==0:
                        gen_type = "GEN_4"
                    elif mp4_count>2 and png_count==0 and wav_count==0:
                        gen_type = "recon_full_dataset"
                        break
                    elif mp4_count>1 and png_count==1 and wav_count==0:
                        gen_type = "multi_voice_1"
                        break
                    elif mp4_count==0 and png_count==1 and wav_count>1:
                        gen_type = "multi_voice_2"
                        break
                    elif mp4_count==1 and png_count==0 and wav_count>1:
                        gen_type = "multi_voice_3"
                        break
                    elif mp4_count>1 and png_count==0 and wav_count==1:
                        gen_type = "multi_id_1"
                        break
                    elif mp4_count==0 and png_count>1 and wav_count==1:
                        gen_type = "multi_id_2"
                        break
                    elif mp4_count==1 and png_count>1 and wav_count==0:
                        gen_type = "multi_id_3"
                        break
                    else:
                        return None,None
                    dir_type="recursive"
                

    return gen_type,dir_type


def run_single(args):

    gen_type,dir_type = check_directory_structure(args.input_dir)
    if gen_type is None:
        print("error in data structure")
        return
    print(gen_type)

    global_start_time = time.time()
    start_time = time.time()

    if gen_type=="recon_full_dataset":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.mp4')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {work_dir}/ff.png -loglevel error -hide_banner")
            os.system(f"ffmpeg -y -i {vid} -vn {work_dir}/audio.wav -loglevel error -hide_banner")
    elif gen_type=="multi_voice_1":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.mp4')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vn {work_dir}/audio.wav -loglevel error -hide_banner")
            vid_directory = os.path.dirname(vid)
            # List all files in the directory
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".png"):
                    png_path = os.path.join(vid_directory, file)
                    break
            os.system(f"ffmpeg -y -i {png_path} -vf \"scale={args.res}:{args.res}\" {work_dir}/ff.png -loglevel error -hide_banner")
    elif gen_type=="multi_voice_2":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.wav')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.wav', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"cp {vid} {work_dir}/audio.wav")
            vid_directory = os.path.dirname(vid)
            # List all files in the directory
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".png"):
                    png_path = os.path.join(vid_directory, file)
                    break
            os.system(f"ffmpeg -y -i {png_path} -vf \"scale={args.res}:{args.res}\" {work_dir}/ff.png -loglevel error -hide_banner")
    elif gen_type=="multi_voice_3":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.wav')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.wav', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"cp {vid} {work_dir}/audio.wav")
            vid_directory = os.path.dirname(vid)
            # List all files in the directory
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".mp4"):
                    mp4_path = os.path.join(vid_directory, file)
                    break
            os.system(f"ffmpeg -y -i {mp4_path} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {work_dir}/ff.png -loglevel error -hide_banner")
    elif gen_type=="multi_id_1":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.mp4')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {work_dir}/ff.png -loglevel error -hide_banner")
            # List all files in the directory
            vid_directory = os.path.dirname(vid)
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".wav"):
                    wav_path = os.path.join(vid_directory, file)
                    break
            os.system(f"cp {wav_path} {work_dir}/audio.wav")
    elif gen_type=="multi_id_2":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.png')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.png', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vf \"scale={args.res}:{args.res}\" {work_dir}/ff.png -loglevel error -hide_banner")
            # List all files in the directory
            vid_directory = os.path.dirname(vid)
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".wav"):
                    wav_path = os.path.join(vid_directory, file)
                    break
            os.system(f"cp {wav_path} {work_dir}/audio.wav")
    elif gen_type=="multi_id_3":
        if dir_type=="direct":
            vid_list=glob.glob(f'{args.input_dir}/*.png')
        else:
            vid_list=glob.glob(f'{args.input_dir}/**/*.png', recursive=True)
        for vid in vid_list:
            work_dir = vid.replace(args.input_dir,args.output_dir)[:-4]
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vf \"scale={args.res}:{args.res}\" {work_dir}/ff.png -loglevel error -hide_banner")
            # List all files in the directory
            vid_directory = os.path.dirname(vid)
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".mp4"):
                    mp4_path = os.path.join(vid_directory, file)
                    break
            os.system(f"ffmpeg -y -i {mp4_path} -vn {work_dir}/audio.wav -loglevel error -hide_banner")
    elif gen_type=="recon":
        vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        for vid in vid_list:
            work_dir = os.path.dirname(vid).replace(args.input_dir,args.output_dir)
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {work_dir}/ff.png -loglevel error -hide_banner")
            os.system(f"ffmpeg -y -i {vid} -vn {work_dir}/audio.wav -loglevel error -hide_banner")
    elif gen_type=="GEN_1":
        vid_list=glob.glob(f'{args.input_dir}/**/*.wav', recursive=True)
        for vid in vid_list:
            work_dir = os.path.dirname(vid).replace(args.input_dir,args.output_dir)
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"cp {vid} {work_dir}/audio.wav")
            vid_directory = os.path.dirname(vid)
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".png"):
                    png_path = os.path.join(vid_directory, file)
                    break
            os.system(f"ffmpeg -y -i {png_path} -vf \"scale={args.res}:{args.res}\" {work_dir}/ff.png -loglevel error -hide_banner")
    elif gen_type=="GEN_2":
        vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        for vid in vid_list:
            work_dir = os.path.dirname(vid).replace(args.input_dir,args.output_dir)
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {work_dir}/ff.png -loglevel error -hide_banner")
            vid_directory = os.path.dirname(vid)
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".wav"):
                    wav_path = os.path.join(vid_directory, file)
                    break
            os.system(f"cp {wav_path} {work_dir}/audio.wav")
    elif gen_type=="GEN_3":
        vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        for vid in vid_list:
            work_dir = os.path.dirname(vid).replace(args.input_dir,args.output_dir)
            os.makedirs(work_dir,exist_ok=True)
            os.system(f"ffmpeg -y -i {vid} -vn {work_dir}/audio.wav -loglevel error -hide_banner")
            vid_directory = os.path.dirname(vid)
            for file in os.listdir(vid_directory):
                # Check if the file has a .png extension
                if file.endswith(".png"):
                    png_path = os.path.join(vid_directory, file)
                    break
            os.system(f"ffmpeg -y -i {png_path} -vf \"scale={args.res}:{args.res}\" {work_dir}/ff.png -loglevel error -hide_banner")
    elif gen_type=="GEN_4":
        vid_list=glob.glob(f'{args.input_dir}/**/*.mp4', recursive=True)
        vid_list=sorted(vid_list)
        # Group files by directory
        dir_dict = {}
        for vid in vid_list:
            parent_dir = os.path.dirname(vid)
            if parent_dir not in dir_dict:
                dir_dict[parent_dir] = []
            dir_dict[parent_dir].append(vid)
        for parent_dir, files in dir_dict.items():
            if len(files) < 2:
                print(f"Skipping {parent_dir} as it doesn't contain 2 .mp4 files.")
                continue
            # Sort files within the directory
            files = sorted(files)
            # Use the first file for image extraction
            work_dir = os.path.dirname(files[0]).replace(args.input_dir,args.output_dir)
            os.makedirs(work_dir, exist_ok=True)
            os.system(f"ffmpeg -y -i {files[0]} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {work_dir}/ff.png -loglevel error -hide_banner")
            os.system(f"ffmpeg -y -i {files[1]} -vn {work_dir}/audio.wav -loglevel error -hide_banner")

    preprocess_time = time.time() - start_time
    start_time = time.time()

    print("extracting 3DMM")
    os.system(f"python -u -m Deep3DFaceRecon_pytorch.extract_kp_videos_multi --input_dir {args.output_dir} --output_dir {args.output_dir} --device_ids 0 --workers 8") 
    os.system(f"python -u -m Deep3DFaceRecon_pytorch.face_recon_videos_multi --input_dir {args.output_dir}  --keypoint_dir {args.output_dir} --output_dir {args.output_dir} --inference_batch_size 100 --name=default --epoch=20 --model facerecon")

    dmm_time = time.time() - start_time
    start_time = time.time()


    print("extracting audio features")
    os.system(f"python extract_audio/extract_audio_features_multi.py --model_name \'wavlm\' --dataroot \'{args.output_dir}/**/*.wav\'")

    audio_time = time.time() - start_time
    start_time = time.time()

    print("generating face motion")
    os.system(f"python -u  Dimitra++/Inference_Multi.py -load_face_weights Dimitra++/weights/EXP_weights -load_lips_weights Dimitra++/weights/LIPS_weights -load_head_weights Dimitra++/weights/HEADPOSE_weights -root_dir {args.output_dir} ")

    CMT_time = time.time() - start_time
    start_time = time.time()

    print("rendering video")
    gen_list=glob.glob(f'{args.output_dir}/**/generated.mat', recursive=True)
    for gen_mat in tqdm(gen_list, desc="Processing videos"):
        gen_dir = os.path.dirname(gen_mat)

        if args.res=='224':
            os.system(f"mv {gen_dir}/ff.png {gen_dir}/ff_old.png")
            os.system(f"ffmpeg -y -i {gen_dir}/ff_old.png -vf \"scale=256:256\" {gen_dir}/ff.png -loglevel error -hide_banner")
            os.system(f"rm {gen_dir}/ff_old.png")
            os.system(f"python Pirender/Pirender_inference_256_vox.py --pose_path {gen_mat} --src_img_path {gen_dir}/ff.png --wav_path {gen_dir}/audio.wav --output_path {gen_dir}/Dimitra_output.mp4")
        elif args.res=="512":
            os.system(f"python Pirender/Pirender_inference_512.py --pose_path {gen_mat} --src_img_path {gen_dir}/ff.png --wav_path {gen_dir}/audio.wav --output_path {gen_dir}/Dimitra_output.mp4")
        else:
            os.system(f"python Pirender/Pirender_inference_256.py --pose_path {gen_mat} --src_img_path {gen_dir}/ff.png --wav_path {gen_dir}/audio.wav --output_path {gen_dir}/Dimitra_output.mp4")

    renderer_time = time.time() - start_time

    if args.remove_artifacts:
        start_time = time.time()
        print("removing artifacts")
        if args.res=="512":
            os.system(f"python RestoreFormer++/inference_loop.py --videos_path {args.output_dir} --name 01 -v RestoreFormer++ -s 1 --save")
        else:
            os.system(f"python RestoreFormer++/inference_loop.py --videos_path {args.output_dir} --name 01 -v RestoreFormer++ -s 2 --save")
        restore_time = time.time() - start_time


    if not args.no_watermark:
        #ADD watermarks
        gen_list=glob.glob(f'{args.output_dir}/**/Dimitra_output.mp4', recursive=True)
        for gen_mat in tqdm(gen_list, desc="Processing videos"):
            gen_dir = os.path.dirname(gen_mat)
            os.system(f"mv {gen_dir}/Dimitra_output.mp4 {gen_dir}/Dimitra_output_old.mp4")
            if args.res=="512":
                os.system(f"ffmpeg -y -i {gen_dir}/Dimitra_output_old.mp4 -i images/watermark_512.png -filter_complex \"overlay=0:0\" {gen_dir}/Dimitra_output.mp4 -loglevel error -hide_banner")
            else:
                os.system(f"ffmpeg -y -i {gen_dir}/Dimitra_output_old.mp4 -i images/watermark_256.png -filter_complex \"overlay=0:0\" {gen_dir}/Dimitra_output.mp4 -loglevel error -hide_banner")
            os.system(f"rm {gen_dir}/Dimitra_output_old.mp4")
        if args.remove_artifacts:
            gen_list=glob.glob(f'{args.output_dir}/**/Dimitra_output_cleaned.mp4', recursive=True)
            for gen_mat in tqdm(gen_list, desc="Processing videos"):
                gen_dir = os.path.dirname(gen_mat)
                os.system(f"mv {gen_dir}/Dimitra_output_cleaned.mp4 {gen_dir}/Dimitra_output_cleaned_old.mp4")
                os.system(f"ffmpeg -y -i {gen_dir}/Dimitra_output_cleaned_old.mp4 -i images/watermark_512.png -filter_complex \"overlay=0:0\" {gen_dir}/Dimitra_output_cleaned.mp4 -loglevel error -hide_banner")
                os.system(f"rm {gen_dir}/Dimitra_output_cleaned_old.mp4")

    total_time = time.time() - global_start_time


    print(f"preprocessing took : {preprocess_time}")
    print(f"3dmm extraction took : {dmm_time}")
    print(f"audio extraction took : {audio_time}")
    print(f"motion generation took : {CMT_time}")
    print(f"rendering took : {renderer_time}")
    if args.remove_artifacts:
        print(f"artifact removing took : {restore_time}")
    print(f"total generation took : {total_time}")


def main():
    parser = argparse.ArgumentParser(description="Run single")
    parser.add_argument("--input_dir", type=str, default='input_multi')
    parser.add_argument("--output_dir", type=str, default='output_multi')
    parser.add_argument("--res", type=str, default='512', choices=['256','512'])
    parser.add_argument("--remove_artifacts", action='store_true') 
    parser.add_argument("--no_watermark", action='store_true')
    parser.add_argument("--vox", action='store_true')

    args = parser.parse_args()

    if args.vox:
        args.res = '224'

    run_single(args)

if __name__ == "__main__":
    main()