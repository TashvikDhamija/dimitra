import os
import argparse
import time


def check_directory(directory):
    mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    # Check for a single .mp4 file
    if len(mp4_files) == 1 and not png_files and not wav_files:
       return mp4_files[0], None , "recon"

    # Check for a .png file with a .wv file
    if len(png_files) == 1 and len(wav_files) == 1 and not mp4_files:
        return png_files[0], wav_files[0], "GEN_1"
    
    if len(mp4_files) == 1 and len(wav_files) == 1 and not png_files:
        return mp4_files[0], wav_files[0],"GEN_2"
    
    if len(mp4_files) == 1 and len(png_files) == 1 and not wav_files:
        return mp4_files[0], png_files[0],"GEN_3"
    
    if len(mp4_files) == 2 and not png_files and not wav_files:
        mp4_files=sorted(mp4_files)
        return mp4_files[0], mp4_files[1],"GEN_4"

    return None, None, None

def run_single(args):

    global_start_time = time.time()


    start_time = time.time()

    input_path = args.input_dir
    output_path = args.output_dir
    file_1, file_2, gen_type = check_directory(input_path)
    if file_1 is None:
        print("please clen the input directory. supported data: .mp4 X1; .mp4 X2 ; .wav+.png ; .wav+.mp4 ; .png+.mp4")
        return
    elif gen_type == "recon":
        os.system(f"ffmpeg -y -i {input_path}/{file_1} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {output_path}/ff.png -vn {output_path}/audio.wav -loglevel error -hide_banner")
    elif gen_type =="GEN_1":
        os.system(f"ffmpeg -y -i {input_path}/{file_1} -vf \"scale={args.res}:{args.res}\" {output_path}/ff.png -loglevel error -hide_banner")
        os.system(f'mv {input_path}/{file_2} {output_path}/audio.wav')
    elif gen_type =="GEN_2":
        os.system(f"ffmpeg -i {input_path}/{file_1} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {output_path}/ff.png -loglevel error -hide_banner")
        os.system(f'mv {input_path}/{file_2} {output_path}/audio.wav')

    elif gen_type=="GEN_3":
        os.system(f"ffmpeg -y -i {input_path}/{file_1} -vn {output_path}/audio.wav -loglevel error -hide_banner")
        os.system(f"ffmpeg -y -i {input_path}/{file_2} -vf \"scale={args.res}:{args.res}\" {output_path}/ff.png -loglevel error -hide_banner")
    elif gen_type=="GEN_4":
        os.system(f"ffmpeg -y -i {input_path}/{file_1} -vf \"select=eq(n\\,0),scale={args.res}:{args.res}\" -frames:v 1 {output_path}/ff.png -loglevel error -hide_banner")
        os.system(f"ffmpeg -y -i {input_path}/{file_2} -vn {output_path}/audio.wav -loglevel error -hide_banner")

    file_1="ff.png"
    file_2="audio.wav"
    png_path=output_path+"/"+file_1
    audio_path = output_path+"/"+file_2

    preprocess_time = time.time() - start_time
    start_time = time.time()

    print("extracting 3DMM")
    os.system(f"python -u -m Deep3DFaceRecon_pytorch.extract_kp_videos_single --input_dir {output_path} --output_dir {output_path} --device_ids 0 --workers 8") 
    os.system(f"python -u -m Deep3DFaceRecon_pytorch.face_recon_videos_single --input_dir {output_path}  --keypoint_dir {output_path} --output_dir {output_path} --inference_batch_size 100 --name=default --epoch=20 --model facerecon")
    dmm_path = output_path+"/"+file_1.replace(".png",".mat")

    dmm_time = time.time() - start_time
    start_time = time.time()


    print("extracting audio features")
    os.system(f"python extract_audio/extract_audio_features_single.py --model_name \'wavlm\' --dataroot \'{output_path}/*.wav\'")
    aud_path = output_path+"/"+file_2.replace(".wav",".npy")

    audio_time = time.time() - start_time
    start_time = time.time()

    print("generating face motion")
    os.system(f"python -u  Dimitra++/Inference_Single.py -face_weights_path Dimitra++/weights/EXP_weights -lips_weights_path Dimitra++/weights/LIPS_weights -head_weights_path Dimitra++/weights/HEADPOSE_weights -dmm_path {dmm_path} -wlm_path {aud_path}")
    output_dmm_path  = output_path+"/"+file_1.replace(".png",".mat")

    CMT_time = time.time() - start_time
    start_time = time.time()




    print("rendering video")
    if args.res=='224':
        os.system(f"mv {output_path}/ff.png {output_path}/ff_old.png")
        os.system(f"ffmpeg -y -i {output_path}/ff_old.png -vf \"scale=256:256\" {output_path}/ff.png -loglevel error -hide_banner")
        os.system(f"rm {output_path}/ff_old.png")
        os.system(f"python Pirender/Pirender_inference_256_vox.py --pose_path {output_dmm_path} --src_img_path {png_path} --wav_path {audio_path} --output_path {output_path}/Dimitra_output.mp4")

    elif args.res=="512":
        os.system(f"python Pirender/Pirender_inference_512.py --pose_path {output_dmm_path} --src_img_path {png_path} --wav_path {audio_path} --output_path {output_path}/Dimitra_output.mp4")
    else:
        os.system(f"python Pirender/Pirender_inference_256.py --pose_path {output_dmm_path} --src_img_path {png_path} --wav_path {audio_path} --output_path {output_path}/Dimitra_output.mp4")

    renderer_time = time.time() - start_time

    if args.remove_artifacts:
        start_time = time.time()
        print("removing artifacts")
        if args.res=="512":
            os.system(f"python RestoreFormer++/inference_loop.py --videos_path {output_path} --name 01 -v RestoreFormer++ -s 1 --save --single")
        else:
            os.system(f"python RestoreFormer++/inference_loop.py --videos_path {output_path} --name 01 -v RestoreFormer++ -s 2 --save --single")
        restore_time = time.time() - start_time

    if not args.no_watermark:
        #ADD watermarks
        os.system(f"mv {output_path}/Dimitra_output.mp4 {output_path}/Dimitra_output_old.mp4")
        if args.res=="512":
            os.system(f"ffmpeg -y -i {output_path}/Dimitra_output_old.mp4 -i images/watermark_512.png -filter_complex \"overlay=0:0\" {output_path}/Dimitra_output.mp4 -loglevel error -hide_banner")
        else:
            os.system(f"ffmpeg -y -i {output_path}/Dimitra_output_old.mp4 -i images/watermark_256.png -filter_complex \"overlay=0:0\" {output_path}/Dimitra_output.mp4 -loglevel error -hide_banner")
        os.system(f"rm {output_path}/Dimitra_output_old.mp4")
        if args.remove_artifacts:
            os.system(f"mv {output_path}/Dimitra_output_cleaned.mp4 {output_path}/Dimitra_output_cleaned_old.mp4")
            os.system(f"ffmpeg -y -i {output_path}/Dimitra_output_cleaned_old.mp4 -i images/watermark_512.png -filter_complex \"overlay=0:0\" {output_path}/Dimitra_output_cleaned.mp4 -loglevel error -hide_banner")
            os.system(f"rm {output_path}/Dimitra_output_cleaned_old.mp4")


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
    parser.add_argument("--input_dir", type=str, default='input_single')
    parser.add_argument("--output_dir", type=str, default='output_single')
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