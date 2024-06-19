import numpy as np
from models import generator, discriminator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm

import librosa
import json

@torch.no_grad()
def enhance_one_track(
    gen_model, dis_model, clean_path, noisy_path, enhanced_path, cut_len, n_fft=400, hop=100):
    noisy, sr = torchaudio.load(noisy_path)
    clean, sr = torchaudio.load(clean_path)

    noisy = noisy.cuda()
    clean = clean.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    clean = torch.cat([clean, clean[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))
        clean = torch.reshape(clean, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
    )
    clean_spec = torch.stft(
        clean, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True,
    )

    noisy_spec_gen = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = gen_model(noisy_spec_gen)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    est_mag = torch.sqrt(est_real**2 + est_imag**2)

    noisy_spec = power_compress(noisy_spec)
    noisy_real = noisy_spec[:, 0, :, :].unsqueeze(1)
    noisy_imag = noisy_spec[:, 1, :, :].unsqueeze(1)
    noisy_mag = torch.sqrt(noisy_real**2 + noisy_imag**2)

    clean_spec = power_compress(clean_spec)
    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
    clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

    # calculate discriminator score
    enhanced_score = dis_model(
                clean_mag, est_mag.detach()
            ).mean().item()

    noisy_score = dis_model(
                clean_mag, noisy_mag.detach()
            ).mean().item()

    clean_score = dis_model(
                clean_mag, clean_mag.detach()
            ).mean().item()

    if (enhanced_score < noisy_score):
        return None

    # save enhanced audio
    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    # assert len(est_audio) == length
    sf.write(enhanced_path, est_audio, sr)

    return [clean_score, noisy_score, enhanced_score]


def main(gen_path, dis_path, noisy_dir, clean_dir, enhanced_dir):
    # load generator

    n_fft = 400
    gen_model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    gen_model.load_state_dict((torch.load(gen_path)))
    gen_model.eval()

    # load discriminator
    dis_model = discriminator.Discriminator(ndf=16).cuda()
    dis_model.load_state_dict((torch.load(dis_path)))
    dis_model.eval()

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)

    clean_score_dict = dict()
    noisy_score_dict = dict()
    enhanced_score_dict = dict()

    count = 0

    for audio in tqdm(audio_list):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        enhanced_path = os.path.join(enhanced_dir, audio)     

        # clean_audio, sr = sf.read(clean_path)
        # if (len(clean_audio) > 100000):
        #     continue

        dis_score = enhance_one_track(
            gen_model, dis_model, clean_path, noisy_path, enhanced_path, 16000 * 16, n_fft, n_fft // 4
        )

        if (dis_score != None):
            clean_score, noisy_score, enhanced_score = dis_score
        
            clean_score_dict[audio] = clean_score
            noisy_score_dict[audio] = noisy_score
            enhanced_score_dict[audio] = enhanced_score
        else:
            count += 1
    
    with open(f'{args.data_dir}/clean_score.json', 'w') as file:
        json.dump(clean_score_dict, file)

    with open(f'{args.data_dir}/noisy_score.json', 'w') as file:
        json.dump(noisy_score_dict, file)

    with open(f'{args.data_dir}/enhanced_score.json', 'w') as file:
        json.dump(enhanced_score_dict, file)

    print(len(enhanced_score_dict))
    print(count)

parser = argparse.ArgumentParser()
parser.add_argument("--gen_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the generator is saved")
parser.add_argument("--dis_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the discriminator is saved")
parser.add_argument("--data_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")

args = parser.parse_args()


if __name__ == "__main__":
    noisy_dir = os.path.join(args.data_dir, "noisy")
    clean_dir = os.path.join(args.data_dir, "clean")
    enhanced_dir = os.path.join(args.data_dir, "enhanced")

    if (not os.path.exists(enhanced_dir)):
        os.makedirs(enhanced_dir)

    main(args.gen_path, args.dis_path, noisy_dir, clean_dir, enhanced_dir)