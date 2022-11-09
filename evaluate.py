from models.fullsubnet_plus import FullSubNet_Plus_SCM, FullSubNet_Plus_ERB
from models.dpt_fsnet import DPT_FSNet_SCM, DPT_FSNet_ERB
from audio_zen.acoustics.feature import mag_phase
from signal_processing import iSTFT_module_1_8, iSTFT_module_1_7
# use iSTFT_module_1_7 with pytorch version < 1.8, use iSTFT_module_1_8 with pytorch version >= 1.8

import soundfile as sf
import librosa
import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf as OC

from pesq import pesq
from pystoi import stoi

'''
metrics
'''
def LSD(clean, est):
    '''
    

    Parameters
    ----------
    clean : 
        [F, T]
    est : 
        [F, T]
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''    
    clean_log = np.log10(np.clip(clean, a_min=1e-8, a_max=None))
    est_log = np.log10(np.clip(est, a_min=1e-8, a_max=None))
    lsd = np.mean(np.power((est_log - clean_log), 2), axis=0)
    lsd = np.mean(np.sqrt(lsd))

    return lsd

def SI_SDR(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    References
        SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)

'''
inference
'''
def infer(model_type, model, x_audio, hparams):
    device = torch.device(hparams.train.device)
    
    def infer_fullsubp(model, x_audio, hparams):
        with torch.no_grad():
            WINDOW = torch.sqrt(torch.hann_window(hparams.feature.n_win, device=device) + 1e-8)
            noisy_48k = torch.from_numpy(x_audio.reshape((1,len(x_audio)))).to(device) 
            noisy_stft = torch.stft(noisy_48k, hparams.feature.n_fft, hparams.feature.n_hop, win_length=hparams.feature.n_win, window=WINDOW, center=True, return_complex=True) #[Bs, F, T]   
            noisy_mag, _ = mag_phase(noisy_stft)
            noisy_real = torch.real(noisy_stft)
            noisy_imag = torch.imag(noisy_stft)
            
            noisy_mag = noisy_mag.unsqueeze(1) 
            noisy_real = noisy_real.unsqueeze(1)
            noisy_imag = noisy_imag.unsqueeze(1)
            
            #-------------mask predition-----------------------
            cRM = model(noisy_mag, noisy_real, noisy_imag) 
            #------------------------------------         
            
            enh_real = cRM[:,0,:,:].unsqueeze(1) * noisy_real - cRM[:,1,:,:].unsqueeze(1) * noisy_imag
            enh_imag = cRM[:,0,:,:].unsqueeze(1) * noisy_imag + cRM[:,1,:,:].unsqueeze(1) * noisy_real
            enh_stft = torch.cat((enh_real, enh_imag), dim=1) 
            enh_stft = enh_stft.permute(0, 1, 3, 2).contiguous() 
            enhan_48k = iSTFT_module_1_7(n_fft=hparams.feature.n_fft, hop_length=hparams.feature.n_hop, win_length=hparams.feature.n_win, window=WINDOW, center=True, length=noisy_48k.shape[-1])(enh_stft)
            
            enhan_48k = enhan_48k[0,:].cpu().detach().numpy()
            
            length = len(x_audio)
            enhan_48k = librosa.util.fix_length(enhan_48k, length)
        return enhan_48k
        
    def infer_dptfsnet(model, x_audio, hparams):
        with torch.no_grad():
            WINDOW = torch.sqrt(torch.hann_window(hparams.feature.n_win, device=device) + 1e-8)
            noisy_48k = torch.from_numpy(x_audio.reshape((1,len(x_audio)))).to(device) 
            noisy_stft = torch.stft(noisy_48k, hparams.feature.n_fft, hparams.feature.n_hop, win_length=hparams.feature.n_win, window=WINDOW, center=True, return_complex=False) #[Bs, F, T]   
            noisy_stft = noisy_stft.permute(0, 3, 2, 1).contiguous() #[Bs, 2, T, F]
                
            #-------------spectrogram predition-----------------------
            enhan_stft = model(noisy_stft) #[B, 2, T, F]
            #------------------------------------         
            enhan_48k = iSTFT_module_1_7(n_fft=hparams.feature.n_fft, hop_length=hparams.feature.n_hop, win_length=hparams.feature.n_win, window=WINDOW, center=True, length=noisy_48k.shape[-1])(enhan_stft)            
            enhan_48k = enhan_48k[0,:].cpu().detach().numpy()
            
            length = len(x_audio)
            enhan_48k = librosa.util.fix_length(enhan_48k, length)
        return enhan_48k

    
    if model_type in ['f_scm', 'f_erb']:
        return infer_fullsubp(model, x_audio, hparams)
    else:
        return infer_dptfsnet(model, x_audio, hparams)


def eval(args):
    assert args.model in ['f_scm', 'f_erb', 'd_scm', 'd_erb'], 'Invalid model name'
    if args.model == 'f_scm':
        model = FullSubNet_Plus_SCM(weight_init=False)
        hparams = OC.load('./config_fullsubp.yaml')
    elif args.model == 'f_erb':
        model = FullSubNet_Plus_ERB(weight_init=False)
        hparams = OC.load('./config_fullsubp.yaml')
    elif args.model == 'd_scm':
        model = DPT_FSNet_SCM()
        hparams = OC.load('./config_dptfsnet.yaml')
    else: 
        model = DPT_FSNet_ERB()
        hparams = OC.load('./config_dptfsnet.yaml')
        
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    os.makedirs(args.output_dir, exist_ok=True)
    
    noisy, _ = sf.read(args.noisy_audio_path, dtype='float32')
    hparams.train.device = args.device
    enhan = infer(args.model, model, noisy, hparams)
    sf.write(os.path.join(args.output_dir, os.path.split(args.noisy_audio_path)[-1]), enhan, 48000)
    
    if args.clean_audio_path:
        clean, _ = sf.read(args.clean_audio_path)
        clean_16k = librosa.resample(clean, 48000, 16000)
        enhan_16k = librosa.resample(enhan, 48000, 16000)
        clean_spec = np.abs(librosa.stft(clean, hparams.feature.n_fft, hparams.feature.n_hop, hparams.feature.n_win)) **2
        enhan_spec = np.abs(librosa.stft(enhan, hparams.feature.n_fft, hparams.feature.n_hop, hparams.feature.n_win)) **2
        
        pesq_score = pesq(16000, clean_16k, enhan_16k, 'wb')
        stoi_score = stoi(clean, enhan, 48000)
        lsd_score = LSD(clean_spec, enhan_spec)
        sisdr_score = SI_SDR(clean, enhan, 48000)
        print('Pesq-wb:{}, Stoi:{}, LSD:{}, SI-SDR:{}'.format(pesq_score, stoi_score, lsd_score, sisdr_score))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Choose the model to be evaluated. \
        Available models include: "f_scm" = fullsubplus_scm; \
                                  "f_erb" = fullsubplus_erb; \
                                  "d_scm" = dptfsnet_scm;\
                                  "d_erb" = dptfsnet_erb.')
    parser.add_argument('--ckpt_path', required=True, help='Path for the pretrained model.')
    parser.add_argument('--noisy_audio_path', required=True, help='Path for the test audio.')
    parser.add_argument('--clean_audio_path', required=False, help='Path for the target audio. Not required if there is no reference signal.')
    parser.add_argument('--device', required=False, help='Device to run the inference algorithm.', default='cuda:0')
    parser.add_argument('--output_dir', required=False, help='Directory to save the enhanced audio.', default='./enhanced_result')
    args = parser.parse_args()
    eval(args)