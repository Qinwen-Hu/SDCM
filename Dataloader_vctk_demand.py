import soundfile as sf
import librosa
import torch
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import random
from omegaconf import OmegaConf as OC

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fs=48000, length_in_seconds=8, random_start_point=False, train=True):
        self.train_clean_list = pd.read_csv('./vctk_train_clean_data.csv')['file_path'].to_list()
        self.valid_clean_list = pd.read_csv('./vctk_valid_clean_data.csv')['file_path'].to_list()
     
        self.L = length_in_seconds * fs
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.train = train
        print('%s audios for training, %s for validation' %(len(self.train_clean_list), len(self.valid_clean_list)))

    def __getitem__(self, idx):
        if self.train:
            clean_list = self.train_clean_list
        else:
            clean_list = self.valid_clean_list      
        if self.random_start_point:
            Begin_S = int(np.random.uniform(0,10 - self.length_in_seconds)) * self.fs
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            noisy, sr_n = sf.read(clean_list[idx].replace('clean','noisy'), dtype='float32', start= Begin_S, stop = Begin_S + self.L)
        else:
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= 0,stop = self.L) 
            noisy, sr_n = sf.read(clean_list[idx].replace('clean','noisy'), dtype='float32', start= 0, stop = self.L)
            
        
        return noisy, clean 

    def __len__(self):
        if self.train:
            return len(self.train_clean_list)
        else:
            return len(self.valid_clean_list)
        
        
def dataset_preprocess(args):
    
    def get_audio(file_path, sr):
        x, sr_x = sf.read(file_path, dtype='int16')
        if len(x.shape) != 1:
            x = x[:,0]
        if sr_x != sr:
            x = x.astype('float32')
            x = x / 32767.0
            x = librosa.resample(x, sr_x, sr)
            x = x * 32767
            x = x.astype('int16')
        return x
            
    clean_data_path = os.path.join(args.data.data_path, 'clean_trainset_28spk_wav')
    noisy_data_path = os.path.join(args.data.data_path, 'noisy_trainset_28spk_wav')
    tgt_clean_data_path = os.path.join(args.data.target_data_path, 'clean_split')
    tgt_noisy_data_path = os.path.join(args.data.target_data_path, 'noisy_split')
    clean_file_list = librosa.util.find_files(clean_data_path, ext='wav')
    os.makedirs(tgt_clean_data_path, exist_ok=True)
    os.makedirs(tgt_noisy_data_path, exist_ok=True)

    buffer_clean = np.array([], dtype='int16')
    buffer_noisy = np.array([], dtype='int16')
    fileid = 0
    length = args.data.length_seconds
    sr = args.data.target_sr
    
    for clean_file in tqdm(clean_file_list):
        s = get_audio(clean_file, sr)
        x = get_audio(os.path.join(noisy_data_path, os.path.split(clean_file)[-1]), sr)
        buffer_clean = np.concatenate([buffer_clean, s])
        buffer_noisy = np.concatenate([buffer_noisy, x])
        
        if len(buffer_clean) > (length * sr):
            audio_clean = buffer_clean[:(length * sr)]
            audio_noisy = buffer_noisy[:(length * sr)]
            buffer_clean = buffer_clean[(length * sr):]
            buffer_noisy = buffer_noisy[(length * sr):]
            sf.write(os.path.join(tgt_clean_data_path, '{}.wav'.format(fileid)), audio_clean, sr)
            sf.write(os.path.join(tgt_noisy_data_path, '{}.wav'.format(fileid)), audio_noisy, sr)
            fileid += 1
            
    tgt_clean_list = librosa.util.find_files(tgt_clean_data_path, ext='wav')
    random.shuffle(tgt_clean_list)
    train_clean_list = [[file] for file in tgt_clean_list[:int(len(tgt_clean_list) * args.data.train_split)]]
    valid_clean_list = [[file] for file in tgt_clean_list[int(len(tgt_clean_list) * args.data.train_split):]]
    df = pd.DataFrame(train_clean_list, columns=['file_path'])
    df.to_csv('./vctk_train_clean_data.csv', index=False)
    df = pd.DataFrame(valid_clean_list, columns=['file_path'])
    df.to_csv('./vctk_valid_clean_data.csv', index=False)



if __name__=='__main__':
    
    args = OC.load('config_fullsubp.yaml')
    dataset_preprocess(args)