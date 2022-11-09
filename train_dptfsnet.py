import os
from omegaconf import OmegaConf as OC

import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import soundfile as sf
from math import ceil

from Dataloader_vctk_demand import Dataset
from models.dpt_fsnet import DPT_FSNet_SCM, DPT_FSNet_ERB

from signal_processing import iSTFT_module_1_8, iSTFT_module_1_7
# use iSTFT_module_1_7 with pytorch version < 1.8, use iSTFT_module_1_8 with pytorch version >= 1.8

torch.set_default_tensor_type(torch.FloatTensor)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, k1, k2, warmup, step_per_epoch, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.k1 = k1
        self.k2 = k2
        self.warmup = warmup
        self.model_size = model_size
        self.step_per_epoch = step_per_epoch
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        
        if step <= self.warmup:
            return  self.k1 * self.model_size ** (-0.5) * step * self.warmup ** (-1.5) 
        else:
            epoch = ceil(step / self.step_per_epoch)
            return self.k2 * 0.98 ** (epoch / 2)
        
def train(args):
    device = torch.device(args.train.device)
    WINDOW = torch.sqrt(torch.hann_window(args.feature.n_win, device=device) + 1e-8)
    
    def Loss(y_pred, y_true):
        snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),(torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        snr_loss = 10 * torch.log10(snr + 1e-7)
        
        pred_stft = torch.stft(y_pred, args.feature.n_fft, args.feature.n_hop,win_length=args.feature.n_win, window=WINDOW, center=True)
        true_stft = torch.stft(y_true, args.feature.n_fft, args.feature.n_hop,win_length=args.feature.n_win, window=WINDOW, center=True)
        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = torch.mean((pred_real_c - true_real_c)**2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
        mag_loss = torch.mean((pred_mag**(0.3)-true_mag**(0.3))**2)
        
        return 0.3*(real_loss + imag_loss) + 0.7*mag_loss, snr_loss
    
    '''create ckpt dir'''
    os.makedirs(args.log.ckpt_dir, exist_ok=True)

    '''model'''
    model = DPT_FSNet_SCM()
    model = torch.nn.DataParallel(model, device_ids=args.train.device_ids)
    model = model.to(device)


    '''load train data'''
    dataset = Dataset(length_in_seconds=args.train.length_in_seconds, random_start_point=True, train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, drop_last=True)
    dataset_valid = Dataset(length_in_seconds=args.train.length_in_seconds, random_start_point=True, train=False)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, drop_last=True)
    
    '''optimizer & lr_scheduler'''
    optimizer = NoamOpt(model_size=32, k1=args.train.k1, k2=args.train.k2, warmup=args.train.warmup, step_per_epoch=len(data_loader),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0))


    '''start train'''
    for epoch in range(args.train.epochs):
        train_loss = []
        valid_loss = []
        snr_valid_loss = []
        model.train()
        dataset.train = True
        dataset.random_start_point = True
        idx = 0

        '''train'''
        print('epoch %s--training' %(epoch))
        for i, data in enumerate(tqdm(data_loader)):
            noisy, clean = data
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.optimizer.zero_grad() 

            noisy_stft = torch.stft(noisy, args.feature.n_fft, args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center=True, return_complex=False) #[Bs, F, T, 2]            
            noisy_stft = noisy_stft.permute(0, 3, 2, 1).contiguous() #[Bs, 2, T, F]
            
            #-------------spectrogram predition-----------------------
            enhan_stft = model(noisy_stft) #[B, 2, T, F]
            #------------------------------------         
            
            enhan_s = iSTFT_module_1_7(n_fft=args.feature.n_fft, hop_length=args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center = True, length = noisy.shape[-1])(enhan_stft)
            clean = clean[:,:enhan_s.shape[-1]]
            
            loss, _ = Loss(enhan_s, clean)

            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step() 
            train_loss.append(loss.cpu().detach().numpy())
        train_loss = np.mean(train_loss) 

        '''eval'''
        model.eval()
        
        ##------------------Test on valid testset---------------------
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader_valid)):
                noisy, clean = data
                noisy = noisy.to(device)
                clean = clean.to(device)

                noisy_stft = torch.stft(noisy, args.feature.n_fft, args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center=True, return_complex=False) #[Bs, F, T, 2]            
                noisy_stft = noisy_stft.permute(0, 3, 2, 1).contiguous() #[Bs, 2, T, F]
                
                #-------------spectrogram predition-----------------------
                enhan_stft = model(noisy_stft) #[B, 2, T, F]
                #------------------------------------         
                
                enhan_s = iSTFT_module_1_7(n_fft=args.feature.n_fft, hop_length=args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center = True, length = noisy.shape[-1])(enhan_stft)
                clean = clean[:,:enhan_s.shape[-1]]
                
                loss, snr_loss = Loss(enhan_s, clean)

                valid_loss.append(loss.cpu().detach().numpy())
                snr_valid_loss.append(snr_loss.cpu().detach().numpy())
            valid_loss = np.mean(valid_loss) 
            snr_valid_loss = np.mean(snr_valid_loss)

 
        print('====> Epoch: {} train loss: {} val loss: {} snr loss:{}'.format(
                  epoch+1, train_loss, valid_loss, snr_valid_loss))
    
        torch.save(
                {'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.optimizer.state_dict()},
               os.path.join(args.log.ckpt_dir, 'model_epoch_{}_trainloss_{:.8f}_validloss_{:.8f}_snrloss_{:4f}.pth'.format(epoch+1, train_loss, valid_loss, snr_valid_loss)))


if __name__ == '__main__':
    
    args = OC.load('config_dptfsnet.yaml')
    train(args)
