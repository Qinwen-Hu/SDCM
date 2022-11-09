import os
from omegaconf import OmegaConf as OC

import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import soundfile as sf

from Dataloader_vctk_demand import Dataset
from models.fullsubnet_plus import FullSubNet_Plus_SCM, FullSubNet_Plus_ERB
from audio_zen.acoustics.feature import mag_phase
from signal_processing import iSTFT_module_1_8, iSTFT_module_1_7
# use iSTFT_module_1_7 with pytorch version < 1.8, use iSTFT_module_1_8 with pytorch version >= 1.8

torch.set_default_tensor_type(torch.FloatTensor)


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
    model = FullSubNet_Plus_SCM(weight_init=False) 
    model = torch.nn.DataParallel(model, device_ids=args.train.device_ids)
    model = model.to(device)

    '''optimizer & lr_scheduler'''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.train.lr_reduce_ratio, patience=args.train.lr_reduce_patience, verbose=True)

    '''load train data'''
    dataset = Dataset(length_in_seconds=args.train.length_in_seconds, random_start_point=True, train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, drop_last=True)
    dataset_valid = Dataset(length_in_seconds=args.train.length_in_seconds, random_start_point=True, train=False)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.train.batch_size, shuffle=True, num_workers=args.train.num_workers, drop_last=True)


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
            optimizer.zero_grad() 

            noisy_stft = torch.stft(noisy, args.feature.n_fft, args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center=True, return_complex=True) #[B, F, T]            
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
            enh_stft = torch.cat((enh_real, enh_imag), dim=1) #[B, 2, F, T]
            enh_stft = enh_stft.permute(0, 1, 3, 2).contiguous() #[B, 2, T, F]
            enhan_s = iSTFT_module_1_7(n_fft=args.feature.n_fft, hop_length=args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center = True, length = clean.shape[-1])(enh_stft)
            clean = clean[:,:enhan_s.shape[-1]]
            
            loss, _ = Loss(enhan_s, clean)

            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
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

                noisy_stft = torch.stft(noisy, args.feature.n_fft, args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center=True, return_complex=True) #[B, F, T]            
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
                enh_stft = torch.cat((enh_real, enh_imag), dim=1) #[B, 2, F, T]
                enh_stft = enh_stft.permute(0, 1, 3, 2).contiguous() #[B, 2, T, F]
                enhan_s = iSTFT_module_1_7(n_fft=args.feature.n_fft, hop_length=args.feature.n_hop, win_length=args.feature.n_win, window=WINDOW, center = True, length = clean.shape[-1])(enh_stft)
                clean = clean[:,:enhan_s.shape[-1]]
                
                loss, snr_loss = Loss(enhan_s, clean)

                valid_loss.append(loss.cpu().detach().numpy())
                snr_valid_loss.append(snr_loss.cpu().detach().numpy())
            valid_loss = np.mean(valid_loss) 
            snr_valid_loss = np.mean(snr_valid_loss)

        scheduler.step(valid_loss)  
        print('====> Epoch: {} train loss: {} val loss: {} snr loss:{}'.format(
                  epoch+1, train_loss, valid_loss, snr_valid_loss))
    
        torch.save(
                {'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
               os.path.join(args.log.ckpt_dir, 'model_epoch_{}_trainloss_{:.8f}_validloss_{:.8f}_snrloss_{:4f}.pth'.format(epoch+1, train_loss, valid_loss, snr_valid_loss)))


if __name__ == '__main__':
    
    args = OC.load('config_fullsubp.yaml')
    train(args)
