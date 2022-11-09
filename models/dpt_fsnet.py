from operator import index
import torch
from torch import nn
import numpy as np
import math
from torch.nn import init

scm = np.load('./models/filter_banks/1200fft_256scm.npy').astype(np.float32)
erb = np.load('./models/filter_banks/1200fft_256erb.npy').astype(np.float32)
inv_erb = np.load('./models/filter_banks/1200fft_256inverb.npy').astype(np.float32)

class Encoder(nn.Module):
    def __init__(self, feature_dim=64, hidden_channel=64):
        '''
        区分一下hidden channel和最终的feature_dim，以方便缩减模型
        LN换BN
        '''
        # 1 x 1 conv
        super(Encoder, self).__init__()
        self.conv1x1 = nn.Conv2d(2, hidden_channel, (1,1))
        self.norm_0 = nn.BatchNorm2d(hidden_channel)
        self.act_0 = nn.PReLU()
        # dilated dense block
        self.conv_1 = nn.Conv2d(hidden_channel, hidden_channel, (2,3), padding=(1,1))
        self.norm_1 = nn.BatchNorm2d(hidden_channel)
        self.act_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(2 * hidden_channel, hidden_channel, (2,3), padding=(1,2), dilation=2)
        self.norm_2 = nn.BatchNorm2d(hidden_channel)
        self.act_2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(3 * hidden_channel, hidden_channel, (2,3), padding=(2,4), dilation=4)
        self.norm_3 = nn.BatchNorm2d(hidden_channel)
        self.act_3 = nn.PReLU()
        self.conv_4 = nn.Conv2d(4 * hidden_channel, feature_dim, (2,3), padding=(4,8), dilation=8)
        self.norm_4 = nn.BatchNorm2d(feature_dim)
        self.act_4 = nn.PReLU()
        
    def forward(self, x):
        '''
        input x: [bs, 2, T, F]
        '''
        x = self.act_0(self.norm_0(self.conv1x1(x)))
        layer_output = self.act_1(self.norm_1(self.conv_1(x)[:,:,:-1,:]))
        x = torch.cat([x, layer_output], dim=1) # -> 2c
        layer_output = self.act_2(self.norm_2(self.conv_2(x)))
        x = torch.cat([x, layer_output], dim=1) # -> 3c
        layer_output = self.act_3(self.norm_3(self.conv_3(x)))
        x = torch.cat([x, layer_output], dim=1) # -> 4c
        layer_output = self.act_4(self.norm_4(self.conv_4(x))) # -> c_output
        return layer_output
    
    
class Decoder(nn.Module):
    def __init__(self, feature_dim=64, hidden_channel=64):
        '''
        区分一下hidden channel和最终的feature_dim，以方便缩减模型
        LN换BN
        '''
        super(Decoder, self).__init__()
        # dilated dense block
        self.conv_1 = nn.Conv2d(feature_dim, hidden_channel, (2,3), padding=(1,1))
        self.norm_1 = nn.BatchNorm2d(hidden_channel)
        self.act_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(feature_dim + hidden_channel, hidden_channel, (2,3), padding=(1,2), dilation=2)
        self.norm_2 = nn.BatchNorm2d(hidden_channel)
        self.act_2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(feature_dim + 2 * hidden_channel, hidden_channel, (2,3), padding=(2,4), dilation=4)
        self.norm_3 = nn.BatchNorm2d(hidden_channel)
        self.act_3 = nn.PReLU()
        self.conv_4 = nn.Conv2d(feature_dim + 3 * hidden_channel, hidden_channel, (2,3), padding=(4,8), dilation=8)
        self.norm_4 = nn.BatchNorm2d(hidden_channel)
        self.act_4 = nn.PReLU()
        # 1 x 1 conv
        self.conv1x1 = nn.Conv2d(hidden_channel, 2, (1,1))
        self.norm_0 = nn.BatchNorm2d(2)
        self.act_0 = nn.PReLU()
        
    def forward(self, x):
        '''
        input x: [bs, C, T, F]
        '''
        layer_output = self.act_1(self.norm_1(self.conv_1(x[:,:,:-1,:])))
        x = torch.cat([x, layer_output], dim=1) # -> 2c
        layer_output = self.act_2(self.norm_2(self.conv_2(x)))
        x = torch.cat([x, layer_output], dim=1) # -> 3c
        layer_output = self.act_3(self.norm_3(self.conv_3(x)))
        x = torch.cat([x, layer_output], dim=1) # -> 4c
        layer_output = self.act_4(self.norm_4(self.conv_4(x))) # -> c_output
        x = self.act_0(self.norm_0(self.conv1x1(layer_output)))
        return x
    
    
class AttentionMask(nn.Module):
    
    def __init__(self, causal):
        super(AttentionMask, self).__init__()
        self.causal = causal
        
    def lower_triangular_mask(self, shape):
        '''
        

        Parameters
        ----------
        shape : a tuple of ints

        Returns
        -------
        a square Boolean tensor with the lower triangle being False

        '''
        row_index = torch.cumsum(torch.ones(size=shape), dim=-2)
        col_index = torch.cumsum(torch.ones(size=shape), dim=-1)
        return torch.lt(row_index, col_index)  # lower triangle:True, upper triangle:False
    
    def merge_masks(self, x, y):
        
        if x is None: return y
        if y is None: return x
        return torch.logical_and(x, y)
        
        
    def forward(self, inp):
        #input (bs, L, ...)
        max_seq_len = inp.shape[1]
        if self.causal ==True:
            causal_mask = self.lower_triangular_mask([max_seq_len, max_seq_len])      #(L, l)
            return causal_mask
        else:
            return torch.zeros(size=(max_seq_len, max_seq_len), dtype=torch.float32)
        
class MHAblock_GRU(nn.Module):
    
    def __init__(self, d_model, d_ff, n_heads, bidirectional=False):        
        super(MHAblock_GRU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        self.MHA = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, bias=False)
        self.norm_1 = nn.LayerNorm([self.d_model], eps=1e-6)
        
        self.gru= nn.GRU(self.d_model, self.d_ff, batch_first=True, bidirectional=bidirectional)
        self.act = nn.ReLU()
        self.fc = nn.Conv1d(self.d_ff, self.d_model, 1)
        self.norm_2 = nn.LayerNorm([self.d_model], eps=1e-6)
        
        
    def forward(self, x, att_mask):
        
        # x input: (bs, L, d_model)
        x = x.permute(1,0,2).contiguous() #(L, bs, d_model)
        layer_1,_ = self.MHA(x, x, x, attn_mask=att_mask, need_weights=False) #(L, bs, d_model)
        layer_1 = torch.add(x, layer_1).permute(1,0,2).contiguous() #(L, bs, d_model) ->  (bs, L, d_model)
        layer_1 = self.norm_1(layer_1) #(bs, L, d_model)
        
        layer_2,_ = self.gru(layer_1) #(bs, L, d_ff) 
        layer_2 = self.act(layer_2) #(bs, L, d_ff)
        layer_2 = self.fc(layer_2.permute(0,2,1)).permute(0,2,1).contiguous() #(bs, L, d_ff) -> (bs, d_ff, L)  -> (bs, d_model, L) -> (bs, L, d_model)
        layer_2 = torch.add(layer_1, layer_2)
        layer_2 = self.norm_2(layer_2)
        return layer_2
    
class DualpathMHA(nn.Module):
    
    def __init__(self, feature_dim, n_heads=4, group=4):
        '''
        if the model is causal, bidiractional is False for inter mha
        ? group norm 的 group数量
        '''
        super(DualpathMHA, self).__init__()
        self.intra_mha = MHAblock_GRU(feature_dim, 4*feature_dim, n_heads)
        self.norm_1 = nn.GroupNorm(group, feature_dim)
        
        self.inter_mha = MHAblock_GRU(feature_dim, 4*feature_dim, n_heads)
        self.norm_2 = nn.GroupNorm(group, feature_dim)
        
    def forward(self, x, att_mask_1, att_mask_2):
        '''
        input x: [bs, C', T, F]
        '''
        BS, C, T, F = x.shape
        intra_output = x.permute(0, 2, 3, 1).contiguous().view(BS*T, F, C) #->[bs, T, F, C'] -> [bs*T, F, C']
        intra_output = self.intra_mha(intra_output, att_mask_1)
        intra_output = self.norm_1(intra_output.permute(0, 2, 1).contiguous()) #-> [bs*T, C', F]
        intra_output = intra_output.view(BS, T, C, F).permute(0, 2, 1, 3).contiguous() #[bs, T, C', F] -> [bs, C', T, F]
        
        x = x + intra_output
        
        inter_output = x.permute(0, 3, 2, 1).contiguous().view(BS*F, T, C) #->[bs, F, T, C'] -> [bs*F, T, C']
        inter_output = self.inter_mha(inter_output, att_mask_2)
        inter_output = self.norm_2(inter_output.permute(0, 2, 1).contiguous()) # -> [bs*F, C', T]
        inter_output = inter_output.view(BS, F, C, T).permute(0, 2, 3, 1).contiguous() #[bs, F, T, C'] -> [bs, C', T, F] 
        
        x = x + inter_output
        
        return x
    
class DPTPM(nn.Module):
    
    def __init__(self, repeat=4, feature_dim=64, n_heads=4, group=4, causal=False):
        
        super(DPTPM, self).__init__()
        self.causal = causal
        # 1 x 1 conv to halve the feature dimension
        self.conv1x1_1 = nn.Conv2d(feature_dim, feature_dim//2, (1,1))
        self.act_1 = nn.PReLU()
        # dual path transformer
        self.dualpathmha_list = nn.ModuleList([DualpathMHA(feature_dim//2, n_heads, group) for _ in range(repeat)])
        # 1 x 1 conv to double the feature dimension
        self.conv1x1_2 = nn.Conv2d(feature_dim//2, feature_dim, (1,1))
        self.act_2 = nn.PReLU()
        # 1 x 1 gated conv to smooth the output
        self.conv1x1_3 = nn.Conv2d(feature_dim, feature_dim, (1,1))  
        self.conv1x1_gate = nn.Conv2d(feature_dim, feature_dim, (1,1))  
        self.sigmoid = nn.Sigmoid()
        self.act_3 = nn.PReLU()
        
    def forward(self, x):
        '''
        input x: [bs, C, T, F]
        '''       
        x = self.act_1(self.conv1x1_1(x))
        
        BS, C, T, F = x.shape
        
        att_mask_1 = AttentionMask(causal=False)(x.permute(0, 2, 3, 1).contiguous().view(BS*T, F, C)).to(x.device)
        att_mask_2 = AttentionMask(causal=self.causal)(x.permute(0, 3, 2, 1).contiguous().view(BS*F, T, C)).to(x.device)
            
        for dualpathmha in self.dualpathmha_list:
            x = dualpathmha(x, att_mask_1, att_mask_2)
            
        x = self.act_2(self.conv1x1_2(x))
        
        x_gated = self.conv1x1_3(x)
        gate = self.sigmoid(self.conv1x1_gate(x))
        x_gated = x_gated * gate
        x = self.act_3(x_gated)
        
        return x
    
class DPT_FSNet(nn.Module):
    
    def __init__(self):
        super(DPT_FSNet, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.DPTPM = DPTPM()
        
    def forward(self, x):
        
        y = self.encoder(x)
        y = self.DPTPM(y)
        y = self.decoder(y)
        
        y_real = y[:,0,:,:] * x[:,0,:,:] - y[:,1,:,:] * x[:,1,:,:]
        y_imag = y[:,0,:,:] * x[:,1,:,:] + y[:,1,:,:] * x[:,0,:,:]
        y = torch.stack([y_real, y_imag], dim=1)
        return y

class DPT_FSNet_SCM(nn.Module):
    
    def __init__(self, num_fft=1200, num_freqs=256):
        super(DPT_FSNet_SCM, self).__init__()
        self.bin_width = 48000 / num_fft
        num_freqs_orig = num_fft // 2 + 1
        n_low = int(num_fft / 48000 * 5000)
        
        self.flc_low =  nn.Linear(num_freqs_orig, n_low, bias=False)
        self.flc_low.weight = nn.Parameter(torch.from_numpy(scm[:n_low, :]), requires_grad=False)

        self.weight_high = nn.Parameter(torch.from_numpy(scm[n_low:, :]), requires_grad=True)
        index_scm = np.argmax(scm[n_low:,:],axis=1)
        self.weight_list = []
        self.padding_list = []
        for i in range(num_freqs - n_low):
            start_index, end_index = self.bandwith_cal(index_scm[i], num_freqs_orig)
            weight = scm[i, start_index:end_index]
            self.weight_list.append(torch.from_numpy(np.ones_like(weight)))
            pad_mat = nn.functional.pad(self.weight_list[-1],[start_index, num_freqs_orig - end_index])
            self.padding_list.append(pad_mat)
        self.mask = torch.stack(self.padding_list, axis = 0)
        
        
        self.inv_flc = nn.Linear(num_freqs, num_freqs_orig, bias=False)
        self.inv_flc.weight = nn.Parameter(torch.from_numpy(scm.T), requires_grad=True)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.DPTPM = DPTPM(repeat=3,causal=True)

    def bandwith_cal(self, k, num_freqs_orig, bandwidth_ratio=0.5):
        f = (k * self.bin_width ) / 1000
        erb_width = 6.23 * (f ** 2) + 93.99 * f + 28.52
        start_index = k - int(bandwidth_ratio * erb_width / self.bin_width)
        end_index = k + int(bandwidth_ratio * erb_width / self.bin_width)
        return np.maximum(0, start_index), np.minimum(num_freqs_orig, end_index)

    def forward(self, x):
        '''
        input x: [bs, 2, T, F]
        '''
        y_low = self.flc_low(x)
        self.weight_high = self.weight_high.to(x.device)
        self.mask = self.mask.to(x.device)
        y_high = x @ (self.weight_high * self.mask).T
        y = torch.cat([y_low, y_high], dim=-1)
        
        y = self.encoder(y)
        y = self.DPTPM(y)
        y = self.decoder(y)
        
        y = self.inv_flc(y)
        
        y_real = y[:,0,:,:] * x[:,0,:,:] - y[:,1,:,:] * x[:,1,:,:]
        y_imag = y[:,0,:,:] * x[:,1,:,:] + y[:,1,:,:] * x[:,0,:,:]
        y = torch.stack([y_real, y_imag], dim=1)
        return y

class DPT_FSNet_ERB(nn.Module):
    
    def __init__(self, num_fft=1200, num_freqs=256):
        super(DPT_FSNet_ERB, self).__init__()
        num_freqs_orig = num_freqs // 2 + 1
        
        self.erb =  nn.Linear(num_freqs_orig, num_freqs, bias=False)
        self.erb.weight = nn.Parameter(torch.from_numpy(erb), requires_grad=False)

        self.inv_erb = nn.Linear(num_freqs, num_freqs_orig, bias=False)
        self.inv_erb.weight = nn.Parameter(torch.from_numpy(inv_erb), requires_grad=False)
        
        self.encoder = Encoder()#(hidden_channel=32, feature_dim=48)
        self.decoder = Decoder()#(hidden_channel=32, feature_dim=48)
        self.DPTPM = DPTPM(repeat=3,causal=True)#(repeat=2, feature_dim=48)
        
    def forward(self, x):
        '''
        input x: [bs, 2, T, F]
        '''
        y = self.erb(x)
        
        y = self.encoder(y)
        y = self.DPTPM(y)
        y = self.decoder(y)
        
        y = self.inv_erb(y)
        
        y_real = y[:,0,:,:] * x[:,0,:,:] - y[:,1,:,:] * x[:,1,:,:]
        y_imag = y[:,0,:,:] * x[:,1,:,:] + y[:,1,:,:] * x[:,0,:,:]
        y = torch.stack([y_real, y_imag], dim=1)
        return y







    