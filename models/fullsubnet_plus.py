import torch
from torch.nn import functional
from torch import nn
import numpy as np

from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
from audio_zen.model.module.attention_model import ChannelSELayer, ChannelECAlayer, ChannelCBAMLayer, \
    ChannelTimeSenseSELayer, ChannelTimeSenseAttentionSELayer, ChannelTimeSenseSEWeightLayer

scm = np.load('./models/filter_banks/1536fft_256scm.npy').astype(np.float32)
erb = np.load('./models/filter_banks/1536fft_256erb.npy').astype(np.float32)
inv_erb = np.load('./models/filter_banks/1536fft_256inverb.npy').astype(np.float32)
  
class FullSubNet_Plus_ERB(BaseModel):
    def __init__(self,
                 num_fft=1536,
                 num_freqs=256,
                 look_ahead=2,
                 sequence_model="LSTM",
                 fb_num_neighbors=0,
                 sb_num_neighbors=15,
                 fb_output_activate_function="ReLU",
                 sb_output_activate_function=False,
                 fb_model_hidden_size=512,
                 sb_model_hidden_size=384,
                 channel_attention_model="TSSE",
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=1,
                 output_size=2,
                 subband_num=1,
                 kersize=[3, 5, 10],
                 weight_init=True,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM", "TCN"), f"{self.__class__.__name__} only support GRU, LSTM and TCN."
        num_freqs_orig = num_fft // 2 + 1
        
        self.erb = nn.Linear(num_freqs_orig, num_freqs, bias=False)
        self.erb.weight = nn.Parameter(torch.from_numpy(erb), requires_grad=False)
        
        self.inv_erb = nn.Linear(num_freqs, num_freqs_orig, bias=False)
        self.inv_erb.weight = nn.Parameter(torch.from_numpy(inv_erb), requires_grad=False)
        
        if subband_num == 1:
            self.num_channels = num_freqs
        else:
            self.num_channels = num_freqs // subband_num + 1

        if channel_attention_model:
            if channel_attention_model == "SE":
                self.channel_attention = ChannelSELayer(num_channels=self.num_channels)
                self.channel_attention_real = ChannelSELayer(num_channels=self.num_channels)
                self.channel_attention_imag = ChannelSELayer(num_channels=self.num_channels)
            elif channel_attention_model == "ECA":
                self.channel_attention = ChannelECAlayer(channel=self.num_channels)
                self.channel_attention_real = ChannelECAlayer(channel=self.num_channels)
                self.channel_attention_imag = ChannelECAlayer(channel=self.num_channels)
            elif channel_attention_model == "CBAM":
                self.channel_attention = ChannelCBAMLayer(num_channels=self.num_channels)
                self.channel_attention_real = ChannelCBAMLayer(num_channels=self.num_channels)
                self.channel_attention_imag = ChannelCBAMLayer(num_channels=self.num_channels)
            elif channel_attention_model == "TSSE":
                self.channel_attention = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
                self.channel_attention_real = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
                self.channel_attention_imag = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
            else:
                raise NotImplementedError(f"Not implemented channel attention model {self.channel_attention}")

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.fb_model_real = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.fb_model_imag = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + 3 * (fb_num_neighbors * 2 + 1),
            output_size=output_size,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )
        self.subband_num = subband_num
        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band
        self.output_size = output_size

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag, noisy_real, noisy_imag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            noisy_real: [B, 1, F, T]
            noisy_imag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        batch_size = noisy_mag.shape[0]
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        noisy_real = functional.pad(noisy_real, [0, self.look_ahead])  # Pad the look ahead
        noisy_imag = functional.pad(noisy_imag, [0, self.look_ahead])  # Pad the look ahead
        
        #SCM layer
        noisy_input = torch.cat([noisy_mag, noisy_real, noisy_imag], dim=0)
        noisy_input = noisy_input.permute(0,1,3,2).contiguous() #[B*3, 1, T, F]
        noisy_input = self.erb(noisy_input)
        noisy_input = noisy_input.permute(0,1,3,2).contiguous() #[B*3, 1, F, T]
        noisy_mag, noisy_real, noisy_imag = torch.split(noisy_input, batch_size)
        
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        if self.subband_num == 1:
            fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
            fb_input = self.channel_attention(fb_input)
        else:
            pad_num = self.subband_num - num_freqs % self.subband_num
            # Fullband model
            fb_input = functional.pad(self.norm(noisy_mag), [0, 0, 0, pad_num], mode="reflect")
            fb_input = fb_input.reshape(batch_size, (num_freqs + pad_num) // self.subband_num,
                                        num_frames * self.subband_num)  # [B, subband_num, T]
            fb_input = self.channel_attention(fb_input)
            fb_input = fb_input.reshape(batch_size, num_channels * (num_freqs + pad_num), num_frames)[:, :num_freqs, :]
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Fullband real model
        fbr_input = self.norm(noisy_real).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fbr_input = self.channel_attention_real(fbr_input)
        fbr_output = self.fb_model_real(fbr_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Fullband imag model
        fbi_input = self.norm(noisy_imag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fbi_input = self.channel_attention_imag(fbi_input)
        fbi_output = self.fb_model_imag(fbi_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold the output of the fullband model, [B, N=F, C, F_f, T]
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                        num_frames)

        # Unfold the output of the fullband real model, [B, N=F, C, F_f, T]
        fbr_output_unfolded = self.unfold(fbr_output, num_neighbor=self.fb_num_neighbors)
        fbr_output_unfolded = fbr_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)

        # Unfold the output of the fullband imag model, [B, N=F, C, F_f, T]
        fbi_output_unfolded = self.unfold(fbi_output, num_neighbor=self.fb_num_neighbors)
        fbi_output_unfolded = fbi_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)

        # Unfold attention noisy input, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(fb_input.reshape(batch_size, 1, num_freqs, num_frames),
                                         num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1,
                                                        num_frames)

        # Concatenation, [B, F, (F_s + 3 * F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded, fbr_output_unfolded, fbi_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation. These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3),
                                 num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + 3 * (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, self.output_size, num_frames).permute(0, 2, 3, 1).contiguous() #[B, 2, T, F]
        sb_mask = sb_mask[:, :, self.look_ahead:, :]
        
        output = self.inv_erb(sb_mask).permute(0, 1, 3, 2).contiguous() # -> [B, 2, T, F0] -> [B, 2, F0, T]
        return output


class FullSubNet_Plus_SCM(BaseModel):
    def __init__(self,
                 num_fft=1536,
                 num_freqs=256,
                 look_ahead=2,
                 sequence_model="LSTM",
                 fb_num_neighbors=0,
                 sb_num_neighbors=15,
                 fb_output_activate_function="ReLU",
                 sb_output_activate_function=False,
                 fb_model_hidden_size=512,
                 sb_model_hidden_size=384,
                 channel_attention_model="TSSE",
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=1,
                 output_size=2,
                 subband_num=1,
                 kersize=[3, 5, 10],
                 weight_init=True,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM", "TCN"), f"{self.__class__.__name__} only support GRU, LSTM and TCN."
        num_freqs_orig = num_fft // 2 + 1
        self.bin_width = 48000 / num_fft
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
        
        if subband_num == 1:
            self.num_channels = num_freqs
        else:
            self.num_channels = num_freqs // subband_num + 1

        if channel_attention_model:
            if channel_attention_model == "SE":
                self.channel_attention = ChannelSELayer(num_channels=self.num_channels)
                self.channel_attention_real = ChannelSELayer(num_channels=self.num_channels)
                self.channel_attention_imag = ChannelSELayer(num_channels=self.num_channels)
            elif channel_attention_model == "ECA":
                self.channel_attention = ChannelECAlayer(channel=self.num_channels)
                self.channel_attention_real = ChannelECAlayer(channel=self.num_channels)
                self.channel_attention_imag = ChannelECAlayer(channel=self.num_channels)
            elif channel_attention_model == "CBAM":
                self.channel_attention = ChannelCBAMLayer(num_channels=self.num_channels)
                self.channel_attention_real = ChannelCBAMLayer(num_channels=self.num_channels)
                self.channel_attention_imag = ChannelCBAMLayer(num_channels=self.num_channels)
            elif channel_attention_model == "TSSE":
                self.channel_attention = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
                self.channel_attention_real = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
                self.channel_attention_imag = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
            else:
                raise NotImplementedError(f"Not implemented channel attention model {self.channel_attention}")

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.fb_model_real = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.fb_model_imag = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + 3 * (fb_num_neighbors * 2 + 1),
            output_size=output_size,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )
        self.subband_num = subband_num
        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band
        self.output_size = output_size

        if weight_init:
            self.apply(self.weight_init)
            
    def bandwith_cal(self, k, num_freqs_orig, bandwidth_ratio=0.5):
        f = (k * self.bin_width ) / 1000
        erb_width = 6.23 * (f ** 2) + 93.99 * f + 28.52
        start_index = k - int(bandwidth_ratio * erb_width / self.bin_width)
        end_index = k + int(bandwidth_ratio * erb_width / self.bin_width)
        return np.maximum(0, start_index), np.minimum(num_freqs_orig, end_index)

    def forward(self, noisy_mag, noisy_real, noisy_imag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            noisy_real: [B, 1, F, T]
            noisy_imag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        batch_size = noisy_mag.shape[0]
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        noisy_real = functional.pad(noisy_real, [0, self.look_ahead])  # Pad the look ahead
        noisy_imag = functional.pad(noisy_imag, [0, self.look_ahead])  # Pad the look ahead
        
        #SCM layer
        noisy_input = torch.cat([noisy_mag, noisy_real, noisy_imag], dim=0)
        noisy_input = noisy_input.permute(0,1,3,2).contiguous() #[B*3, 1, T, F]
        noisy_input_low = self.flc_low(noisy_input)
        self.weight_high = self.weight_high.to(noisy_mag.device)
        self.mask = self.mask.to(noisy_mag.device)
        noisy_input_high = noisy_input @ (self.weight_high * self.mask).T
        noisy_input = torch.cat([noisy_input_low, noisy_input_high], dim=-1)
        noisy_input = noisy_input.permute(0,1,3,2).contiguous() #[B*3, 1, F, T]
        noisy_mag, noisy_real, noisy_imag = torch.split(noisy_input, batch_size)
        
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        if self.subband_num == 1:
            fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
            fb_input = self.channel_attention(fb_input)
        else:
            pad_num = self.subband_num - num_freqs % self.subband_num
            # Fullband model
            fb_input = functional.pad(self.norm(noisy_mag), [0, 0, 0, pad_num], mode="reflect")
            fb_input = fb_input.reshape(batch_size, (num_freqs + pad_num) // self.subband_num,
                                        num_frames * self.subband_num)  # [B, subband_num, T]
            fb_input = self.channel_attention(fb_input)
            fb_input = fb_input.reshape(batch_size, num_channels * (num_freqs + pad_num), num_frames)[:, :num_freqs, :]
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Fullband real model
        fbr_input = self.norm(noisy_real).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fbr_input = self.channel_attention_real(fbr_input)
        fbr_output = self.fb_model_real(fbr_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Fullband imag model
        fbi_input = self.norm(noisy_imag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fbi_input = self.channel_attention_imag(fbi_input)
        fbi_output = self.fb_model_imag(fbi_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold the output of the fullband model, [B, N=F, C, F_f, T]
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                        num_frames)

        # Unfold the output of the fullband real model, [B, N=F, C, F_f, T]
        fbr_output_unfolded = self.unfold(fbr_output, num_neighbor=self.fb_num_neighbors)
        fbr_output_unfolded = fbr_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)

        # Unfold the output of the fullband imag model, [B, N=F, C, F_f, T]
        fbi_output_unfolded = self.unfold(fbi_output, num_neighbor=self.fb_num_neighbors)
        fbi_output_unfolded = fbi_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)

        # Unfold attention noisy input, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(fb_input.reshape(batch_size, 1, num_freqs, num_frames),
                                         num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1,
                                                        num_frames)

        # Concatenation, [B, F, (F_s + 3 * F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded, fbr_output_unfolded, fbi_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation. These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3),
                                 num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + 3 * (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, self.output_size, num_frames).permute(0, 2, 3, 1).contiguous() #[B, 2, T, F]
        sb_mask = sb_mask[:, :, self.look_ahead:, :]
        
        output = self.inv_flc(sb_mask).permute(0, 1, 3, 2).contiguous() # -> [B, 2, T, F0] -> [B, 2, F0, T]
        return output

