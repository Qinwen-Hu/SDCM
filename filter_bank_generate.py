import numpy as np
# import librosa
import matplotlib.pyplot as plt
# import soundfile as sf
import python_speech_features as psf
# from DataLoader_from_csv import Dataset, collate_fn
# import torch
from tqdm import tqdm

'''
scm filter banks
'''
def hz2mel1(hz):
    if type(hz) == np.ndarray:
        for i in range(hz.shape[0]):
            hz[i] = hz2mel1(hz[i])
        return hz
    else:
        if hz < 5000:
            return hz
        else:
            return 2500*np.log((hz-2500)/2500)+5000

def mel2hz1(mel):
    
    if type(mel) == np.ndarray:
        for i in range(mel.shape[0]):
            mel[i] = mel2hz1(mel[i])
        return mel
    else:
        if mel<5000:
            return mel
        else:
            return 2500*np.exp(mel/2500 - 2)+2500
    
def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def mk_mel_matrix(n_filt, n_fft, samplerate, lowfreq, highfreq):
    '''
    Method for getting mel matrix
    '''
    mel_matrix = psf.get_filterbanks(nfilt = n_filt, nfft = n_fft, samplerate = samplerate, lowfreq = lowfreq, highfreq=highfreq)
    return mel_matrix

def get_filterbanks_scm(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a SCM-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel1(lowfreq)
    highmel = hz2mel1(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz1(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def get_bin(lowfreq=50,highfreq=24000,nfilt=256,nfft=1200,samplerate=48000):
    lowmel = hz2mel1(lowfreq)
    highmel = hz2mel1(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz1(melpoints)/samplerate)
    return bin[1:-1]

def inter(mat):
    N,F = mat.shape
    freq = np.zeros(N,dtype=np.int64)
    
    bin = get_bin()
    for i in range(N):
        freq[i] = np.argmax(mat[i])
        if freq[i] == 0:
            mat[i,int(bin[i])] = 1
    return freq,bin,mat

'''
erb filter banks
'''
def hz2erb(hz):
    if type(hz) == np.ndarray:
        for i in range(hz.shape[0]):
            hz[i] = hz2erb(hz[i])
        return hz
    else:
        return 11.17268 * np.log( 1 + (46.06538 * hz)/( hz + 14678.49))
    
def erb2hz(erb):
    if type(erb) == np.ndarray:
        for i in range(erb.shape[0]):
            erb[i] = erb2hz(erb[i])
        return erb
    else:
        return 676170.4 / (47.06538 - np.exp(0.08950404 * erb)) - 14678.49

def get_filterbanks_erb(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a ERB-filterbanks. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects erb spacing.
    :param lowfreq: lowest band edge of erb filters, default 0 Hz
    :param highfreq: highest band edge of erb filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowerb = hz2erb(lowfreq) # compute the lowest mel value
    higherb = hz2erb(highfreq) # compute the highest mel value
    erbpoints = np.linspace(lowerb,higherb,nfilt+2) # compute the mel frequencies
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*erb2hz(erbpoints)/samplerate)    

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


if __name__=='__main__':
    
    '''
    compute scm filter banks
    '''
    num_fft = 1536             # number of fft points
    num_bins = num_fft//2 + 1   # number of frequency points before compression
    num_comp = 256              # number of frequency points after compression
    freq_thre = 5000            # threshold frequency 
    num_high = num_comp - int(num_fft / 48000 * freq_thre)        # number of high points
    scm = np.zeros([num_comp, num_bins])
    mat = get_filterbanks_scm(num_high, num_fft, 48000, freq_thre, 24000)
    scm[:num_comp-num_high,:num_comp-num_high] = np.eye(num_comp-num_high)
    scm[num_comp-num_high:,:] = mat
    plt.imshow(scm)
    plt.savefig('./models/filter_banks/{}fft_{}scm.png'.format(num_fft, num_comp),dpi=600, bbox_inches='tight')
    
    np.save('./models/filter_banks/{}fft_{}scm.npy'.format(num_fft, num_comp), scm)
    
    '''
    compute erb filter banks
    '''
    num_fft = 1536             # number of fft points
    num_bins = num_fft//2 + 1   # number of frequency points before compression
    num_comp = 256              # number of frequency points after compression

    erb = np.zeros([num_comp,num_bins])
    mat = get_filterbanks_erb(num_comp, num_fft, 48000, 0, 24000)
    plt.imshow(mat)
    plt.savefig('./models/filter_banks/{}fft_{}erb.png'.format(num_fft, num_comp), dpi=600, bbox_inches='tight')
    np.save('./models/filter_banks/{}fft_{}erb.npy'.format(num_fft, num_comp), mat)
    np.save('./models/filter_banks/{}fft_{}inverb.npy'.format(num_fft, num_comp), np.linalg.pinv(mat))