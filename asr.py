# -*- coding: utf-8 -*-

import numpy as np, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import sklearn
import math
from scipy.io import wavfile
import sys

#and in the training set to apply Fir Filter
#segmentation into the training set file as well
def fir_band_pass(samples, fs, fL, fH, NL, NH, outputType):
    # Referece: https://fiiir.com

    fH = fH / fs
    fL = fL / fs

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2.))
    hlpf *= np.blackman(NH)
    hlpf /= np.sum(hlpf)
    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2.))
    hhpf *= np.blackman(NL)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((NL - 1) / 2)] += 1
    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)

    return s

def preprocessing (filename):
    
    #sampling at 8000 --> input
    y,s = librosa.load(filename,sr=8000) #alex/4digit.wav
    
    #input signal 
    ipd.Audio(y,rate=s)
    
    #duration of input signal
    dur = librosa.core.get_duration(y,s)
    print("Original audio signal duration: ",dur)
    
    #plot waveplot
    plt.figure(1)
    plt.title('Waveform')
    librosa.display.waveplot(y,s)
    
    #display sectrogram
    Y = librosa.stft(y)
    Yto_db = librosa.amplitude_to_db(abs(Y))
    plt.figure(2)
    plt.title('Spectrograph')
    librosa.display.specshow(Yto_db,sr=s,x_axis='time',y_axis='hz')
    
    #filtered signal --> FIR filter
    y = fir_band_pass(y,s,200,4000,100,100,np.float32)
    y = y*2 #sound amplification
    
    #output of filtered signal
    wavfile.write('filtered.wav',s, y)
    dur = librosa.core.get_duration(y,s)
    
    print('New signal sound duration after filtering: ', dur)
    
    #pass rate from 0
    zero_cros_rate = librosa.feature.zero_crossing_rate(y,frame_length,frame_step)[0]
    #pass rate from 0
    zero_cros_rate = zero_cros_rate*100
    
    #short time energy
    energy_of_signal = np.array([sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y),frame_step)])
    #logarithm of short time energy
    logEnergy = np.array([math.log(energy_of_signal[i])
    for i in range(0, len(energy_of_signal))])

    #average pass rate from 0 for the first 10 frames
    zcavg= np.mean(zero_cros_rate[:10])
    #average logarithmic energy value for the first 10 frames
    eavg = np.mean(logEnergy[:10]) 

    #standard deviation of logarithmic energy
    esig = np.std(logEnergy[:10])
    #standard deviation of pass rate from 0
    zcsig = np.std(zero_cros_rate[:10])
    
    plt.figure(3)
    plt.plot(logEnergy)
    plt.xlabel('Frames')
    plt.ylabel('Short time energy logarithm')
    
    plt.figure(4)
    plt.plot(zero_cros_rate)
    plt.xlabel('Frames')
    plt.ylabel('Pass rate from 0')
    
    return y

        
#input signal    
def segmentation_multiple_digits(y,frame_length,frame_step,s):
    
    dur = librosa.get_duration(y,sr=s)
    #reverse audio
    y_rev = y[::-1]
    
    #onset detect
    onset_frames = librosa.onset.onset_detect(y, sr=s, hop_length=frame_length, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=s, hop_length=frame_length)
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=frame_length)
    
    onset_rev_frames = librosa.onset.onset_detect(y_rev, sr=s, hop_length=frame_length, backtrack=True)
    onset_rev_times = librosa.frames_to_time(onset_rev_frames, sr=s, hop_length=frame_length)
    #onset_rev_samples = librosa.frames_to_samples(onset_rev_frames, hop_length=frame_length)
    
    i=0
    while (i < len(onset_rev_times)-1):
        onset_rev_times[i] = dur - onset_rev_times[i]
        i+=1
    
    onset_rev_times = sorted(onset_rev_times)  
    
    i=0
    while (i < len(onset_rev_times)-1):
        if(onset_rev_times[i+1] - onset_rev_times[i] < 1):
            onset_rev_times = np.delete(onset_rev_times, i)
            i-=1
        i+=1;
        
            
    i=0
    while (i < len(onset_times)-1):
        if(onset_times[i+1] - onset_times[i] < 1):
            onset_times = np.delete(onset_times, i+1)
            onset_frames = np.delete(onset_frames, i+1)
            onset_samples = np.delete(onset_samples, i+1)
            i = i-1
        i=i+1;

    merged_onset_times = [*onset_times, *onset_rev_times]
    merged_onset_times = sorted(merged_onset_times)
        
    onset_samples = librosa.time_to_samples(merged_onset_times,sr=s)
    
    #spectrograph with detected onset spots
    plt.figure(5)
    plt.title('Spectroscopy with points resulting from onset')
    Y = librosa.stft(y)
    Yto_db = librosa.amplitude_to_db(abs(Y))
    librosa.display.specshow(Yto_db,sr=s,x_axis='time',y_axis='hz')
    plt.vlines(merged_onset_times, 0, 10000, color='k')
    
    i=0
    #number of valid digits from onset detection
    numbSongs=0
    song = {}
    while (i < len(onset_samples)):
        if i == len(onset_samples)-1 and len(onset_samples)%2 == 1:
            song[numbSongs] = y[onset_samples[i-1]:onset_samples[i]]#ipd.Audio(y[onset_samples[i]:onset_samples[i+1]],rate = s)            
        else:
            song[numbSongs] = y[onset_samples[i]:onset_samples[i+1]]#ipd.Audio(y[onset_samples[i]:onset_samples[i+1]],rate = s)
        numbSongs+=1
        i+=2
    #song[numbSongs] = y[onset_samples[-1]:]#ipd.Audio(y[onset_samples[-1]:],rate = s)
    
    print('Total digits: ',len(song))
    
    return song

def training_set(labels,where,who):
    j=0
    signals={}
    #for i in range(len(labels)):
    #0-9 digits that exist
	for i in range(10):
        for name in who:
            #from db
            y,s = librosa.load(where+'/{}'.format(i)+name+'.wav',sr = 8000)
            signals[j] = y
            #print(where+'/{}'.format(i)+name+'.wav')
            j+=1
    
    return signals

def cross_validation(labels):
	from sklearn.model_selection import train_test_split

	train, test = train_test_split(labels,test_size=0.3,shuffle = True) 


	D = np.ones((len(labels),len(labels))) *-1

	score = 0.0
	for i in range(len(test)):
		x = mfccs[i]
		dmin,jmin = math.inf,-1
			
		for j in range(len(train)):
			y = mfccs[j]
			d = D[i,j]
				
			if d.all() == -1:
				d = librosa.core.dtw(x,y,metric='euclidean',backtrack=True)
					
			if d.all()<dmin:
				dmin = d
				jmin=j
					
		score += 1.0 if (labels[i] == labels[jmin]) else 0.0
         
	print('Rec rate {}%'.format(100.* score/len(test)))

def recognition(digits,s,frame_length,syn_ekp):

    #for figures
    k=6
    j=0
    #for each input digit I want to compare it with the training set
    while j<len(digits):
        #compute mfcc for INPUT SIGNAL. (each digit from input)
        mfcc_input = librosa.feature.mfcc(digits[j], s,hop_length=frame_length, n_mfcc=13) #number of mfcc set to 13
        #logarithmisi twn features
        mfcc_input_mag = librosa.amplitude_to_db(abs(mfcc_input))
        
        '''#MFCC Plot
        plt.figure(k)
        k+=1
        plt.title('MFCC of ' +str(j+1)+ ' digit')
        librosa.display.specshow(mfcc_input,x_axis='time')
        plt.colorbar()
        plt.tight_layout()
        '''
        Dnew = []
        mfccs= []
        #0-9 from training set
        for i in range(len(syn_ekp)):
             syn_ekp[i] = fir_band_pass(syn_ekp[i],s,200,4000,100,100,np.float32)
             #MFCC for each digit from the training set
             mfcc = librosa.feature.mfcc(syn_ekp[i],s,hop_length=80,n_mfcc=13)
             #logarithm of the features ADDEED
             mfcc_mag = librosa.amplitude_to_db(abs(mfcc))
             #apply dtw
             D,wp =librosa.core.dtw(X=mfcc_input_mag,Y=mfcc_mag,metric='euclidean',backtrack=True)
             #make a list with minimum cost of each digit
             Dnew.append(D[-1,-1])
             #make a list with all mfccs, for each digit, in order to show the optimal path for recognized digit
             mfccs.append(mfcc_mag)
        
        #index of MINIMUM COST
        index = Dnew.index(min(Dnew))
        
        #show similarity line
        plt.figure(k)
        k+=1
        D,wp =librosa.core.dtw(X=mfcc_input_mag,Y=mfccs[index],metric='euclidean',backtrack=True)
        librosa.display.specshow(D, x_axis='frames', y_axis='frames')
        plt.title('Database excerpt')
        plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
        plt.legend()
        
        
        print (str(j+1) + ' Recognized Digit: '+labels[index])
        #print ('Score matrix from DTW: '+ Dnew)
        
        j+=1
    
s = 8000
#frame_duration-length
L = 0.03
#frame_step
R = 0.01
#frame length and sliding step
frame_length, frame_step = round(L * s), round(R * s)  # Convert from seconds to samples


y = preprocessing(input('Please enter a filename for input signal: '))
digits = segmentation_multiple_digits(y,frame_length,frame_step,s)

#naming training set files
with  open("tags.txt") as f:
    labels = np.array([l.replace('\n','') for l in f.readlines()])

t_set =  training_set(labels,'ekp',['a','p','n'])
recognision(digits,s,frame_length,t_set)
cross_validation(labels)
