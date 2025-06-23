import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # Use seaborn's default style to make attractive graphs

# Plot nice figures using Python's "standard" matplotlib library
snd = parselmouth.Sound("vox2_2.wav")
plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")


def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

intensity = snd.to_intensity()
spectrogram = snd.to_spectrogram()
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_intensity(intensity)
plt.xlim([snd.xmin, snd.xmax])
plt.show() # or plt.savefig("spectrogram.pdf")

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

pitch = snd.to_pitch()
# If desired, pre-emphasize the sound fragment before calculating the spectrogram
pre_emphasized_snd = snd.copy()
pre_emphasized_snd.pre_emphasize()
spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show() # or plt.savefig("spectrogram_0.03.pdf")


pitch = snd.to_pitch()
pitch_values = pitch.selected_array['frequency']


pitch = snd.to_formant_burg()
f1 = pitch.get_value_at_time(1,4)
bw_f1 = pitch.get_bandwidth_at_time(1,4)

# Find all .wav files in a directory, pre-emphasize and save as new .wav and .aiff file
import parselmouth

import glob
import os.path

for wave_file in glob.glob("audio/*.wav"):
    print("Processing {}...".format(wave_file))
    s = parselmouth.Sound(wave_file)
    s.pre_emphasize()
    s.save(os.path.splitext(wave_file)[0] + "_pre.wav", 'WAV') # or parselmouth.SoundFileFormat.WAV instead of 'WAV'
    s.save(os.path.splitext(wave_file)[0] + "_pre.aiff", 'AIFF')
    
    




##############################################################################
##############################################################################
##############################################################################

import os
import numpy as np
from scipy.io import wavfile
import parselmouth 
from parselmouth.praat import call
from IPython.display import Audio
import matplotlib.pyplot as plt

fs, x = wavfile.read('vox2_2.wav')
plt.plot(x)
Audio(x,rate=fs)


def formants_praat(x, fs):
        f0min, f0max  = 75, 300
        sound = parselmouth.Sound(x, sampling_frequency=fs) # read the sound
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        formants = sound.to_formant_burg(time_step=0.010, maximum_formant=5000)
        
        f0_list, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list  = [],[],[],[],[],[],[],[],[]
        for t in formants.ts():
            print(t)             
          #if f0[t]:  
            #f0_list.append(f0[t])
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)
            f4 = formants.get_value_at_time(4, t)
            f1bw = formants.get_bandwidth_at_time(1, t)
            f2bw = formants.get_bandwidth_at_time(2, t)
            f3bw = formants.get_bandwidth_at_time(3, t)
            f4bw = formants.get_bandwidth_at_time(4, t)
            if np.isnan(f1): f1 = 0
            if np.isnan(f2): f2 = 0
            if np.isnan(f3): f3 = 0
            if np.isnan(f4): f4 = 0
            if np.isnan(f1bw): f1 = 0
            if np.isnan(f2bw): f2 = 0
            if np.isnan(f3bw): f3 = 0
            if np.isnan(f4bw): f4 = 0
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
            f1bw_list.append(f1bw)
            f2bw_list.append(f2bw)
            f3bw_list.append(f3bw)
            f4bw_list.append(f4bw)
            
        return f0, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list
    
f0, f1, f2, f3, f4,f1bw,f2bw,f3bw,f4bw = formants_praat(x,fs)

plt.plot(f0,'k')
#plt.plot(f1,'b')
#plt.plot(f2,'r')
#plt.plot(f3,'g')
#plt.plot(f4,'m')
#plt.legend(['f0','f1','f2','f3','f4'])
plt.legend(['f0'])
plt.grid(True)
plt.ylabel('formants(Hz)')    

f0_list, f1_list, f2_list, f3_list, f4_list  = [], [], [], [], []
for i in range(len(f0)):
   if f0[i] and f0[i] < np.mean(f0) + 50 and f0[i] > np.mean(f0) - 30: 
      f0_list.append(f0[i])
      f1_list.append(f1[i])
      f2_list.append(f2[i])
      f3_list.append(f3[i])
      f4_list.append(f4[i])

def snr(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return -1*np.log10(abs(np.where(sd == 0, 0, m/sd))) 
 
print(snr(x)) 

    
plt.subplot(2, 1, 1)
plt.plot(x) 
plt.subplot(2, 1, 2)      
plt.plot(f0_list,'k')
#plt.plot(f1_list,'b')
#plt.plot(f2_list,'r')
#plt.plot(f3_list,'g')
#plt.plot(f4_list,'m')
#plt.legend(['f0','f1','f2','f3','f4'])
plt.grid(True)
plt.ylabel('F(Hz)')  




###############################################################################


###############################################################################
import os
import numpy as np
from scipy.io import wavfile
import parselmouth 
from parselmouth.praat import call
from IPython.display import Audio
import matplotlib.pyplot as plt
import glob

def formants_praat(x, fs):
        f0min, f0max  = 100, 350
        sound = parselmouth.Sound(x, sampling_frequency=fs) # read the sound
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        formants = sound.to_formant_burg(time_step=0.010, max_number_of_formants=4, maximum_formant=5000,window_length=0.025,pre_emphasis_from=50)
        
        f0_list, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list  = [],[],[],[],[],[],[],[],[]
        for t in formants.ts():
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)
            f4 = formants.get_value_at_time(4, t)
            f1bw = formants.get_bandwidth_at_time(1, t)
            f2bw = formants.get_bandwidth_at_time(2, t)
            f3bw = formants.get_bandwidth_at_time(3, t)
            f4bw = formants.get_bandwidth_at_time(4, t)
            if np.isnan(f1): f1 = 0
            if np.isnan(f2): f2 = 0
            if np.isnan(f3): f3 = 0
            if np.isnan(f4): f4 = 0
            if np.isnan(f1bw): f1bw = 0
            if np.isnan(f2bw): f2bw = 0
            if np.isnan(f3bw): f3bw = 0
            if np.isnan(f4bw): f4bw = 0
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
            f1bw_list.append(f1bw)
            f2bw_list.append(f2bw)
            f3bw_list.append(f3bw)
            f4bw_list.append(f4bw)
            
        return f0, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list
    
####################################
res_fem, res_male = [],[]
#####################################
f0_fem,f0_mel = [],[]
f1f_bw, f1m_bw = [],[]
f2f_bw, f2m_bw = [],[]
f3f_bw, f3m_bw = [],[]
f4f_bw, f4m_bw = [],[]
#########################################

res_fem, res_male = [],[]
f0_fem,f0_mel = [],[]
f1f_bw, f1m_bw = [],[]
f2f_bw, f2m_bw = [],[]
f3f_bw, f3m_bw = [],[]
f4f_bw, f4m_bw = [],[]


datasets = ['00','01','02','03','04','05','06','07','08','09','10','vox2']    
for data in datasets:
 #print("for_plot/"+data)  
 #os.listdir("for_plots_analysis_female/for_plots_analysis_female/"+data) 
 f0_list, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list  = [],[],[],[],[],[],[],[],[]
 for wave_file in glob.glob("cslu_fem_analysis/"+data+"/*.wav"):
    #print("for_plots_analysis_female/"+data)
    #print("Processing {}...".format(wave_file))
    fs, x = wavfile.read(wave_file)
    #plt.plot(x)
    #Audio(x,rate=fs)
    f0, f1,f1bw, f2,f2bw, f3,f3bw, f4,f4bw = formants_praat(x,fs)
    for i in range(min(len(f0),len(f1))):
       if f0[i] > 0: 
          f0_list.append(f0[i])
          f1_list.append(f1[i])
          f2_list.append(f2[i])
          f3_list.append(f3[i])
          f4_list.append(f4[i])
          f1bw_list.append(f1bw[i])
          f2bw_list.append(f2bw[i])
          f3bw_list.append(f3bw[i])
          f4bw_list.append(f4bw[i])
 f0_fem.append (np.mean(f0_list))
 f1f_bw.append (np.mean(f1bw_list))
 f2f_bw.append (np.mean(f2bw_list))
 f3f_bw.append (np.mean(f3bw_list))
 f4f_bw.append (np.mean(f4bw_list))
 #,np.mean(f1_list),np.mean(f1bw_list),np.mean(f2_list),np.mean(f2bw_list),np.mean(f3_list),np.mean(f3bw_list),np.mean(f4_list),np.mean(f4bw_list)])
 #print(data,":",np.mean(f0_list),np.mean(f1_list),np.mean(f1bw_list),np.mean(f2_list),np.mean(f2bw_list),np.mean(f3_list),np.mean(f3bw_list),np.mean(f4_list),np.mean(f4bw_list))


datasets = ['00','01','02','03','04','05','06','07','08','09','10','vox2','wav_wsj']    
for data in datasets:
 #print("for_plots_analysis_female/"+data)  
 #os.listdir("for_plots_analysis_female/for_plots_analysis_female/"+data) 
 f0_list, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list  = [],[],[],[],[],[],[],[],[]
 for wave_file in glob.glob("cslu_male_analysis/"+data+"/*.wav"):
    #print("for_plots_analysis_female/"+data)
    #print("Processing {}...".format(wave_file))
    fs, x = wavfile.read(wave_file)
    #plt.plot(x)
    #Audio(x,rate=fs)
    f0, f1,f1bw, f2,f2bw, f3,f3bw, f4,f4bw = formants_praat(x,fs)
    for i in range(min(len(f0),len(f1))):
       if f0[i] > 0: 
          f0_list.append(f0[i])
          f1_list.append(f1[i])
          f2_list.append(f2[i])
          f3_list.append(f3[i])
          f4_list.append(f4[i])
          f1bw_list.append(f1bw[i])
          f2bw_list.append(f2bw[i])
          f3bw_list.append(f3bw[i])
          f4bw_list.append(f4bw[i])
 f0_mel.append (np.mean(f0_list))
 f1m_bw.append (np.mean(f1bw_list))
 f2m_bw.append (np.mean(f2bw_list))
 f3m_bw.append (np.mean(f3bw_list))
 f4m_bw.append (np.mean(f4bw_list))





x=['00','01','02','03','04','05','06','07','08','09','10','vox2']
import matplotlib.pyplot as plt
plt.plot(x,f0_fem[0:12]/f0_mel[12],'r',label='Female')
plt.plot(x,f0_mel[0:12]/f0_mel[12],'b',label ='Male')
plt.ylabel('f0 Scaling Factors') 
plt.legend()
plt.show()



plt.plot(x[0:11],f1f_bw[0:11]/f1m_bw[12],'r',label='Female')
plt.plot(x[0:11],f1m_bw[0:11]/f1m_bw[12],'b',label='Male')
plt.ylabel('f1_bw Scaling Factors') 
plt.legend()
plt.show()

plt.plot(x[0:11],f2f_bw[0:11]/f2m_bw[11],'r',label='Female')
plt.plot(x[0:11],f2m_bw[0:11]/f2m_bw[11],'b',label='Male')
plt.ylabel('f2_bw Scaling Factors') 
plt.legend()
plt.show()

plt.plot(x[0:11],f3f_bw[0:11]/f3m_bw[11],'r',label='Female')
plt.plot(x[0:11],f3m_bw[0:11]/f3m_bw[11],'b',label='Male')
plt.ylabel('f3_bw Scaling Factors') 
plt.legend()
plt.show()

plt.plot(x[0:11],f4f_bw[0:11]/f4m_bw[11],'r',label='Female')
plt.plot(x[0:11],f4m_bw[0:11]/f4m_bw[11],'b',label='Male')
plt.ylabel('f4_bw Scaling Factors') 
plt.legend()
plt.show()

###########################################
plt.plot(x[0:11],f1f_bw[0:11]/f1m_bw[11],'r',label='Female')
plt.plot(x[0:11],f1m_bw[0:11]/f1m_bw[11],'b',label='Male')
plt.ylabel('f1_bw Scaling Factors') 
plt.legend()
plt.show()

plt.plot(x[0:11],f2f_bw[0:11]/f2m_bw[11],'r',label='Female')
plt.plot(x[0:11],f2m_bw[0:11]/f2m_bw[11],'b',label='Male')
plt.ylabel('f2_bw Scaling Factors') 
plt.legend()
plt.show()

plt.plot(x[0:11],f3f_bw[0:11]/f3m_bw[11],'r',label='Female')
plt.plot(x[0:11],f3m_bw[0:11]/f3m_bw[11],'b',label='Male')
plt.ylabel('f3_bw Scaling Factors') 
plt.legend()
plt.show()

plt.plot(x[0:11],f4f_bw[0:11]/f4m_bw[11],'r',label='Female')
plt.plot(x[0:11],f4m_bw[0:11]/f4m_bw[11],'b',label='Male')
plt.ylabel('f4_bw Scaling Factors') 
plt.legend()
plt.show()
  


#vox: 161.2813201797606 522.4618217302313 188.32372486129498 1554.9291156927125 396.66149299937985 2555.244967419427 486.19798592564734 3610.545545693918 615.2036777664775
#cslu: 204.72989993889024 560.344379356569 284.1530589062684 1962.8212110787488 490.8882432779923 3088.4517033670777 519.6719832113796 3945.4895778574505 484.77017759369704
########################################################################################
               

def formants_praat(x, fs):
        f0min, f0max  = 75, 300
        sound = parselmouth.Sound(x, sampling_frequency=fs) # read the sound
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        formants = sound.to_formant_burg(time_step=0.010, maximum_formant=5000)
        
        f0_list, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list  = [],[],[],[],[],[],[],[],[]
        for t in formants.ts():
            print(t)             
          #if f0[t]:  
            #f0_list.append(f0[t])
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)
            f4 = formants.get_value_at_time(4, t)
            f1bw = formants.get_bandwidth_at_time(1, t)
            f2bw = formants.get_bandwidth_at_time(2, t)
            f3bw = formants.get_bandwidth_at_time(3, t)
            f4bw = formants.get_bandwidth_at_time(4, t)
            if np.isnan(f1): f1 = 0
            if np.isnan(f2): f2 = 0
            if np.isnan(f3): f3 = 0
            if np.isnan(f4): f4 = 0
            if np.isnan(f1bw): f1 = 0
            if np.isnan(f2bw): f2 = 0
            if np.isnan(f3bw): f3 = 0
            if np.isnan(f4bw): f4 = 0
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
            f1bw_list.append(f1bw)
            f2bw_list.append(f2bw)
            f3bw_list.append(f3bw)
            f4bw_list.append(f4bw)
            
        return f0, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list
    
##########################With IN SPEAKER VARIABILITY############################




#######################################################################################

f0_list, f1_list,f1bw_list, f2_list,f2bw_list, f3_list,f3bw_list, f4_list,f4bw_list  = [],[],[],[],[],[],[],[],[]
#for wave_file in glob.glob("cslu_fem_analysis/"+data+"/*.wav"):
    #print("for_plots_analysis_female/"+data)
    #print("Processing {}...".format(wave_file))
wave_file =  "cslu_fem_analysis/05/"+"/ksg3k450.wav"
fs, x = wavfile.read(wave_file)
    #plt.plot(x)
    #Audio(x,rate=fs)
f0, f1,f1bw, f2,f2bw, f3,f3bw, f4,f4bw = formants_praat(x,fs)
for i in range(min(len(f0),len(f1),len(f2),len(f3),len(f4))):
       if f0[i] > 0 and f0[i] < np.mean(f0) + 200 and f0[i] > 140: 
          f0_list.append(f0[i])
          f1_list.append(f1[i])
          f2_list.append(f2[i])
          f3_list.append(f3[i])
          f4_list.append(f4[i])
          f1bw_list.append(f1bw[i])
          f2bw_list.append(f2bw[i])
          f3bw_list.append(f3bw[i])
          f4bw_list.append(f4bw[i])
f0_fem.append (np.mean(f0_list))
f1f_bw.append (np.mean(f1bw_list))
f2f_bw.append (np.mean(f2bw_list))
f3f_bw.append (np.mean(f3bw_list))
f4f_bw.append (np.mean(f4bw_list))




f10_list, f11_list,f11bw_list, f12_list,f12bw_list, f13_list,f13bw_list, f14_list,f14bw_list  = [],[],[],[],[],[],[],[],[]
wave_file =  "cslu_fem_analysis/05/"+"/ksg3n4w0.wav"
fs, x = wavfile.read(wave_file)
    #plt.plot(x)
    #Audio(x,rate=fs)
f0, f1,f1bw, f2,f2bw, f3,f3bw, f4,f4bw = formants_praat(x,fs)
for i in range(min(len(f0),len(f1),len(f2),len(f3),len(f4))):
       if f0[i] > 0 and f0[i] < np.mean(f0) + 200 and f0[i] > 120:
          f10_list.append(f0[i])
          f11_list.append(f1[i])
          f12_list.append(f2[i])
          f13_list.append(f3[i])
          f14_list.append(f4[i])
          f11bw_list.append(f1bw[i])
          f12bw_list.append(f2bw[i])
          f13bw_list.append(f3bw[i])
          f14bw_list.append(f4bw[i])
f0_fem.append (np.mean(f0_list))
f1f_bw.append (np.mean(f1bw_list))
f2f_bw.append (np.mean(f2bw_list))
f3f_bw.append (np.mean(f3bw_list))
f4f_bw.append (np.mean(f4bw_list))





f20_list, f21_list,f21bw_list, f22_list,f22bw_list, f23_list,f23bw_list, f24_list,f24bw_list  = [],[],[],[],[],[],[],[],[]
wave_file =  "cslu_fem_analysis/05/"+"/ksg3n4q0.wav"
fs, x = wavfile.read(wave_file)
    #plt.plot(x)
    #Audio(x,rate=fs)
f0, f1,f1bw, f2,f2bw, f3,f3bw, f4,f4bw = formants_praat(x,fs)
for i in range(min(len(f0),len(f1),len(f2),len(f3),len(f4))):
       if f0[i] > 0 and f0[i] < np.mean(f0) + 200 and f0[i] > 120:
          f20_list.append(f0[i])
          f21_list.append(f1[i])
          f22_list.append(f2[i])
          f23_list.append(f3[i])
          f24_list.append(f4[i])
          f21bw_list.append(f1bw[i])
          f22bw_list.append(f2bw[i])
          f23bw_list.append(f3bw[i])
          f24bw_list.append(f4bw[i])
f0_fem.append (np.mean(f0_list))
f1f_bw.append (np.mean(f1bw_list))
f2f_bw.append (np.mean(f2bw_list))
f3f_bw.append (np.mean(f3bw_list))
f4f_bw.append (np.mean(f4bw_list))



f30_list, f31_list,f31bw_list, f32_list,f32bw_list, f33_list,f33bw_list, f34_list,f34bw_list  = [],[],[],[],[],[],[],[],[]
wave_file =  "cslu_fem_analysis/vox2/"+"00023.wav"
fs, x = wavfile.read(wave_file)
    #plt.plot(x)
    #Audio(x,rate=fs)
f0, f1,f1bw, f2,f2bw, f3,f3bw, f4,f4bw = formants_praat(x,fs)
for i in range(min(len(f0),len(f1),len(f2),len(f3),len(f4))):
       if f0[i] > 0 and f0[i] < np.mean(f0) + 200 and f0[i] > 120:
          f30_list.append(f0[i])
          f31_list.append(f1[i])
          f32_list.append(f2[i])
          f33_list.append(f3[i])
          f34_list.append(f4[i])
          f31bw_list.append(f1bw[i])
          f32bw_list.append(f2bw[i])
          f33bw_list.append(f3bw[i])
          f34bw_list.append(f4bw[i])
f0_fem.append (np.mean(f0_list))
f1f_bw.append (np.mean(f1bw_list))
f2f_bw.append (np.mean(f2bw_list))
f3f_bw.append (np.mean(f3bw_list))
f4f_bw.append (np.mean(f4bw_list))


import seaborn as sns
sns.distplot(f0_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f10_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f20_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')


import seaborn as sns
sns.distplot(f30_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'vox')
    
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'F0')
plt.title('With in Speaker Density Plot of f0')
plt.xlabel('f0')
xlim(left=10)
plt.ylabel('Density')







import seaborn as sns
sns.distplot(f1_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f11_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f21_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')
    
import seaborn as sns
sns.distplot(f31_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'vox')

# Plot formatting
plt.legend(prop={'size': 16}, title = 'F1')
plt.title('With in Speaker Density Plot of F1')
plt.xlabel('F1')
xlim(left=10)
plt.ylabel('Density')



import seaborn as sns
sns.distplot(f2_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f12_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f22_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'F2')
plt.title('With in Speaker Density Plot of F2')
plt.xlabel('F2')
xlim(left=10)
plt.ylabel('Density')





import seaborn as sns
sns.distplot(f3_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f13_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f23_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'F3')
plt.title('With in Speaker Density Plot of F3')
plt.xlabel('F3')
xlim(left=10)
plt.ylabel('Density')





import seaborn as sns
sns.distplot(f1bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f11bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f21bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'BW F1')
plt.title('With in Speaker Density Plot of BW of F1')
plt.xlabel('BW F1')
xlim(left=10)
plt.ylabel('Density')




import seaborn as sns
sns.distplot(f2bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f12bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f22bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'BW F2')
plt.title('With in Speaker Density Plot of BW of F2')
plt.xlabel('BW F2')
xlim(left=10)
plt.ylabel('Density')



import seaborn as sns
sns.distplot(f3bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-1')

import seaborn as sns
sns.distplot(f13bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-2')

import seaborn as sns
sns.distplot(f23bw_list, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'utt-3')
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'BW F3')
plt.title('With in Speaker Density Plot of BW of F3')
plt.xlabel('BW F3')
xlim(left=10)
plt.ylabel('Density')
