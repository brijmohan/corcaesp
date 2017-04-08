from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features.sigproc import framesig

import scipy.io.wavfile as wav
import numpy as np

import os, glob
from os.path import join, isdir
import subprocess
import math

train_dir = '/media/brij/CELLARDOOR/BRIJ/Corpus/TIMIT/timit/train'
test_dir = '/media/brij/CELLARDOOR/BRIJ/Corpus/TIMIT/timit/test'

mfcc_train_dir = 'feats/mfcc_sil/train'
#mfcc_test_dir = 'feats/mfcc_sil/test'
fbank_train_dir = 'feats/fbank_sil/train'
#fbank_test_dir = 'feats/fbank_sil/test'
sad_dir = 'feats/sad'
os.makedirs(mfcc_train_dir)
os.makedirs(fbank_train_dir)
os.makedirs(sad_dir)

import webrtcvad
vad = webrtcvad.Vad()
vad.set_mode(3)

frame_len = 480
frame_step = 160

'''
Soundfile returns numpy float array but webrtc.vad requires 16bit byte stream
Convert numpy float to 16bit pcm (byte stream)
'''
def convert_np_to_pcm(audio):
    return np.int16(audio*32768/max(abs(audio))).tobytes()


print "Extracting train feats..."
for feat_dir in [train_dir, test_dir]:
    for dr in os.listdir(feat_dir):
        print dr
        for spk in os.listdir(join(feat_dir, dr)):
            print spk
            spk_mfcc = []
            spk_fbank = []
            g_sad_mask = []
            for wavf in glob.glob(join(feat_dir, dr, spk) + '/*.wav'):
                #print wav
                subprocess.call(['sox', wavf, 'out.wav'])

                (rate, sig) = wav.read("out.wav")

                # Get silent frames
                sad_mask = []
                for idx, frame in enumerate(framesig(sig, frame_len, frame_step)):
                    if vad.is_speech(convert_np_to_pcm(frame), rate):
                        sad_mask.append(1)
                    else:
                        sad_mask.append(0)
                g_sad_mask += sad_mask
                print "SAD length => ", len(sad_mask)


                mfcc_feat = mfcc(sig, rate, winlen=0.030)
                d_mfcc_feat = delta(mfcc_feat, 2)

                mfcc_vel = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
                if len(spk_mfcc) == 0:
                    spk_mfcc = mfcc_vel
                else:
                    spk_mfcc = np.concatenate((spk_mfcc, mfcc_vel))
                print "MFCC feat size => ", spk_mfcc.shape

                fbank_feat = logfbank(sig,rate, winlen=0.030)
                if len(spk_fbank) == 0:
                    spk_fbank = fbank_feat
                else:
                    spk_fbank = np.concatenate((spk_fbank, fbank_feat))

                print "FBANK feat size => ", spk_fbank.shape

            np.save(join(mfcc_train_dir, spk), spk_mfcc)
            np.save(join(fbank_train_dir, spk), spk_fbank)
            np.save(join(sad_dir, spk), np.array(g_sad_mask))
