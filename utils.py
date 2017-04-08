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

import matplotlib.pyplot as plt
from operator import itemgetter
from collections import defaultdict

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


def get_feats_sad(wavpath):
	subprocess.call(['sox', wavpath, 'out.wav'])

	(rate, sig) = wav.read("out.wav")

	# Get silent frames
	sad_mask = []
	for idx, frame in enumerate(framesig(sig, frame_len, frame_step)):
	    if vad.is_speech(convert_np_to_pcm(frame), rate):
	        sad_mask.append(1)
	    else:
	        sad_mask.append(0)
	print "SAD length => ", len(sad_mask)

	fbank_feat = logfbank(sig,rate, winlen=0.030)

	return fbank_feat, np.array(sad_mask)


def plot_timeline(dataset, **kwargs):
    """
    Plots a timeline of events from different sources to visualize a relative
    sequence or density of events. Expects data in the form of:
        (timestamp, source, category)
    Though this can be easily modified if needed. Expects sorted input.
    """
    outpath = kwargs.pop('savefig', None)  # Save the figure as an SVG
    colors  = kwargs.pop('colors', {})     # Plot the colors for the series.
    series  = set([])                      # Figure out the unique series

    # Bring the data into memory and sort
    dataset = sorted(list(dataset), key=itemgetter(0))

    # Make a first pass over the data to determine number of series, etc.
    for _, source, category in dataset:
        series.add(source)
        if category not in colors:
            colors[category] = 'k'

    # Sort and index the series
    series  = sorted(list(series))

    # Create the visualization
    x = []  # Scatterplot X values
    y = []  # Scatterplot Y Values
    c = []  # Scatterplot color values

    # Loop over the data a second time
    for timestamp, source, category in dataset:
        x.append(timestamp)
        y.append(series.index(source))
        c.append(colors[category])

    plt.figure(figsize=(14,4))
    plt.title(kwargs.get('title', "Timeline Plot"))
    plt.ylim((-1,len(series)))
    plt.xlim((-1000, dataset[-1][0]+1000))
    plt.yticks(range(len(series)), series)
    plt.scatter(x, y, color=c, alpha=0.85, s=10)

    if outpath:
        return plt.savefig(outpath, format='svg', dpi=1200)

    return plt

def get_colors():
	from matplotlib import colors as mcolors

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	# Sort colors by hue, saturation, value and name.
	by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
	                for name, color in colors.items())
	sorted_names = [name for hsv, name in by_hsv]

	return sorted_names