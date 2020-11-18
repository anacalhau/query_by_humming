# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:30:26 2020

@author: Ana Calhau
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
    
def smoothDownsampleFeature(f_feature, parameter):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Main program
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Temporal Smoothing
    if (parameter['winLenSmooth'] != 1) or (parameter['downsampSmooth'] != 1):
        winLenSmooth = parameter['winLenSmooth']
        downsampSmooth = parameter['downsampSmooth']
        stat_window = np.hanning(winLenSmooth)
        stat_window = stat_window / np.sum(stat_window)
        print(stat_window)

        # upfirdn filters and downsamples each column of f_stat_help
        f_feature_stat = np.zeros_like(f_feature)
        f_feature_stat = (scipy.signal.upfirdn(stat_window, f_feature.T, 1, downsampSmooth)).T
        seg_num = (f_feature.shape[1])
        stat_num = math.ceil(seg_num / downsampSmooth)
        cut = math.floor((winLenSmooth - 1) / (2 * downsampSmooth))
        f_feature_stat = f_feature_stat[:, 1 + cut:stat_num + cut]  # adjust group delay

    else:
        f_feature_stat = f_feature

    newFeatureRate = parameter['inputFeatureRate'] / parameter['downsampSmooth']

    return f_feature_stat, newFeatureRate


def normalizeFeature(f_feature, normP, threshold):
    f_featureNorm = np.zeros_like(f_feature)

    # normalise the vectors according to the l^p norm
    unit_vec = np.ones(12)
    unit_vec = unit_vec / np.linalg.norm(unit_vec, ord=normP)
    for k in range(f_feature.shape[1]):
        n = np.linalg.norm(f_feature[:, k], normP)
        if (n < threshold):
            f_featureNorm[:, k] = unit_vec
        else:
            f_featureNorm[:, k] = f_feature[:, k] / n

    return f_featureNorm

def internal_DCT(l):

    matrix = np.zeros((l,l))

    for m in range(l):
        for n in range(l):
            matrix[m,n] = np.sqrt(2/l)*np.cos((m*(n+0.5)*np.pi)/l)

    matrix[0,:] = matrix[0,:]/np.sqrt(2)

    return matrix


def pitch_to_CRP(f_pitch, parameter, sideinfo):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Main program
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    seg_num = f_pitch.shape[1]

    # % log compression
    if parameter["applyLogCompr"]:
        f_pitch_log = np.log10(parameter["addTermLogCompr"] + f_pitch * parameter["factorLogCompr"])
    else:
        f_pitch_log = f_pitch

    # DCT based reduction
    DCT = internal_DCT(f_pitch_log.shape[0])
    DCTcut = DCT
    DCTcut[np.setdiff1d(np.arange(120), parameter['coeffsToKeep']), :] = 0
    DCT_filter = np.dot(DCT.T, DCTcut)  # DCT.T @ DCTcut
    f_pitch_log_DCT = np.dot(DCT_filter, f_pitch_log)

    # calculate energy for each chroma band
    f_CRP = np.zeros((12, seg_num))
    for p in range(120):
        chroma = p % 12
        f_CRP[chroma, :] = f_CRP[chroma, :] + f_pitch_log_DCT[p, :]

    # normalize the vectors according to the norm l^p
    f_CRP = normalizeFeature(f_CRP, parameter['normP'], parameter['normThresh'])

    if (parameter['winLenSmooth'] != 1) or (parameter['downsampSmooth'] != 1):
        # Temporal smoothing and downsampling
        f_CRP, CrpFeatureRate = smoothDownsampleFeature(f_CRP, parameter)

        # re-normalize the vectors according to the norm l^p
        f_CRP = normalizeFeature(f_CRP, parameter['normP'], parameter['normThresh'])
    else:
        CrpFeatureRate = parameter['inputFeatureRate']

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Update sideinfo
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sideinfo["CRP"] = {}
    sideinfo['CRP']['coeffsToKeep'] = parameter['coeffsToKeep']
    sideinfo['CRP']['applyLogCompr'] = parameter['applyLogCompr']
    sideinfo['CRP']['factorLogCompr'] = parameter['factorLogCompr']
    sideinfo['CRP']['addTermLogCompr'] = parameter['addTermLogCompr']
    sideinfo['CRP']['normP'] = parameter['normP']
    sideinfo['CRP']['winLenSmooth'] = parameter['winLenSmooth']
    sideinfo['CRP']['downsampSmooth'] = parameter['downsampSmooth']
    sideinfo['CRP']['normThresh'] = parameter['normThresh']
    sideinfo['CRP']['featureRate'] = CrpFeatureRate

    return f_CRP, sideinfo


def main(args=None):
    # signal,fs = librosa.load(args.wavfile,sr=16000, mono=True)
    # chroma = librosa.feature.chroma_stft(y=signal, sr=fs)

    from loadmatlab import loadmat

    win_len = 4410
    f_pitch_mat = loadmat(args['matfilename'])

    f_pitch = f_pitch_mat['f_pitch']
    sideinfo = f_pitch_mat['sideinfo']

    parameter = {}
    parameter['coeffsToKeep'] = np.arange(54, 120)
    parameter['applyLogCompr'] = 1
    parameter['factorLogCompr'] = 1000
    parameter['featureRate'] = sideinfo["pitch"]["featureRate"]
    parameter['addTermLogCompr'] = 1
    parameter['normP'] = 2
    parameter['winLenSmooth'] = 1
    parameter['downsampSmooth'] = 1
    parameter['normThresh'] = 10 ^ -6
    parameter['inputFeatureRate'] = 0

    f_crp, sideinfo = pitch_to_CRP(f_pitch, parameter, sideinfo)

    parameter['xlabel'] = 'Time [Seconds]'
    parameter['title'] = 'CRP chromagram'
    # visualizeCRP(f_crp,parameter);
    # specshow(visualizeCRP(f_crp, parameter))

    #print(f_crp.shape)
    #print(sideinfo)


def getArgs():
    
    return None


if __name__ == '__main__':

    #args = getArgs()    
    args = {}
    args['wavfile'] = 'sax_audio.wav'
    args['matfilename'] = "/Users/Ana Calhau/MATLAB Drive/Chromas/data_feature/Systematic_Chord-C-Major_Eight-Instruments_pitch_4410.mat"
    main(args=args)