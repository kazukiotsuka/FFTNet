#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# tts/FFTNet/dataset.py
#
# FFTNet dataset class
#

import sys
sys.path.append('../')
import os
import numpy as np
import torch
from datasets.voice_dataset import VoiceDataset, normalize
from tools.libaudio.encodes import mulaw_encode, mulaw_decode


class FFTNetDataset(VoiceDataset):

    def __init__(self, fft_net_depth=11, sample_rate=16000, key_name='jsut_ver1.1'):
        super(FFTNetDataset, self).__init__()
        self.__root_dir__ = f'/diskB/6/Datasets/VoiceData/{key_name}/preprocessed'
        self.__wav_dir__ == f'{self.__root_dir__}/wav'
        self.__f0_dir__ = f'{self.__root_dir__}/f0'
        self.__phoneme_dir__ = f'{self.__root_dir__}/phoneme'
        self.__sample_rate__ = sample_rate
        self.__mcc_coefficient__ = 25
        self.__receptive_field__ = 2 ** fft_net_depth

        self.wav_file_names = os.listdir(self.__wav_dir__)
        #self.mcc_file_names = os.listdir(self.__mcc_dir__)
        self.f0_file_names = os.listdir(self.__f0_dir__)
        self.phonemes_file_names = os.listdir(self.__phoneme_dir__)

    def __len__(self):
        return len(self.wav_file_names)

    def collate_fn(self, items):
        """
        wav: ndarray [x0, x1,x2,x3,..,xM,0.0,..,0.0] (B, max_len)
        f0: ndarray [f0_0, f0_1, ....,f0_M,0.0,..,0.0] (B, max_len)
        phoneme: ndarray [f0_0, f0_1, ....,f0_M,0.0,..,0.0] (B, max_len)
        lens: list [length of wav(0), length of wav(1), ...] (B)
        """
        B = len(items)
        lens = [item['wav'].shape[0] for item in items]
        max_len = np.max(lens)
        L = max_len + self.__receptive_field__ - 1

        wav_batch = np.zeros([B, L])
        target_batch = np.zeros([B, L])
        f0_batch = np.zeros([B, L])
        phonemes_batch = np.zeros([B, L])
        wav_file_names = []

        for i, item in enumerate(items):
            wav = normalize(item.get('wav'))
            f0 = item.get('f0')
            phonemes = item.get('phonemes')
            wav_file_name = item.get('wav_file_name')
            pad_len = max_len - len(wav)
            # add padding to right (FFTNet)
            wav = np.pad(
                wav,
                (self.__receptive_field__ - 1, pad_len),
                mode='constant',
                constant_values=0)
            f0 = np.pad(
                f0,
                (self.__receptive_field__ - 1, pad_len),
                mode='constant',
                constant_values=0.0)
            phonemes = np.pad(
                phonemes,
                (self.__receptive_field__ - 1, pad_len),
                mode='constant',
                constant_values=0.0)
            # sanity check
            assert len(wav) == L and len(f0) == L and len(phonemes) == L,\
                f'wav:{len(wav)} f0:{len(f0)} phonemes:{len(phonemes)} expected len:{L}'
            wav_batch[i] = wav
            target_batch[i] = mulaw_encode(wav)
            wav_file_names += [wav_file_name]
            f0_batch[i] = f0
            phonemes_batch[i] = phonemes

            print(f'wav_batch {wav_batch}')
            wav = torch.FloatTensor(wav_batch[:, :-1]).view(1, -1)
            print(f'wav {wav}')
            target = torch.LongTensor(target_batch[:, 1:]).view(1, -1)
            f0 = torch.FloatTensor(f0_batch[:, 1:]).view(1, -1)
            phonemes = torch.FloatTensor(phonemes_batch[:, 1:]).view(1, -1)
            # return wav_batch, f0_batch, phonemes_batch, lens, wav_file_names, wav_raw_batch
            return wav, target, f0, phonemes, lens, wav_file_names
