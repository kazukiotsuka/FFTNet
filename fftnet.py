#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# tts/FFTNet/fftnet.py
#
# - FFTLayer
# - FFTNetQueue
# - FFTNet


import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
sys.path.append('../../')
from datasets.voice_dataset import mulaw_decode
from IPython.core.debugger import Pdb


class FFTNet(nn.Module):
    def __init__(self, n_channels=256, n_depth=11, n_classes=256, disable_cuda=False, is_phoneme_conditioned=False, is_f0_conditioned=False):
        super().__init__()
        self.__disable_cuda__ = disable_cuda
        self.__n_channels__ = n_channels
        # self.__f0_channels__ = f0_channels
        # self.__phonemes_channels__ = phonemes_channels
        self.__n_depth__ = n_depth
        self.__n_classes__ = n_classes  # 256 categorical μ-law encoding
        self.__receptive_field_size__ = 2 ** n_depth  # 2**11 = 2048
        self.__is_phoneme_conditioned__ = is_phoneme_conditioned
        self.__is_f0_conditioned__ = is_f0_conditioned
        fft_layers = []
        for idx, input_size in enumerate([2 ** i for i in range(n_depth, 0, -1)]):
            if idx == 0:
                fft_layers += [FFTLayer(1, n_channels, idx, input_size)]
            else:
                fft_layers += [FFTLayer(n_channels, n_channels, idx, input_size)]
        self.fft_layers = nn.ModuleList(fft_layers)
        #self.fft_layers = nn.Sequential(*fft_layers)
        self.fully_connected = nn.Linear(n_channels, n_classes)
        print(f'Receptive Field: {self.__receptive_field_size__} samples')
        self.num_params(self)

        self.signal_buffer = None
        self.f0_buffer = None
        self.phoneme_buffer = None

    def settings(self) -> dict:
        return {
            'n_channels': self.__n_channels__,
            'n_depth': self.__n_depth__,
            'n_classes': self.__n_classes__,
            'receptive_field_size': self.__receptive_field_size__
        }

    def settings_str(self) -> str:
        return f'channels{self.__n_channels__}_depth{self.__n_depth__}_class{self.__n_classes__}_rfsize{self.__receptive_field_size__}'

    def num_params(self, model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def duration_to_seq_len(self, duration_ms: int, sample_rate=16000):
        return int(sample_rate / 1000) * duration_ms

    def init_buffers(self):
        N = self.__receptive_field_size__
        # assign random values for the top layer buffer (signal)
        x = torch.rand(1, 1, N)
        self.signal_buffer = [x[:, :, N // 2 - 1:]]

        # assign random values for the top layer buffer (f0 condition)
        if self.__is_f0_conditioned__:
            f0 = torch.rand(1, 1, N).cuda()
            self.f0_buffer = f0s[:, :, N // 2 - 1:]

        # assign random values for the top layer buffer (phoneme condition)
        if self.__is_phoneme_conditioned__:
            phonemes = torch.rand(1, 1, N).cuda()
            self.phoneme_buffer = phonemes[:, :, N // 2 - 1:]

        # assign forwarded tensor to each layers buffer
        for i, f in enumerate(self.fft_layers):
            N = f.__input_size__
            x_L = f.conv1d_L(x[:, :, :N // 2])
            x_R = f.conv1d_R(x[:, :, N // 2:])
            x = x_L + x_R
            if i == 0 and self.__is_f0_conditioned__:
                f0_L = f.conv1d_f0_L(f0[:, :, :-N // 2])
                f0_R = f.conv1d_f0_R(f0[:, :, N // 2:])
                x = x + f0_L + f0_R
            if i == 0 and self.__is_phoneme_conditioned__:
                # NOTE: categorical sequencial data works for convnets, but works without any spatial encoding?
                phonemes_L = self.conv1d_phonemes_L(phonemes[:, :, :-N // 2])
                phonemes_R = self.conv1d_phonemes_R(phonemes[:, :, N // 2:])
                x = x + phonemes_L + phonemes_R
            x = F.relu(x)
            x = F.relu(f.conv1d_out(x))
            self.signal_buffer += [x[:, :, N // 4 - 1:]]

    def push(self, i, y):
        self.signal_buffer[i] = torch.cat([self.signal_buffer[i], y], dim=-1)[:, :, 1:]

    def forward(self, x, f0=None, phonemes=None, is_padding=True):
        B, C, T = x.size()
        if not is_padding:
            padding = torch.zeros(B, C, self.__receptive_field_size__)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat([padding, x], dim=-1)

        for i, fft_layer in enumerate(self.fft_layers):
            if i == 0 and f0 is not None and phonemes is not None:
                x = fft_layer(x, f0=f0, phonemes=phonemes)
            elif i == 0 and f0 is not None and phonemes is None:
                x = fft_layer(x, f0=f0, phonemes=None)
            else:
                x = fft_layer(x)
        # self.fft_layers(x, f0, phonemes)  # NOTE: nn.Sequential version
        x = self.fully_connected(x.transpose(1, 2))
        return F.log_softmax(x, dim=-1)  # to use NLLLoss criterion

    #def push_to_signal_buffer(self, i, y):
    #    self.signal_buffer[i] = torch.cat([self.signal_buffer[i], y], dim=-1)[:, :, 1:]

    #def push_to_f0_buffer(self, i, y):
    #    self.f0_buffer[i] = torch.cat([self.f0_buffer[i], y], dim=-1)[:, :, 1:]

    #def push_to_phonemes_buffer(self, i, y):
    #    self.phonemes_buffer[i] = torch.cat([self.phonemes_buffer[i], y], dim=-1)[:, :, 1:]

    def class2float(self, x) :
        return 2 * x.float() / (256 - 1) - 1

    def generate(self, duration_ms: int, phonemes=None, pitches=None, sample_rate=16000):
        """Generate signal.

        args:
            - duration_ms: int
            - phonemes: list
            - pitches: list
            - sample_rate:  int
        """
        # Pdb().set_trace()
        with torch.no_grad():
            seq_len = self.duration_to_seq_len(duration_ms)
            self.init_buffers()
            samples = []
            start = time.time()
            for t in range(seq_len):
                for i, f in enumerate(self.fft_layers):
                    #print('-'*100)
                    #print(f'i {i} {f}')
                    x_R = f.conv1d_R(self.signal_buffer[i][:, :, -1:])
                    x_L = f.conv1d_L(self.signal_buffer[i][:, :, :1])
                    #print(f'x_R {x_R} {x_R.shape}')
                    #print(f'x_L {x_L} {x_L.shape}')
                    x = x_L + x_R
                    #print(f'x {x} {x.shape}')
                    if i == 0 and self.__is_f0_conditioned__:
                        f0_R = f.conv1d_f0_R(self.f0_buffer[:, :, :-self.__receptive_field_size__ // 2])
                        f0_L = f.conv1d_f0_L(self.f0_buffer[:, :, self.__receptive_field_size__ // 2:])
                        x = x + f0_L + f0_R
                    if i == 0 and self.__is_phoneme_conditioned__:
                        phonemes_R = f.conv1d_phonemes_R(self.phoneme_buffer[:, :, :-self.__receptive_field_size__ // 2])
                        phonemes_L = f.conv1d_phonemes_L(self.phoneme_buffer[:, :, self.__receptive_field_size__ // 2:])
                        x = x + phonemes_L + phonemes_R
                    x = F.relu(x)
                    #print(f'x <-relu {x} {x.shape}')
                    x = F.relu(f.conv1d_out(x))
                    #print(f'x <-relu(conv1d_out(x)) {x} {x.shape}')
                    self.push(i + 1, x)

                #print('='*100)
                #print(f'x {x}')
                x = self.fully_connected(x.squeeze(-1))
                #print(f'x <-fully_connected(x)) {x} {x.shape}')
                posterior = F.softmax(2 * x.view(-1), dim=0)
                #print(f'posterior {posterior} {posterior.shape}')
                #print(f'posterior {posterior}')
                dist = torch.distributions.Categorical(posterior)
                #print(f'dist {dist}')
                #print(f'dist {dist}')
                sample = dist.sample()
                #print(f'sample {sample}')
                sample_decoded = mulaw_decode(sample, qc=256)
                samples.append(sample_decoded)
                self.push(0, torch.tensor(sample_decoded).view(1, 1, 1))
                speed = (t + 1) / (time.time() - start)
                print(f'generate {sample}:{sample_decoded} {t+1}/{seq_len}, Speed: {speed:.2f} samples/sec')
                print('='*100)

        self.signal_buffer = None
        self.f0_buffer = None
        self.phoneme_buffer = None
        return np.array(samples)

    #def generate(self, duration_ms: int, phonemes=None, pitches=None, sample_rate=16000):
    #    # Pdb().set_trace()
    #    with torch.no_grad():
    #        c = 2
    #        B = 1
    #        output = []
    #        start = time.time()
    #        seq_len = self.duration_to_seq_len(duration_ms)
    #        for t in range(seq_len):
    #            if t == 0:
    #                # start sequence with random input
    #                x = torch.randint(1, 256, (1, 1, 1))
    #            for i, f in enumerate(self.fft_layers):
    #                if i == 0 and (pitches and phonemes):
    #                    x = f.generate_step(x, f0=pitches[t], phoneme=phonemes[t])
    #                else:
    #                    x = f.generate_step(x)
    #            x = self.fully_connected(x.squeeze(-1))
    #            posterior = F.softmax(c * x.view(-1), dim=0)
    #            # TODO: 2.3.2 Conditional sampling
    #            dist = torch.distributions.Categorical(posterior)
    #            output.append(dist.sample())
    #            speed = (t + 1) / (time.time() - start)
    #            print(f'generate class {dist.sample()}: {t+1}/{seq_len}, Speed: {speed:.2f} samples/sec')
    #            x = torch.FloatTensor([dist.sample()]).view(B, 1, 1)
    #            # Pdb().set_trace()
    #    return torch.stack(output).cpu().numpy()

    def save_model(self, save_model_path: str):
        try:
            torch.save(self.state_dict(), save_model_path)
            # torch.save(self, save_model_path)  # * this fails when data parallel
        except Exception as e:
            print(e)

    def load_model(self, model_file_path: str):
        try:
            self.load_state_dict(
                torch.load(model_file_path, map_location=lambda storage, loc: storage))
            # torch.load(model_file_path)  # * this fails if trained on multiple GPU. use state dict.
        except Exception as e:
            print(e)

class FFTLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layer_idx :int, input_size: int):
        super().__init__()
        # settings
        self.__in_channels__ = in_channels
        self.__out_channels__ = out_channels
        # self.__f0_channels__ = f0_channels  # always 1 because only the first layer
        # self.__phonemes_channels__ = phonemes_channels   # always 1 because only the first layer
        self.__input_size__ = input_size  # input size for this layer (=depth**2)
        self.__layer_idx__ = layer_idx
        # generate queues
        self.sample_queue = None
        self.f0_queue = None
        self.phonemes_queue = None
        # layers
        self.conv1d_L = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv1d_R = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        if layer_idx == 0:
            self.conv1d_f0_L = nn.Conv1d(1, out_channels, kernel_size=1)  # nn.Conv1d(f0_channels, out_channels, kernel_size=1)
            self.conv1d_f0_R = nn.Conv1d(1, out_channels, kernel_size=1)  # nn.Conv1d(f0_channels, out_channels, kernel_size=1)
            self.conv1d_phonemes_L = nn.Conv1d(1, out_channels, kernel_size=1)  # nn.Conv1d(phonemes_channels, out_channels, kernel_size=1)
            self.conv1d_phonemes_R = nn.Conv1d(1, out_channels, kernel_size=1)  # nn.Conv1d(phonemes_channels, out_channels, kernel_size=1)
        self.conv1d_out = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x, f0=None, phonemes=None):
        # Pdb().set_trace()
        x_L = self.conv1d_L(x[:, :, :-self.__input_size__//2])
        x_R = self.conv1d_R(x[:, :, self.__input_size__//2:])
        if f0 is not None and phonemes is not None:
            # only called in the first layer
            f0_L = self.conv1d_f0_L(f0[:, :, :-self.__input_size__//2])
            f0_R = self.conv1d_f0_R(f0[:, :, self.__input_size__//2:])
            # NOTE: categorical sequencial data works for convnets, but works without any spatial encoding?
            phonemes_L = self.conv1d_phonemes_L(phonemes[:, :, :-self.__input_size__//2])
            phonemes_R = self.conv1d_phonemes_R(phonemes[:, :, self.__input_size__//2:])
            x = x_L + x_R + f0_L + f0_R + phonemes_L + phonemes_R
        if f0 is not None and phonemes is None:
            # only called in the first layer
            f0_L = self.conv1d_f0_L(f0[:, :, :-self.__input_size__//2])
            f0_R = self.conv1d_f0_R(f0[:, :, self.__input_size__//2:])
            x = x_L + x_R + f0_L + f0_R
        else:
            x = x_L + x_R
        x = self.relu(x)
        return self.relu(self.conv1d_out(x))

    #def generate_step(self, x, f0=None, phoneme=None):
    #    # Pdb().set_trace()
    #    B = x.shape[0]

    #    # init queues
    #    if self.sample_queue is None:
    #        self.sample_queue = FFTNetQueue(B, self.__in_channels__, self.__input_size__, x.is_cuda)
    #    if (self.f0_queue is None and f0 is not None) and (self.phonemes_queue is None and phoneme is not None):
    #        self.f0_queue = FFTNetQueue(B, self.__in_channels__, self.__input_size__, x.is_cuda)
    #        self.phonemes_queue = FFTNetQueue(B, self.__in_channels__, self.__input_size__, x.is_cuda)

    #    # input - current sample (L) / previous sample (R)
    #    x_L = x
    #    x_R = self.sample_queue.enqueue(x)  # (B, T+C) (enqueueされるのはxと同じ値. then x_L == x_R)
    #    # forward to conv1d
    #    z1 = self.conv1d_L(x_L)
    #    z2 = self.conv1d_R(x_R)
    #    z = z1 + z2

    #    # f0, phonemes
    #    if f0 is not None and phoneme is not None:
    #        f0 = torch.FloatTensor([f0]).view(1 ,1 ,1)
    #        phoneme = torch.FloatTensor([phoneme]).view(1 ,1 ,1)
    #        f0_L = f0
    #        f0_R = self.f0_queue.enqueue(f0)
    #        phoneme_L = phoneme
    #        phoneme_R = self.phonemes_queue.enqueue(phoneme)
    #        # forward to conv1d
    #        z_f0_1 = self.conv1d_f0_L(f0_L)
    #        z_f0_2 = self.conv1d_f0_R(f0_R)
    #        z_phoneme_1 = self.conv1d_phonemes_L(phoneme_L)
    #        z_phoneme_2 = self.conv1d_phonemes_R(phoneme_R)
    #        z = z + z_f0_1 + z_f0_2 + z_phoneme_1 + z_phoneme_2

    #    z = self.relu(z)
    #    z = self.relu(self.conv1d_out(z))
    #    z = z.view(B, -1, 1)
    #    return z  # (B, 1, 1)


#class FFTNetQueue(object):
#    def __init__(self, batch_size, n_channels, input_size, is_cuda=False):
#        super(FFTNetQueue, self).__init__()
#        self.__batch_size__ = batch_size
#        self.__n_channels__ = n_channels
#        self.__input_size__ = input_size
#        self.__is_cuda__ = is_cuda
#        self.queue = []
#        self.reset(batch_size, is_cuda)
#
#    def reset(self, batch_size, is_cuda=False):
#        self.queue = torch.zeros([batch_size, self.__n_channels__, self.__input_size__])
#        if is_cuda:
#            self.queue = self.queue.cuda()
#
#    def enqueue(self, sample_to_push):
#        """Return the last sample and insert the new sample to the end of queue.
#
#        args:
#            - sample_to_push: (B, C, T=1)
#
#        1. remove first, 2. push sample_to_push to last, 3. return z
#        [0, 1, 2, 3, .., y, z] -> [1, 2, 3, .., y, z, sample_to_push]
#        """
#        # Pdb().set_trace()
#        sample_to_pop = self.queue[:, :, -1:].data  # the last input pointer
#        self.queue[:, :, :-1] = self.queue[:, :, 1:]  # [0, 1, 2, 3, .., y, z] -> [1, 2, 3, .., y, z, z]
#        self.queue[:, :, -1] = sample_to_push.view(sample_to_push.shape[0], sample_to_push.shape[1])
#        return sample_to_pop  # now queue[:, :, -1] is replaced with sample_to_push so that return the value of "sample_to_push"
