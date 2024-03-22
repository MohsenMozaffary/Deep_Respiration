import torch
import torch.nn as nn
import torch.nn.functional as F
tr = torch

class NegativeMaxCrossCorr(nn.Module):
    def __init__(self):
        super(NegativeMaxCrossCorr, self).__init__()
        Fs = 10
        high_pass = (30 / 60) * 60 
        low_pass = (6 / 60) * 60
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.where(denom > 0, cov / denom, torch.zeros_like(cov))
        return output.mean()

class Total_loss(nn.Module):
    def __init__(self, Fs, high_pass, low_pass, a = 0.5, b = 0.5):
        super(NegativeMaxCrossCov, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.a = a
        self.b = b

    def forward(self, preds, labels):
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        N = 8 * preds_pad.shape[-1]
        preds_fft = torch.fft.rfft(preds_pad, dim=-1, n=N)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1, n=N)
        freqs = torch.fft.rfftfreq(n=N) * self.Fs

        X = preds_fft * torch.conj(labels_fft)
        X_real = tr.view_as_real(X)

        Fn = self.Fs / 2
        use_freqs = torch.logical_and(freqs <= self.high_pass / 60, freqs >= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = tr.sum(tr.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = tr.sum(tr.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = tr.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]

        X[:, zero_freqs] = 0.

        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)
        max_cc = torch.max(cc, dim=-1)[0] / energy_ratio


        # finding AFD
        BPM_diff = self.absolute_peak_frequency_difference(preds, labels, self.Fs)

        total_loss = self.a*(-max_cc) + self.b*(-BPM_diff)


        return total_loss
    



    def find_peak_frequency(self, signal, sampling_rate, freq_range=(0.05, 0.7)):
            N = len(signal)
            fft_values = torch.fft.fft(signal)
            fft_freqs = torch.fft.fftfreq(N, d=1.0/sampling_rate, device=signal.device)
            
            positive_freqs = fft_freqs[:N//2]
            positive_fft_values = torch.abs(fft_values[:N//2])
            
            # Find peak frequency within the specified frequency range
            freq_mask = (positive_freqs >= freq_range[0]) & (positive_freqs <= freq_range[1])
            peak_index = torch.argmax(positive_fft_values[freq_mask])
            peak_frequency = positive_freqs[freq_mask][peak_index]
            
            return peak_frequency

    def absolute_peak_frequency_difference(self, signal1, signal2, sampling_rate):
    
        peak_frequency1 = self.find_peak_frequency(signal1, sampling_rate)
        peak_frequency2 = self.find_peak_frequency(signal2, sampling_rate)
        peak_difference = torch.abs(peak_frequency1 - peak_frequency2)
        BPM_diff = peak_difference * 60
        BPM_diff = BPM_diff/3

        if BPM_diff > 1:
            BPM_diff = 1


        return BPM_diff
        