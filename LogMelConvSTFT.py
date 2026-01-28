import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class ConvSTFT(nn.Module):
    """
    Conv1d-based STFT that is ONNX-export-friendly.

    Emulates torch.stft with:
      - center=True  -> zero pad by n_fft//2 at both ends
      - normalized=False
      - return_complex=False (we output real/imag via two filter banks)
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 800,
        window: torch.Tensor | None = None,
        pad_center: bool = True,
    ):
        super().__init__()
        assert win_length <= n_fft, "win_length must be <= n_fft"

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.pad_center = bool(pad_center)

        # Window
        if window is None:
            window = torch.hann_window(win_length, periodic=False, dtype=torch.float32)
        else:
            window = window.to(dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

        # Build Fourier basis for positive frequencies [0 .. n_fft//2]
        # Real kernels:  window[n] * cos(2πkn/N)
        # Imag kernels: -window[n] * sin(2πkn/N)
        
        # Build Fourier basis for positive frequencies [0..n_fft//2]
        num_bins = n_fft // 2 + 1
        
        # Center the window inside an n_fft frame
        offset = (n_fft - win_length) // 2
        win_full = torch.zeros(n_fft, dtype=torch.float32)
        win_full[offset:offset+win_length] = self.window  # centered window
        
        # n over 0..n_fft-1, k over 0..num_bins-1
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)  # [1, n_fft]
        k = torch.arange(num_bins, dtype=torch.float32).unsqueeze(1)  # [num_bins, 1]
        ang = 2 * math.pi * k @ (n / float(n_fft))  # [num_bins, n_fft]
        
        cos_kernels = torch.cos(ang) * win_full      # [num_bins, n_fft]
        sin_kernels = -torch.sin(ang) * win_full     # [num_bins, n_fft]
        
        weight = torch.cat([cos_kernels, sin_kernels], dim=0).unsqueeze(1)  # [2*num_bins, 1, n_fft]
        self.register_buffer("fourier_basis", weight.contiguous(), persistent=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] waveform

        Returns:
            stft_out: [B, num_bins, frames, 2]  where last dim is (real, imag)
        """
        B, T = x.shape

        # center=True -> zero pad by n_fft//2
        if self.pad_center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode="constant", value=0.0)  # [B, T + 2*pad]

        # Convolution to compute dot product with each Fourier kernel at each frame
        # Input must be [B, 1, T]
        y = F.conv1d(
            x.unsqueeze(1),                       # [B, 1, T+pad*2]
            self.fourier_basis,                   # [2*F, 1, win_length]
            bias=None,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
        )  # -> [B, 2*F, frames]

        num_bins = self.n_fft // 2 + 1
        frames = y.shape[-1]
        y = y.view(B, 2, num_bins, frames)        # [B, 2, F, frames]
        real = y[:, 0, :, :]                      # [B, F, frames]
        imag = y[:, 1, :, :]                      # [B, F, frames]

        # Pack to match torch.stft(return_complex=False) output layout
        stft_out = torch.stack([real, imag], dim=-1)  # [B, F, frames, 2]
        return stft_out

class LogMelSpectrogramConvSTFT(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        win_length: int = 800,
        hopsize: int = 320,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: float | None = None,
    ):
        super().__init__()

        if fmax is None:
            fmax = sr // 2

        self.n_mels = int(n_mels)
        self.sr = int(sr)
        self.win_length = int(win_length)
        self.hopsize = int(hopsize)
        self.n_fft = int(n_fft)
        self.fmin = float(fmin)

        nyquist = sr // 2
        if fmax is None:
            fmax = nyquist
        elif fmax > nyquist:
            print(f"[LogMel] fmax={fmax} > Nyquist={nyquist}. Clamping to Nyquist.")
            fmax = float(nyquist)
        self.fmax = float(fmax)

        assert 0.0 <= self.fmin < self.fmax <= (self.sr / 2), \
            f"Invalid band: fmin={self.fmin}, fmax={self.fmax}, nyquist={self.sr/2}"

        # Pre-emphasis kernel y[t] = x[t] - 0.97*x[t-1]
        self.register_buffer(
            "preemphasis_kernel",
            torch.tensor([[[-0.97, 1.0]]], dtype=torch.float32),
            persistent=False
        )

        # STFT via conv: ConvSTFT must center win inside n_fft and use n_fft-length kernels internally
        self.stft = ConvSTFT(
            n_fft=self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, periodic=False, dtype=torch.float32),
            pad_center=True,
        )

        # Kaldi mel filter bank (Nyquist excluded), then pad Nyquist
        mel_bins, _ = torchaudio.compliance.kaldi.get_mel_banks(
            num_bins=self.n_mels,
            window_length_padded=self.n_fft,
            sample_freq=self.sr,
            low_freq=self.fmin,
            high_freq=self.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_bins = F.pad(mel_bins, (0, 1), value=0.0)  # -> [n_mels, n_fft//2 + 1]
        self.register_buffer("mel_basis", mel_bins.to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Current behavior (length T-1) to match EfficientAT reference:
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_kernel).squeeze(1)

        # STFT -> power
        stft_out = self.stft(x)               # [B, F, frames, 2]
        power = (stft_out ** 2).sum(dim=-1)   # [B, F, frames]

        # Mel projection (use correct einsum string, keep dtype/device aligned)
        mel_basis = self.mel_basis.to(dtype=power.dtype, device=power.device)  # [M, F]
        mel = torch.einsum('mf,bft->bmt', mel_basis, power)                    # [B, M, T]

        # Log compression + normalization
        log_mel = (mel + 1e-5).log()
        log_mel = (log_mel + 4.5) / 5.0
        return log_mel

    @torch.no_grad()
    def power_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_kernel).squeeze(1)
        stft_out = self.stft(x)
        power = (stft_out ** 2).sum(dim=-1)
        return power

class MelSpecAugment(nn.Module):
    """
    SpecAugment on log-mel (training only).
    """
    def __init__(self, freqm: int = 48, timem: int = 192):
        super().__init__()
        self.freqm = (
            nn.Identity() if freqm == 0 else torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        )
        self.timem = (
            nn.Identity() if timem == 0 else torchaudio.transforms.TimeMasking(timem, iid_masks=True)
        )

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return log_mel
        x = self.freqm(log_mel)
        x = self.timem(x)
        return x

class RandomMelEdgeProjector(nn.Module):
    """
    Re-project power spectrogram with randomized fmin/fmax during training.
    """
    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: float | None = None,
        fmin_aug_range: int = 10,
        fmax_aug_range: int = 1000,
    ):
        super().__init__()
        nyquist = sr // 2
        if fmax is None:
            fmax = nyquist
        elif fmax > nyquist:
            # Optional: print/log once so you know a clamp happened
            print(f"[LogMel] fmax={fmax} > Nyquist={nyquist}. Clamping to Nyquist.")
            fmax = float(nyquist)
        
        self.base_fmin = float(fmin)
        self.base_fmax = float(fmax)
        assert 0.0 <= self.base_fmin < self.base_fmax <= (sr / 2), f"Invalid band: fmin={self.fmin}, fmax={self.fmax}, nyquist={self.sr/2}"

        assert fmin_aug_range >= 1
        assert fmax_aug_range >= 1

        self.n_mels = n_mels
        self.sr = sr
        self.n_fft = n_fft
        
        self.fmin_aug_range = int(fmin_aug_range)
        self.fmax_aug_range = int(fmax_aug_range)

    def _build_mel_basis(self, fmin: float, fmax: float, device, dtype):
        mb, _ = torchaudio.compliance.kaldi.get_mel_banks(
            num_bins=self.n_mels,
            window_length_padded=self.n_fft,
            sample_freq=self.sr,
            low_freq=fmin,
            high_freq=fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mb = torch.nn.functional.pad(mb, (0, 1), value=0.0)
        return mb.to(device=device, dtype=dtype)

    def forward(self, power_spec: torch.Tensor) -> torch.Tensor:
        if self.training:
            fmin = self.base_fmin + torch.randint(self.fmin_aug_range, (1,), device=power_spec.device).item()
            fmax = self.base_fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,), device=power_spec.device).item()
            nyquist = self.sr // 2 
            if fmax is None:
                fmax = nyquist
            elif fmax > nyquist:
                # Optional: print/log once so you know a clamp happened
                print(f"[LogMel] fmax={fmax} > Nyquist={nyquist}. Clamping to Nyquist.")
                fmax = float(nyquist)
        else:
            fmin, fmax = self.base_fmin, self.base_fmax

        mel_basis = self._build_mel_basis(fmin, fmax, device=power_spec.device, dtype=power_spec.dtype)
        mel = torch.matmul(mel_basis, power_spec)  # [B, n_mels, frames]
        return mel