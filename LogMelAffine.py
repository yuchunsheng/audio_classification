import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional


class ConvSTFT(nn.Module):
    """
    Conv1d-based STFT that is ONNX-export-friendly and matches torch.stft
    with center=True semantics.

    Emulates torch.stft with:
      - center=True  -> pad by n_fft//2 (via reflect/constant/replicate)
      - normalized=False
      - onesided=True  (we build only positive frequencies)
      - return_complex=False (emit real/imag via two filterbanks)
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 800,
        window: Optional[torch.Tensor] = None,
        pad_center: bool = True,
        pad_mode: str = "reflect",   # "reflect" matches torchaudio center-padding
        pad_value: float = 0.0,
    ):
        super().__init__()
        assert win_length <= n_fft, "win_length must be <= n_fft"
        assert pad_mode in ("constant", "reflect", "replicate"), "Unsupported pad_mode"

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.pad_center = bool(pad_center)
        self.pad_mode = pad_mode
        self.pad_value = float(pad_value)

        # Window (make sure periodic=True to match torchaudio)
        if window is None:
            window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        else:
            window = window.to(dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

        # Build Fourier basis for positive freqs [0..n_fft//2], window centered in n_fft kernel
        num_bins = self.n_fft // 2 + 1
        offset = (self.n_fft - self.win_length) // 2
        win_full = torch.zeros(self.n_fft, dtype=torch.float32)
        win_full[offset:offset + self.win_length] = self.window

        # n over 0..n_fft-1, k over 0..num_bins-1
        n = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(0)   # [1, n_fft]
        k = torch.arange(num_bins, dtype=torch.float32).unsqueeze(1)     # [num_bins, 1]
        ang = 2 * torch.pi * (k @ (n / float(self.n_fft)))               # [num_bins, n_fft]

        cos_k = torch.cos(ang) * win_full                                # [num_bins, n_fft]
        sin_k = -torch.sin(ang) * win_full                               # [num_bins, n_fft]

        weight = torch.cat([cos_k, sin_k], dim=0).unsqueeze(1)           # [2*num_bins, 1, n_fft]
        self.register_buffer("fourier_basis", weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] waveform

        Returns:
            stft_out: [B, F, frames, 2]  (real, imag)
        """
        B, T = x.shape

        # center=True -> pad by n_fft//2 using chosen pad_mode
        if self.pad_center:
            pad = self.n_fft // 2
            if self.pad_mode == "constant":
                x = F.pad(x, (pad, pad), mode="constant", value=self.pad_value)
            else:
                x = F.pad(x, (pad, pad), mode=self.pad_mode)

        W = self.fourier_basis.to(dtype=x.dtype, device=x.device)  # [2*F, 1, n_fft]

        y = F.conv1d(
            x.unsqueeze(1),      # [B, 1, T']
            W,                   # [2*F, 1, n_fft]
            bias=None,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
        )  # -> [B, 2*F, frames]

        num_bins = self.n_fft // 2 + 1
        frames = y.shape[-1]
        y = y.view(B, 2, num_bins, frames)
        real = y[:, 0, :, :]
        imag = y[:, 1, :, :]
        stft_out = torch.stack([real, imag], dim=-1)  # [B, F, frames, 2]
        return stft_out


class MelSpectrogramMatched(nn.Module):
    """
    ONNX-friendly Mel front end that matches torchaudio.transforms.MelSpectrogram
    numerics (to floating-point tolerance).

    Steps: STFT -> |X|^power -> Mel-projection -> (optional) depthwise affine -> (optional) log

    Key matching details:
      - Window periodicity: Hann(periodic=True)
      - center=True, pad_mode="reflect"
      - normalized=False (no FFT normalization)
      - onesided=True (only positive frequencies)
      - power exponent applied to magnitude (matches torchaudio)
      - Mel filter via torchaudio.functional.melscale_fbanks

    Args mirror torchaudio.MelSpectrogram; pass the same values to match exactly.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: Optional[int] = 1024,
        hop_length: Optional[int] = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,                     # unused (we emulate center-padding instead)
        n_mels: int = 64,
        window_fn=torch.hann_window,      # keep for API symmetry; we always use Hann(periodic=True)
        power: float = 2.0,               # applied to |X| (magnitude)
        normalized: bool = False,         # keep False to match default
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,            # fixed True in our implementation
        norm: Optional[str] = None,       # e.g., None or "slaney"
        mel_scale: str = "htk",           # "htk" or "slaney"
        apply_log: bool = True,           # optional log compression
        log_eps: float = 1e-6,            # used for log safety
        learn_affine: bool = False,       # optional per-mel affine after mel projection
    ):
        super().__init__()
        assert onesided, "This implementation assumes onesided=True."
        assert not normalized, "This STFT path matches normalized=False."

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else self.win_length // 2
        self.f_min = float(f_min)
        self.f_max = float(self.sample_rate / 2.0 if f_max is None else f_max)
        self.n_mels = int(n_mels)
        self.power = float(power)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.norm = norm
        self.mel_scale = mel_scale
        self.apply_log = bool(apply_log)
        self.log_eps = float(log_eps)
        self.learn_affine = bool(learn_affine)

        # 1) STFT using ConvSTFT
        self.stft = ConvSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, periodic=True),
            pad_center=self.center,
            pad_mode=self.pad_mode,
            pad_value=0.0,
        )

        # 2) Mel filterbank (F, M)
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm=self.norm,
            mel_scale=self.mel_scale,
        )  # shape: (F, M)
        self.register_buffer("mel_fb", fb, persistent=False)

        # 3) Optional per-mel affine (depthwise 1x1 conv), identity init
        if self.learn_affine:
            self.affine = nn.Conv1d(self.n_mels, self.n_mels, kernel_size=1, groups=self.n_mels, bias=True)
            nn.init.ones_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)
        else:
            self.affine = None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) float32/float16 (any floating type)

        Returns:
            (B, n_mels, frames) if apply_log=True -> log-mel, else linear mel power
        """
        # STFT
        stft = self.stft(wav)                        # (B, F, frames, 2)
        mag2 = (stft ** 2).sum(-1)                   # power spectrum: |X|^2  -> (B, F, frames)

        # Apply power exponent on magnitude to match torchaudio: |X|^power
        if self.power == 2.0:
            S = mag2
        elif self.power == 1.0:
            S = torch.sqrt(torch.clamp_min(mag2, 0.0))
        else:
            # general case: |X|^p = (|X|^2)^(p/2)
            S = torch.clamp_min(mag2, 0.0) ** (self.power / 2.0)  # (B, F, frames)

        # Mel projection (B, F, T) -> (B, T, F) @ (F, M) -> (B, T, M) -> (B, M, T)
        mel_fb = self.mel_fb.to(dtype=S.dtype, device=S.device)
        mel = torch.matmul(S.transpose(1, 2), mel_fb).transpose(1, 2)    # (B, M, frames)

        # Optional per-mel affine in linear domain (be careful with log)
        if self.affine is not None:
            mel = self.affine(mel)

        if self.apply_log:
            # Safe log (matching common log-mel: natural log by default)
            mel = torch.log(torch.clamp_min(mel, self.log_eps))

        return mel