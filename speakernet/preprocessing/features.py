"""
Copyright 2022 Jianchen Li
"""

import torch
import torchaudio
from typing import Optional


class Fbank(torch.nn.Module):
    """
    Exportable, `torchaudio`-based implementation of Mel Spectrogram extraction.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        num_bins (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. (Default: ``True``)
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
        preemph (float or None, optional): Amount of pre emphasis to add to audio. Can be disabled by passing None.
            Defaults to None. Recommended as 0.97
        log (bool, optional): log mel. Default: True
        dither (float, optional): Amount of white-noise dithering. Defaults to 0.0. Recommended as 1e-5
        cmvn (bool, optional): Normalize the features. Default to True
        norm_val (bool, optional): Normalized by variance. Default to False
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        f_min: float = 0.,
        f_max: Optional[float] = None,
        pad: int = 0,
        num_bins: int = 80,
        window_fn: str = "hamming",
        power: float = 2.,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        preemph: Optional[float] = None,
        log: bool = True,
        dither: float = 0.0,
        cmvn: bool = True,
        norm_var: bool = False,
        training: bool = True
    ):
        super().__init__()

        # Copied from `AudioPreprocessor` due to the ad-hoc structuring of the Mel Spec extractor class
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

        # Ensure we can look up the window function
        if window_fn not in torch_windows:
            raise ValueError(f"Got window_fn value '{window_fn}' but expected a member of {torch_windows.keys()}")

        self.training = training
        self.use_log = log
        self.preemphasis_value = preemph
        self.dither = dither
        self.cmvn = cmvn
        self.norm_var = norm_var
        self.mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(window_size * sample_rate),
            hop_length=int(window_stride * sample_rate),
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=num_bins,
            window_fn=torch_windows[window_fn],
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
            norm=norm,
            mel_scale=mel_scale,
        )

    def _apply_dithering(self, signals: torch.Tensor) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            noise = torch.randn_like(signals) * self.dither
            signals = signals + noise
        return signals

    def _apply_preemphasis(self, signals: torch.Tensor) -> torch.Tensor:
        if self.preemphasis_value is not None:
            padded = torch.nn.functional.pad(signals, (1, 0))
            signals = signals - self.preemphasis_value * padded[:, :-1]
        return signals

    def _apply_log(self, features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if self.use_log:
            features = torch.log(features + eps)
        return features

    def _extract_mel_spec(self, signals: torch.Tensor) -> torch.Tensor:
        features = self.mel_spec_extractor(waveform=signals)
        return features

    def _apply_cmvn(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.cmvn:
            # [batch, feat_dim, feat_num]
            features = features - torch.mean(features, dim=2, keepdim=True)
            if self.norm_var:
                var = torch.var(features, dim=2, keepdim=True, unbiased=False)
                features = features / torch.sqrt(var + eps)
        return features

    def forward(self, input_signal: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                signals = self._apply_dithering(signals=input_signal)
                signals = self._apply_preemphasis(signals=signals)
                features = self._extract_mel_spec(signals=signals)
                features = self._apply_log(features=features)
                features = self._apply_cmvn(features=features)
                return features


class KaldiFbank(torch.nn.Module):
    """
    https://github.com/csukuangfj/kaldifeat
    Kaldi-compatible online & offline feature extraction with PyTorch, supporting CUDA, batch processing,
    chunk processing, and autograd - Provide C++ & Python API

    Args:
        samp_freq: Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
        frame_shift_ms: Frame shift in milliseconds. (float, default = 10)
        frame_length_ms: Frame length in milliseconds. (float, default = 25)
        dither: Dithering constant (0.0 means no dither). (float, default = 1)
        preemph_coeff: Coefficient for use in signal preemphasis. (float, default = 0.97)
        remove_dc_offset: Subtract mean from waveform on each frame. (bool, default = True)
        window_type: Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"sine"|"blackmann") (string, default = "povey")
        round_to_power_of_two: If true, round window size to power of two by zero-padding input to FFT. (bool, default = True)
        blackman_coeff: Constant coefficient for generalized Blackman window. (float, default = 0.42)
        snip_edges: If true, end effects will be handled by outputting only frames that completely fit in the file, and the number
            of frames depends on the frame-length.  If false , the number of frames depends only on the frame-shift, and we reflect
            the data at the ends. (bool, default = True)
        max_feature_vectors: Memory optimization. If larger than 0, periodically remove feature vectors so that only this number of the
            latest feature vectors is retained. (int, default = -1)
        num_bins: Number of triangular mel-frequency bins (int, default = 23)
        low_freq: Low cutoff frequency for mel bins. (float, default = 20)
        high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
        vtln_low: Low inflection point in piecewise linear VTLN warping function (float, default = 100)
        vtln_high: High inflection point in piecewise linear VTLN warping function (if negative, offset from high-mel-freq (float, default = -500)
        debug_mel: Print out debugging information for mel bin computation (bool, default = False)
        htk_mode (bool, default = False):
        use_energy: Add an extra dimension with energy to the FBANK output. (bool, default = False)
        energy_floor: Floor on energy (absolute, not relative) in FBANK computation. (float, default = 0)
            Only makes a difference if use_energy=true; only necessary if dither=0. Suggested values: 0.1 or 1.0
        raw_energy: If true, compute energy before preemphasis and windowing. (bool, default = True)
        htk_compat: If true, put energy last.  Warning: not sufficient to get HTK compatible features
            (need to change other parameters). (bool, default = False)
        use_log_fbank: If true, produce log-filterbank, else produce linear. (bool, default = True)
        use_power: If true, use power, else use magnitude. (bool, default = True)
        device: cpu or cuda

    """
    def __init__(
        self,
        samp_freq: float = 16000,
        frame_shift_ms: float = 10,
        frame_length_ms: float = 25,
        dither: float = 1,
        preemph_coeff: float = 0.97,
        remove_dc_offset: bool = True,
        window_type: str = "povey",
        round_to_power_of_two: bool = True,
        blackman_coeff: float = 0.42,
        snip_edges: bool = True,
        max_feature_vectors: int = -1,
        num_bins: int = 23,
        low_freq: float = 20,
        high_freq: float = 0,
        vtln_low: float = 100,
        vtln_high: float = -500,
        debug_mel: bool = False,
        htk_mode: bool = False,
        use_energy: bool = False,
        energy_floor: float = 0,
        raw_energy: bool = True,
        htk_compat: bool = False,
        use_log_fbank: bool = True,
        use_power: bool = True,
        device: torch.device = "cuda:0",
        cmvn: bool = True,
        norm_var: bool = False,
    ):
        super().__init__()

        import kaldifeat

        params_dict = locals()
        opts = kaldifeat.FbankOptions()
        opts_dict = opts.as_dict()
        for key, value in opts_dict.items():
            if key == "frame_opts":
                for _key in value.keys():
                    setattr(opts.frame_opts, _key, params_dict[_key])
            elif key == "mel_opts":
                for _key in value.keys():
                    setattr(opts.mel_opts, _key, params_dict[_key])
            else:
                setattr(opts, key, params_dict[key])

        self.fbank_extractor = kaldifeat.Fbank(opts)
        self.cmvn = cmvn
        self.norm_var = norm_var

    def _extract_fbank(self, signals: torch.Tensor) -> torch.Tensor:
        # To compute features that are compatible with Kaldi,
        # wave samples have to be scaled to the range [-32768, 32768]
        signals *= 32768
        signals = [sig for sig in signals]
        features = self.fbank_extractor(signals)
        features = torch.stack(features, dim=0)
        # [batch, feat_num, feat_dim] -> [batch, feat_dim, feat_num]
        features = features.transpose(1, 2)
        return features

    def _apply_cmvn(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.cmvn:
            # [batch, feat_dim, feat_num]
            features = features - torch.mean(features, dim=2, keepdim=True)
            if self.norm_var:
                var = torch.var(features, dim=2, keepdim=True, unbiased=False)
                features = features / torch.sqrt(var + eps)
        return features

    def forward(self, input_signal: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                features = self._extract_fbank(signals=input_signal)
                features = self._apply_cmvn(features=features)
                return features


if __name__ == "__main__":
    fbank = KaldiFbank(device="cuda:1", dither=0)
    filename = "/home/lijianchen/pdata/VoxCeleb2/dev/wav/id00039/y7c_8Xn8G-I/00107.wav"
    wave, samp_freq = torchaudio.load(filename)
    # wave = wave.squeeze()
    wave = torch.cat((wave, wave, wave), dim=0)
    features = fbank(wave)
    # print(features[:3])
