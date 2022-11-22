import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks

# 降噪、平滑时间序列
# Time Series Analysis
# Predicted seasonal macroeconmic indicators by Gaussian Filter method after identifying cyclical patterns by Fast Fourier Transform
# Identifying cyclical patterns by Fast Fourier Transform and predicting by Gaussian Filter method
# Calculated cylces by Fast Fourier Transform and Gaussian Filter
# Predicted seasonal macroeconomic indicators by detrending and then extracting cycles using Gauss Filtering 

def interpolation(data): # 仅在尾部有缺失值会报错
    x1 = np.where(~np.isnan(data))[0]
    x0 = np.where(np.isnan(data))[0]
    y = data[x1]
    func = interp1d(x1, y, kind = 'cubic', bounds_error = False)
    data[x0] = func(x0)
    return data

def period_mean_fft(data, nfft = 4096, nyquist = 1/2, peak_num = 3):
    Y = fft(data, nfft)[1:]     # 去掉零频分量          
    power = np.array([np.abs(y)**2 for y in Y[0:math.floor(nfft/2)]])
    # amplitude = np.array([np.abs(y) for y in Y[0:math.floor(nfft/2)]])
    freq = np.arange(0,math.floor(nfft/2))/math.floor(nfft/2)*nyquist
    prd = 1 / freq
    peak_id, _ = find_peaks(power)
    peaks = power[peak_id]
    ix = np.argsort(peaks)
    peak_id_sort = np.take_along_axis(peak_id,ix,axis=0)
    return prd[peak_id_sort[-peak_num:]]

def gauss_wave_predict(wave, nfft = 4096, period = 42, gauss_alpha = 1, predict_len = 12, filter_type = 1):
    # detrend before gauss
    # signal.detrend
    wave_len = len(wave)
    wave_fill_zero = np.zeros(nfft)
    wave_fill_zero[-wave_len:] = wave # 向前补零提升分辨率
    wave_fft = fft(wave, nfft)
    gauss_index = np.arange(nfft) + 1
    center_frequency = nfft / period + 1
    # 生成高斯滤波响应
    if filter_type == 1:
        gauss_win = np.exp(-(gauss_index - center_frequency)**2 / gauss_alpha**2)
    else:
        gauss_win = np.exp(-(gauss_index - center_frequency)**2 / (center_frequency * gauss_alpha / 100)**2)
    wave_filter = wave_fft * gauss_win
    wave_filter[int(nfft/2):] = wave_filter[int(nfft/2):0:-1].conjugate()
    res = ifft(wave_filter).real
    return np.hstack([res,res[0:predict_len]])[-(wave_len+predict_len):]
