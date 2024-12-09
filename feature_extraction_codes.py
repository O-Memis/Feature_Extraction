
"""
Some useful Feature Extraction codes.

To apply measurements and gain indights on your data.

Especially for 1D (time-series) physiological signals.
"""



import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch



# Simulated 1D signal for demonstration

T_duration = 1  # duration in seconds
fs = 1000  # Sampling frequency (Hz)
N = T_duration * fs  # total number  of samples


t = np.linspace(0, T_duration, N)  # time variable

signal = np.sin(2*np.pi*3*t)*1.3 + np.sin(2*np.pi*50*t) - np.random.randn(1000)   





plt.figure(figsize=(15,7))
plt.plot(signal)
plt.show()



#%% 1) Time-domain statistics & measurements



"""
Time-domain features capture amplitude-based characteristics of the signal.
These features are useful for analyzing the overall structure, intensity, and variability of the signal.
"""


# T1: Maximum value of the signal (Intensity)
T1 = np.max(signal)  
# The highest amplitude value in the signal


# T2: Minimum value of the signal (Intensity)
T2 = np.min(signal)  
# The lowest amplitude value in the signal


# T3: Mean of the signal (Shift)
T3 = np.mean(signal)  
# Indicates the signal's central tendency or baseline shift.


# T4: Variance of the signal (Degree of dispersion)
T4 = np.var(signal)  
# Useful for understanding how much the signal deviates from its average value.


# T5: Standard deviation of the signal (Stability)
T5 = np.std(signal)  
# Standard deviation is the square root of variance


# T6: Mean absolute deviation (Degree of dispersion)
T6 = np.mean(np.abs(signal - T3))  
# Average of the absolute differences from the mean, less sensitive to outliers than variance.


# T7: Root mean square (RMS) (Stability)
T7 = np.sqrt(np.mean(signal**2))  
# Represents the effective power or energy of the signal.


# T8: Average Difference
T8 = np.mean(np.abs(np.diff(signal)))
# Average of the absolute differences between consecutive points in the signal. This feature is used to quantify how much the signal changes on average.


# T9: Absolute energy (Energy distribution)
T9 = np.sum(signal**2)  
# Measures the total energy of the signal, useful for power analysis.


# T10: Peak-to-peak distance (Periodicity)
T10 = np.ptp(signal)  
# Indicates the amplitude range of the signal.


# T11: Sum of absolute differences (Intensity of change)
T11 = np.sum(np.abs(np.diff(signal)))  
# Measures the overall intensity of changes in the signal.


# T12: Shannon entropy (Uncertainty)
hist, bin_edges = np.histogram(signal, bins='auto', density=True)  # Histogram of signal values
probabilities = hist / np.sum(hist)  # Normalize histogram to probabilities
T12 = entropy(probabilities, base=2)  # Shannon entropy of the signal
# Quantifies the randomness or complexity of the signal's amplitude distribution.


# T13: Area under the curve (Energy distribution)
T13 = np.trapz(signal)  
# Integral of the signal


# T14: Autocorrelation (lag-1) (Periodicity)
T14 = np.corrcoef(signal[:-1], signal[1:])[0, 1]  
# Correlation between signal and its lagged version


# T15: Signal center point (Shift)
T15 = np.sum(np.arange(N) * signal**2) / np.sum(signal**2)  
# Represents the "center of mass" of the signal's energy distribution.


# T16: Number of peaks (Repeatability)
peaks, _ = find_peaks(signal)  # Identify peaks in the signal
T16 = len(peaks)  # Count the number of peaks


# T17: Signal distance (Difference in point in time)
T17 = np.sum(np.sqrt(1 + np.diff(signal)**2))  
# Cumulative Euclidean distance between points


# T18: Total energy (Overall strength and power)
T18 = np.sum(signal**2) / N  
# Average energy of the signal


# T19: Zero crossing rate (Rate of change)
T19 = np.sum(signal[:-1] * signal[1:] < 0) / N  
# Measures the rate of sign changes, useful for detecting oscillatory behavior.


# T20: Skewness (Symmetry)
T20 = skew(signal)  
# Asymmetry of the signal distribution


# T21: Kurtosis (Peak distribution pattern)
T21 = kurtosis(signal)  
# Peakedness of the signal distribution


# T22: Number of positive turning points (Waveform characteristics)
T22 = np.sum((signal[:-2] < signal[1:-1]) & (signal[1:-1] > signal[2:]))  #
# Counts the number of upward peaks.


# T23: Number of negative turning points (Waveform characteristics)
T23 = np.sum((signal[:-2] > signal[1:-1]) & (signal[1:-1] < signal[2:]))  
# Counts the number of downward peaks.


# T24: Signal range
T24 = np.max(signal) - np.min(signal)  
# Difference between max and min values


# T25: Signal line length
T25 = np.sum(np.abs(np.diff(signal)))  # Sum of absolute differences
# Measures the complexity or roughness of the signal.


# T26-T27: Hjorth parameters


T26 = np.sqrt(np.var(np.diff(signal)) / T4)  # Hjorth Mobility
# Represents the signal's frequency content.

T27 = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal))) / T26  # Hjorth Complexity
# Measures the change in frequency over time.




# Print all time-domain features
print("Time-Domain Features:")
for i, value in enumerate([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27], start=1):
    print(f"T{i}: {value}")








#%% 2) Frequency-domain measurements

"""
Frequency-domain features analyze the spectral content of the signal.
These features are critical for understanding oscillatory components and periodicity.
"""






# Compute the FFT of the signal
fft_val = np.abs(fft(signal)) 


# Generate frequency bins 
freqs = fftfreq(len(signal), d=1/fs)  # Full frequency range (bilateral)
freqs = freqs[:len(freqs)//2]  # Take only positive side 
fft_val = fft_val[:len(freqs)]  # Positive side of the bilateral FFT values


power_vals = fft_val**2  # linear power spectrum
power_vals = 10 * np.log10( power_vals + 1e-12) # logarithmic power spectrum 



"""
Also you can use a Periodogram, different calculation methods of the power spectrum.
Here is the Welch method, which uses averages of windowed FFTs:
    
    freqs, psd = welch(signal, fs=fs, nperseg=256, noverlap=128, window='hamming')
    
    *nperseg:
        -Length of each segment the signal is divided into. Default is 256 points      
        -For short signals (N < 1000): nperseg ≈ N/4      
        -For medium signals (1000 ≤ N < 10000): nperseg ≈ N/8       
        -For long signals (N ≥ 10000): nperseg ≈ N/16 or N/32
        
    *noverlap: Number of points to overlap between segments, usually set to nperseg/2 (50% overlap)
    
    *window: 'hamming', 'hann', 'blackman', 'boxcar'
        
Here the psd gives you linear, but stable values of power spectrum.

"""




# F1: Maximum power spectrum 
F1 = np.max(power_vals)  
# Maximum value in the power spectrum.


# F2: Maximum frequency 
F2 = freqs[np.argmax(power_vals)]  
# Identifies the dominant frequency in the signal ************************************************


cumulative_power = np.cumsum(power_vals)  # Cumulative power spectrum
total_power = cumulative_power[-1]  # Total power of the spectrum

# F3: Median frequency (Spectral characteristics)
F3 = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]  
# Splits the power spectrum into two equal halves.


# F4: Spectral centroid (Center of frequency)
F4 = np.sum(freqs * power_vals) / np.sum(power_vals)  
# Represents the "center of gravity" of the spectrum.



# F5: Power bandwidth (Bandwidth and energy)
threshold = 0.5 * np.max(power_vals)  # Define threshold as 50% of max power (-3 dB point)
low_cutoff = np.min(freqs[np.where(power_vals >= threshold)])  # Low-frequency -3 dB point
high_cutoff = np.max(freqs[np.where(power_vals >= threshold)])  # High-frequency -3 dB point
F5 = high_cutoff - low_cutoff  # Bandwidth of the signal



# F6: Spectral distance (Similarity between frequency components)
F6 = np.sum((power_vals - power_vals)**2)  # Replace with SECOND signal if available
# This feature compares the similarity between 2 signals in the frequency domain. 



# F7: Spectral entropy (Uncertainty in frequency)
epsilon = 1e-10  # Small constant to prevent division by zero
norm_power_vals = (power_vals + epsilon) / (np.sum(power_vals) + epsilon)  # Normalize the power spectrum
F7 = -np.sum(norm_power_vals * np.log2(norm_power_vals))  # Shannon entropy of the power spectrum



# F8: Spectral spread (Bandwidth)
F8 = np.sqrt(np.sum((freqs - F4)**2 * power_vals) / np.sum(power_vals))  
# Standard deviation of frequencies


# F9: Spectral skewness (Degree of skewness in the frequency)
F9 = np.sum((freqs - F4)**3 * power_vals) / (F8**3 * np.sum(power_vals))  
# Indicates whether the spectrum is biased toward lower or higher frequencies.


# F10: Spectral kurtosis (Degree of kurtosis in the frequency)
F10 = np.sum((freqs - F4)**4 * power_vals) / (F8**4 * np.sum(power_vals)) 
# Identifies whether the spectrum has sharp peaks (high kurtosis) or is flat (low kurtosis).


# F11: Spectral roll-off (Energy decreases with frequency)
F11 = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]  
# Frequency at 95% cumulative power


# F12: Spectral roll-on (Energy rises with frequency)
F12 = freqs[np.where(cumulative_power >= 1.15 * total_power)[0][0]] if np.any(cumulative_power >= 1.15 * total_power) else None
# Represents the frequency where the cumulative power exceeds 115% of the total energy (if applicable).


# F13: Fundamental frequency (Harmonic analysis)
F13 = freqs[np.argmax(power_vals)]  
# Indicates the dominant frequency component, often the fundamental frequency.


# F14: Human range energy: Energy in the 0.6-2.5 Hz range
F14 = np.sum(power_vals[(freqs >= 0.6) & (freqs <= 2.5)]) / np.sum(power_vals) 
# Measures the proportion of energy in the human body-related frequency range.


# F15: Spectral slope (Filter characteristics and frequency response)
H_high = np.max(power_vals[freqs >= 0.5 * fs / 2])  
H_low = np.min(power_vals[freqs >= 0.5 * fs / 2])  
K = 1  # Scaling factor (can be adjusted if needed)
F15 = K*(H_high / H_low)  # slope of the spectrum


# F16: Spectral variation (Spectral structure and frequency response)
F16 = 1 - np.sum(freqs * power_vals * power_vals) / (
    np.sqrt(np.sum(freqs * power_vals) * np.sum(freqs * power_vals))
)
# Measures the variability in the spectral structure, useful for characterizing signal stability.


# F17-F20: Band power ratios - arranged according to EEG
delta_power = np.sum(power_vals[(freqs >= 0.5) & (freqs <= 4)])  # Delta band (0.5-4 Hz)
theta_power = np.sum(power_vals[(freqs > 4) & (freqs <= 8)])  # Theta band (4-8 Hz)
alpha_power = np.sum(power_vals[(freqs > 8) & (freqs <= 13)])  # Alpha band (8-13 Hz)
beta_power = np.sum(power_vals[(freqs > 13) & (freqs <= 30)])  # Beta band (13-30 Hz)
total_power = np.sum(power_vals)  # Total power of the spectrum


F17 = delta_power / total_power  # Ratio of delta power to total power
F18 = theta_power / total_power  # Ratio of theta power to total power
F19 = alpha_power / total_power  # Ratio of alpha power to total power
F20 = beta_power / total_power  # Ratio of beta power to total power





# Print all frequency-domain features
print("\nFrequency-Domain Features:")
for j, value in enumerate([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20], start=1):
    print(f"F{j}: {value}")



plt.figure(figsize=(10,7))
plt.title("Log-scale Power Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.plot(power_vals)




#%% 3) Time-frequency domain with wavelet transform


"""
Time-frequency features analyze the signal in both time and frequency domains simultaneously.
These features are useful for non-stationary signals where frequency content changes over time.

As a hyperparameter, the mother wavelet can be changed to more appropriate one, and tried to obtain better results.
"""


# Decomposition by using DWT
W1 = pywt.wavedec(signal, 'db4', level=5)  # Perform wavelet decomposition (db4 wavelet, 5 levels)
# Contains 5 level decomposition coefficients: A5 + D5 + D4 + D3 + D2 + D1 


# Wavelet energy (TF1)
W2 = [np.sum(np.square(c)) for c in W1]  # Calculate energy for each wavelet band
TF1 = np.sum(W2)  # Total wavelet energy
# Measures the distribution of energy across different time-frequency bands.


# Wavelet entropy (TF2)
W3 = np.sum(W2)  # Total energy across all wavelet bands
W4 = entropy(W2 / W3, base=2)  # Shannon entropy of the wavelet energy distribution
TF2 = W4
# Quantifies the randomness or complexity of the wavelet energy distribution.


# TF3: Wavelet band energy ratios
TF3 = [W2[i] / W2[i + 1] for i in range(len(W2) - 1)]
# Measures the relative energy between consecutive wavelet bands.

# TF4: Wavelet coefficient variance
TF4 = [np.var(d) for d in W1]
# Captures the variability of wavelet coefficients at each level, useful for characterizing signal dynamics.






print("\nTime-Frequency Features:")
for k, value in enumerate([TF1, TF2, TF3, TF4], start=1):
    print(f"TF{k}: {value}")




# Visualizations of decomposed bands
plt.figure(figsize=(15, 15))  
plt.subplot(3, 2, 1)          
plt.plot(W1[5])               
plt.title("D1")
plt.subplot(3, 2, 2)          
plt.plot(W1[4])               
plt.title("D2")
plt.subplot(3, 2, 3)          
plt.plot(W1[3])               
plt.title("D3")
plt.subplot(3, 2, 4)          
plt.plot(W1[2])               
plt.title("D4")
plt.subplot(3, 2, 5)          
plt.plot(W1[1])               
plt.title("D5")
plt.subplot(3, 2, 6)          
plt.plot(W1[0])               
plt.title("A5")                     
plt.show() 



#%% 4) Custom functions


# You can define custom functions to use the most appropriate ones, for transforming your data.


def extract_features(signal, fs=1000):
    """
    Parameters:
    - signal: 1D array-like, the input signal.
    - fs: int, sampling frequency of the signal (default: 1000 Hz)

    Returns:
    - features: 1D numpy array
    """
    
    # Number of samples
    N = len(signal)

    # Time-domain features
    T1 = np.max(signal)
    T2 = np.min(signal)
    T3 = np.mean(signal)
    T4 = np.var(signal)
    T5 = np.std(signal)
    T6 = np.mean(np.abs(signal - T3))
    T7 = np.sqrt(np.mean(signal**2))
    T8 = np.mean(np.abs(np.diff(signal)))
    T9 = np.sum(signal**2)
    T10 = np.ptp(signal)
    T11 = np.sum(np.abs(np.diff(signal)))
    hist, bin_edges = np.histogram(signal, bins='auto', density=True)
    probabilities = hist / np.sum(hist)
    T12 = entropy(probabilities, base=2)
    T13 = np.trapz(signal)
    T14 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    T15 = np.sum(np.arange(N) * signal**2) / np.sum(signal**2)
    peaks, _ = find_peaks(signal)
    T16 = len(peaks)
    T17 = np.sum(np.sqrt(1 + np.diff(signal)**2))
    T18 = np.sum(signal**2) / N
    T19 = np.sum(signal[:-1] * signal[1:] < 0) / N
    T20 = skew(signal)
    T21 = kurtosis(signal)
    T22 = np.sum((signal[:-2] < signal[1:-1]) & (signal[1:-1] > signal[2:]))
    T23 = np.sum((signal[:-2] > signal[1:-1]) & (signal[1:-1] < signal[2:]))
    T24 = np.max(signal) - np.min(signal)
    T25 = np.sum(np.abs(np.diff(signal)))
    T26 = np.sqrt(np.var(np.diff(signal)) / T4)
    T27 = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal))) / T26

    
    fft_val = np.abs(fft(signal))
    freqs = fftfreq(len(signal), d=1/fs)
    freqs = freqs[:len(freqs)//2]
    fft_val = fft_val[:len(freqs)]
    power_vals = fft_val**2
    power_vals = 10 * np.log10(power_vals + 1e-12)
    
    
    # Frequency-domain features
    F1 = np.max(power_vals)
    F2 = freqs[np.argmax(power_vals)]
    cumulative_power = np.cumsum(power_vals)
    total_power = cumulative_power[-1]
    F3 = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]
    F4 = np.sum(freqs * power_vals) / np.sum(power_vals)
    threshold = 0.5 * np.max(power_vals)
    low_cutoff = np.min(freqs[np.where(power_vals >= threshold)])
    high_cutoff = np.max(freqs[np.where(power_vals >= threshold)])
    F5 = high_cutoff - low_cutoff
    F6 = np.sum((power_vals - power_vals)**2)
    epsilon = 1e-10
    norm_power_vals = (power_vals + epsilon) / (np.sum(power_vals) + epsilon)
    F7 = -np.sum(norm_power_vals * np.log2(norm_power_vals))
    F8 = np.sqrt(np.sum((freqs - F4)**2 * power_vals) / np.sum(power_vals))
    F9 = np.sum((freqs - F4)**3 * power_vals) / (F8**3 * np.sum(power_vals))
    F10 = np.sum((freqs - F4)**4 * power_vals) / (F8**4 * np.sum(power_vals))
    F11 = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]
    F12 = freqs[np.where(cumulative_power >= 1.15 * total_power)[0][0]] if np.any(cumulative_power >= 1.15 * total_power) else None
    F13 = freqs[np.argmax(power_vals)]
    F14 = np.sum(power_vals[(freqs >= 0.6) & (freqs <= 2.5)]) / np.sum(power_vals)
    H_high = np.max(power_vals[freqs >= 0.5 * fs / 2])
    H_low = np.min(power_vals[freqs >= 0.5 * fs / 2])
    K = 1
    F15 = K * (H_high / H_low)
    F16 = 1 - np.sum(freqs * power_vals * power_vals) / (
        np.sqrt(np.sum(freqs * power_vals) * np.sum(freqs * power_vals))
    )
    delta_power = np.sum(power_vals[(freqs >= 0.5) & (freqs <= 4)])
    theta_power = np.sum(power_vals[(freqs > 4) & (freqs <= 8)])
    alpha_power = np.sum(power_vals[(freqs > 8) & (freqs <= 13)])
    beta_power = np.sum(power_vals[(freqs > 13) & (freqs <= 30)])
    total_power = np.sum(power_vals)
    F17 = delta_power / total_power
    F18 = theta_power / total_power
    F19 = alpha_power / total_power
    F20 = beta_power / total_power

    # Time-frequency domain features
    W1 = pywt.wavedec(signal, 'db4', level=5)
    W2 = [np.sum(np.square(c)) for c in W1]
    TF1 = np.sum(W2)
    W3 = np.sum(W2)
    W4 = entropy(W2 / W3, base=2)
    TF2 = W4
    TF3 = [W2[i] / W2[i + 1] for i in range(len(W2) - 1)]
    TF4 = [np.var(d) for d in W1]

    # Combine all features into a single 1D array
    features = np.array([
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17,
        T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, F1, F2, F3, F4, F5, F6, F7,
        F8, F9, F10, F11, F13, F14, F15, F16, F17, F18, F19, F20, TF1, TF2, TF3[0], TF3[1],
        TF3[2], TF3[3], TF3[4], TF4[0], TF4[1], TF4[2], TF4[3], TF4[4], TF4[5]
    ])

    return features



# Usage
features = extract_features(signal, fs=1000)

print("Extracted Features:", features)
print("Feature Vector Length:", len(features))

plt.figure(figsize=(10,10))
plt.ylabel("Feature Values")
plt.xlabel("Feature Number")
plt.title("Feature Vector")
plt.plot(features)
plt.show()
