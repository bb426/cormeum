# =============================================================================
# 
# 1-4. Normalizing
# 
# =============================================================================



import numpy as np
def normalize_ecg(ecg):
    """
    Normalizes to a range of [-1; 1]
    Param ecg: input signal
    Return: normalized signal
    """
    ecg = ecg-np.mean(ecg)
    ecg = ecg / max(np.fabs(np.min(ecg)), np.fabs(np.max(ecg)))
    return ecg

print('Range of the first sample before implementing normalization : {}'.format(np.max(subX[0]) - np.min(subX[0])))
subX = [normalize_ecg(i) for i in subX]
print('Range of the first sample after  implementing normalization : {}'.format(np.max(subX[0]) - np.min(subX[0])))	