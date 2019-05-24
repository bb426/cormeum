### Project HAAS ###


# =============================================================================
# 체크할것, 데이터구조 (기록주기)
# 1초에 300번
# 관측치에 따라 30~60초 짜리 ecg 존재
# =============================================================================


#%%
### 1.Load Data
### 1-1. Import all datasets

import csv
import scipy.io as sio

# Default dir where data set is stored
__DATA_DIR = 'd://training2017'
FREQUENCY = 300


def load_data(data_path):
    data = []
    labels = []
    with open(data_path + '/REFERENCE.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            file_name = row[0]
            label = row[1]

            test = sio.loadmat(data_path + '/' + file_name + '.mat')
            content = test['val'][0]

            data.append(content)
            labels.append(label)    

    return data, labels

x, y = load_data(__DATA_DIR)


#%% Sliding windows _ temp

x = sliding_x.copy()
y = sliding_y.copy()

Counter(y)
    
#%%	
### 1-2. Categorizing

#binary ~ 3 classification
#confusion matrix


#__mapping__ = {
#    'A': 0, # AF rhythm
#    'N': 1, # Normal rhythm
#    'O': 2, # Other rhythm
#    '~': 3  # Noise rhythm  #3 class로 축소
#}

__mapping__ = {
    'A': 2, # AF rhythm
#    'N': 1, # Normal rhythm
    'O': 2, # Other rhythm
#    '~': 3  # Noise rhythm  #3 class로 축소
    '~': 2,  # Noise rhythm
    'N': 0 # Normal rhythm
}

def format_labels(labels):
    return [__mapping__[x] for x in labels]
	
y = format_labels(y)
Counter(y)
#%%
### 1-3. Balancing

import numpy as np
from collections import Counter
import matlab ##### modified
from sklearn.utils import shuffle ##### modified


def balance(x, y):
    uniq = np.unique(y)
    selected = dict()

    for val in uniq:
        selected[val] = [x[i] for i in matlab.find(y==val)]
#    min_len = 6 * min([len(x) for x in selected.values()]) 최대 6배까지 허용할 경우
    min_len = 1 * min([len(x) for x in selected.values()]) # 동일한 수로 조정
    x = []
    y = []
    for (key, value) in selected.items():
        slen = min(len(value), min_len)
        x += value[:slen]
        y += [key for i in range(slen)]

    x, y = shuffle(x, y)

    return x, y
	
subX, subY = balance(x, y)

print('------------------------------------------------------------------')
print('Before balancing it : {}\n'.format(Counter(y)))
print('After balancing it  : {}'.format(Counter(subY)))
print('------------------------------------------------------------------')






# =============================================================================
# !pip install imblearn
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(x.reshape(1, -1), y)
# print(sorted(Counter(y_resampled).items()))
# =============================================================================



#%%
### 1-4. Normalizing



#def __remove_dc_component(ecg):
#    mean = np.mean(ecg)
#    # cancel DC components
#    return ecg - mean
#
#def max_normalization(ecg):
#    return ecg / max(np.fabs(np.amin(ecg)), np.fabs(np.amax(ecg)))
#
#def normalize_ecg(ecg):
#    """
#    Normalizes to a range of [-1; 1]
#    :param ecg: input signal
#    :return: normalized signal
#    """
#    ecg = __remove_dc_component(ecg)
#    ecg = max_normalization(ecg)
#    return ecg


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




#%%
### 1-5. Feature extraction

from biosppy.signals import ecg
from scipy import stats
from scipy import signal
from scipy.fftpack import rfft
from scipy.stats import skew, kurtosis
from collections import Iterable


print("Features extraction started")

def get_features_dict(x):
    [ts, fts, rpeaks, tts, thb, hrts, hr] = ecg.ecg(signal=x, sampling_rate=FREQUENCY, show=False)

    """
    Returns:	

    ts (array) – Signal time axis reference (seconds).
    filtered (array) – Filtered ECG signal.
    rpeaks (array) – R-peak location indices.
    templates_ts (array) – Templates time axis reference (seconds).
    templates (array) – Extracted heartbeat templates.
    heart_rate_ts (array) – Heart rate time axis reference (seconds).
    heart_rate (array) – Instantaneous heart rate (bpm).
    """

    fx = dict()
    fx.update(heart_rate_features(hr))
    fx.update(frequency_powers(x, n_power_features=60))
    fx.update(add_suffix(frequency_powers(fts), "fil"))
    fx.update(frequency_powers_summary(fts))
    fx.update(heart_beats_features2(thb))
    fx.update(fft_features(median_heartbeat(thb)))
    # fx.update(heart_beats_features3(thb))
    fx.update(r_features(fts, rpeaks))

    fx['PRbyST'] = fx['PR_interval'] * fx['ST_interval']
    fx['P_form'] = fx['P_kurt'] * fx['P_skew']
    fx['T_form'] = fx['T_kurt'] * fx['T_skew']

    for key, value in fx.items():
        if np.math.isnan(value):
            value = 0
        fx[key] = value

    return fx

def heart_rate_features(hr):
    features = {
        'hr_max': 0,
        'hr_min': 0,
        'hr_mean': 0,
        'hr_median': 0,
        'hr_mode': 0,
        'hr_std': 0
    }

    if len(hr) > 0:
        features['hr_max'] = np.amax(hr)
        features['hr_min'] = np.amin(hr)
        features['hr_mean'] = np.mean(hr)
        features['hr_median'] = np.median(hr)
        features['hr_mode'] = stats.mode(hr)[0]
        features['hr_std'] = np.std(hr)

    return features



def get_feature_names(x):
    features = get_features_dict(x)
    return sorted(list(features.keys()))

def frequency_powers(x, n_power_features=40):
    fxx, pxx = signal.welch(x, FREQUENCY)
    features = dict()
    for i, v in enumerate(pxx[:n_power_features]):
        features['welch' + str(i)] = v

    return features
	
def add_suffix(dic, suffix):
    keys = list(dic.keys())
    for key in keys:
        dic[key + suffix] = dic.pop(key)
    return dic
	
def frequency_powers_summary(x):
    ecg_fs_range = (0, 50)
    band_size = 5

    features = dict()

    fxx, pxx = signal.welch(x, FREQUENCY)
    for i in range((ecg_fs_range[1] - ecg_fs_range[0]) // 5):
        fs_min = i * band_size
        fs_max = fs_min + band_size
        indices = np.logical_and(fxx >= fs_min, fxx < fs_max)
        bp = np.sum(pxx[indices])
        features["power_" + str(fs_min) + "_" + str(fs_max)] = bp

    return features
	
	
def heart_beats_features2(thb):
    means = median_heartbeat(thb)
    stds = np.array([np.std(col) for col in thb.T])

    r_pos = int(0.2 * FREQUENCY)

    PQ = means[:int(0.15 * FREQUENCY)]
    ST = means[int(0.25 * FREQUENCY):]

    QR = means[int(0.13 * FREQUENCY):r_pos]
    RS = means[r_pos:int(0.27 * FREQUENCY)]

    q_pos = int(0.13 * FREQUENCY) + np.argmin(QR)
    s_pos = r_pos + np.argmin(RS)

    p_pos = np.argmax(PQ)
    t_pos = np.argmax(ST)

    t_wave = ST[max(0, t_pos - int(0.08 * FREQUENCY)):min(len(ST), t_pos + int(0.08 * FREQUENCY))]
    p_wave = PQ[max(0, p_pos - int(0.06 * FREQUENCY)):min(len(PQ), p_pos + int(0.06 * FREQUENCY))]

    r_plus = sum(1 if b[r_pos] > 0 else 0 for b in thb)
    r_minus = len(thb) - r_plus

    QRS = means[q_pos:s_pos]

    a = dict()
    a['PR_interval'] = r_pos - p_pos
    a['P_max'] = PQ[p_pos]
    a['P_to_R'] = PQ[p_pos] / means[r_pos]
    a['P_to_Q'] = PQ[p_pos] - means[q_pos]
    a['ST_interval'] = t_pos
    a['T_max'] = ST[t_pos]
    a['R_plus'] = r_plus / max(1, len(thb))
    a['R_minus'] = r_minus / max(1, len(thb))
    a['T_to_R'] = ST[t_pos] / means[r_pos]
    a['T_to_S'] = ST[t_pos] - means[s_pos]
    a['P_to_T'] = PQ[p_pos] / ST[t_pos]
    a['P_skew'] = skew(p_wave)
    a['P_kurt'] = kurtosis(p_wave)
    a['T_skew'] = skew(t_wave)
    a['T_kurt'] = kurtosis(t_wave)
    a['activity'] = calcActivity(means)
    a['mobility'] = calcMobility(means)
    a['complexity'] = calcComplexity(means)
    a['QRS_len'] = s_pos - q_pos

    qrs_min = abs(min(QRS))
    qrs_max = abs(max(QRS))
    qrs_abs = max(qrs_min, qrs_max)
    sign = -1 if qrs_min > qrs_max else 1

    a['QRS_diff'] = sign * abs(qrs_min / qrs_abs)
    a['QS_diff'] = abs(means[s_pos] - means[q_pos])
    a['QRS_kurt'] = kurtosis(QRS)
    a['QRS_skew'] = skew(QRS)
    a['QRS_minmax'] = qrs_max - qrs_min
    a['P_std'] = np.mean(stds[:q_pos])
    a['T_std'] = np.mean(stds[s_pos:])

    return a


def median_heartbeat(thb):
    if len(thb) == 0:
        return np.zeros((int(0.6 * FREQUENCY)), dtype=np.int32)

    m = [np.median(col) for col in thb.T]

    dists = [np.sum(np.square(s - m)) for s in thb]
    pmin = np.argmin(dists)

    median = thb[pmin]

    r_pos = int(0.2 * FREQUENCY)
    if median[r_pos] < 0:
        median *= -1

    return median

def calcActivity(epoch):
    """
    Calculate Hjorth activity over epoch
    """
    return np.nanvar(epoch, axis=0)
	
def calcMobility(epoch):
    """
    Calculate the Hjorth mobility parameter over epoch
    """
    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    return np.divide(
        np.nanstd(np.diff(epoch, axis=0)),
        np.nanstd(epoch, axis=0))
		
		
def calcComplexity(epoch):
    """
    Calculate Hjorth complexity over epoch
    """
    return np.divide(
        calcMobility(np.diff(epoch, axis=0)),
        calcMobility(epoch))
		
def fft_features(beat):
    pff = extract_fft(beat[:int(0.13 * FREQUENCY)])
    rff = extract_fft(beat[int(0.13 * FREQUENCY):int(0.27 * FREQUENCY)])
    tff = extract_fft(beat[int(0.27 * FREQUENCY):])

    features = dict()
    for i, v in enumerate(pff[:10]):
        features['pft' + str(i)] = v

    for i, v in enumerate(rff[:10]):
        features['rft' + str(i)] = v

    for i, v in enumerate(tff[:20]):
        features['tft' + str(i)] = v

    return features
	
	
def extract_fft(x):
    return rfft(x)[:len(x) // 2]
	
	
def r_features(s, r_peaks):
    r_vals = [s[i] for i in r_peaks]

    times = np.diff(r_peaks)
    avg = np.mean(times)
    filtered = sum([1 if i < 0.5 * avg else 0 for i in times])

    total = len(r_vals) if len(r_vals) > 0 else 1

    data = time_domain(times)

    data['beats_to_length'] = len(r_peaks) / len(s)
    data['r_mean'] = np.mean(r_vals)
    data['r_std'] = np.std(r_vals)
    data['filtered_r'] = filtered
    data['rel_filtered_r'] = filtered / total

    return data
	
	
def time_domain(rri: Iterable):
    """
    Computes time domain characteristics of heart rate:

    - RMSSD, Root mean square of successive differences
    - NN50, Number of pairs of successive NN intervals that differ by more than 50ms
    - pNN50, Proportion of NN50 divided by total number of NN intervals
    - NN20, Number of pairs of successive NN intervals that differ by more than 20ms
    - pNN20, Proportion of NN20 divided by total number of NN intervals
    - SDNN, standard deviation of NN intervals
    - mRRi, mean length of RR interval
    - stdRRi, mean length of RR intervals
    - mHR, mean heart rate

    :param rri: RR intervals in ms
    :return: dictionary with computed characteristics
    :rtype: dict
    """
    rmssd = 0
    sdnn = 0
    nn20 = 0
    pnn20 = 0
    nn50 = 0
    pnn50 = 0
    mrri = 0
    stdrri = 0
    mhr = 0

    if len(rri) > 0:
        diff_rri = np.diff(rri)
        if len(diff_rri) > 0:
            # Root mean square of successive differences
            rmssd = np.sqrt(np.mean(diff_rri ** 2))
            # Number of pairs of successive NNs that differ by more than 50ms
            nn50 = sum(abs(diff_rri) > 50)
            # Proportion of NN50 divided by total number of NNs
            pnn50 = (nn50 / len(diff_rri)) * 100

            # Number of pairs of successive NNs that differe by more than 20ms
            nn20 = sum(abs(diff_rri) > 20)
            # Proportion of NN20 divided by total number of NNs
            pnn20 = (nn20 / len(diff_rri)) * 100

        # Standard deviation of NN intervals
        sdnn = np.std(rri, ddof=1)  # make it calculates N-1
        # Mean of RR intervals
        mrri = np.mean(rri)
        # Std of RR intervals
        stdrri = np.std(rri)
        # Mean heart rate, in ms
        mhr = 60 * 1000.0 / mrri

    keys = ['rmssd', 'sdnn', 'nn20', 'pnn20', 'nn50', 'pnn50', 'mrri', 'stdrri', 'mhr']
    values = [rmssd, sdnn, nn20, pnn20, nn50, pnn50, mrri, stdrri, mhr]
    values = np.round(values, 2)
    values = np.nan_to_num(values)

    return dict(zip(keys, values))
	
def features_for_row(x):
    features = get_features_dict(x)
    return np.array([features[key] for key in sorted(list(features.keys()))], dtype=np.float32)
	
	
	
fn = get_feature_names(subX[0])
subX = [features_for_row(i) for i in subX]

print('number of columns before feature extraction : {}'.format(len(x[0])))
print('number of features : {}'.format(len(fn)))
print('number of columns after feature extraction : {}'.format(subX[0].shape))


#%% 2. Modeling


##%% 2-1. Model evaluation
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#
#
#def print_validation_info(trueY, predY):
#    print(classification_report(trueY, predY))
#
#    print('Confusion matrix:')
#    print(confusion_matrix(trueY, predY))


##%% 2-1. EcgModel
#import math
#import numpy as np
#
#class EcgModel:
#    def name(self):
#        return type(self).__name__.lower()
#
#    def restore(self):
#        raise NotImplementedError()
#
#    def fit(self, x, y, validation=None):
#        raise NotImplementedError()
#
#    def predict(self, x):
#        raise NotImplementedError()
#
#    def evaluate(self, x, y_true):
#        y_pred = self.predict(x)
#        evaluation.print_validation_info(y_true, y_pred)
#
#    @staticmethod
#    def get_class_weights(y, mu=0.8):
#        """
#        :param y: labels
#        :param mu: parameter to tune
#        :return: class weights dictionary
#        """
#        train_categories_dist = dict()
#        labels = np.unique(y)
#        for label in labels:
#            train_occurancies = sum([1 if label == y else 0 for y in Y])
#            train_categories_dist[label] = train_occurancies
#
#        total = sum(train_categories_dist.values())
#        keys = train_categories_dist.keys()
#        class_weight = dict()
#
#        for key in keys:
#            score = math.log(mu * total / float(train_categories_dist[key]))
#            class_weight[key] = score if score > 1.0 else 1.0
#
#        return class_weight
#
#
##%% 2-2. EcgModel
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
#
#class RandomForestEcgModel(EcgModel):
#    rf = None
#    model_file = "model.pkl"
#
#    def restore(self):
#        self.rf = joblib.load(self.model_file)
#
#    def fit(self, x, y, validation=None):
#        self.rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1)
#        self.rf.fit(x, y)
#        joblib.dump(self.rf, self.model_file)
#
#    def predict(self, x):
#        return self.rf.predict(x)
#
#    def show_feature_importances(self, features_names=None):
#        import matplotlib.pyplot as plt
#        import numpy as np
#
#        importances = self.rf.feature_importances_
#        std = np.std([tree.feature_importances_ for tree in self.rf.estimators_],
#                     axis=0)
#        indices = np.argsort(importances)[::-1]
#
#        if features_names is None:
#            labels = indices
#        else:
#            labels = [features_names[i] for i in indices]
#
#        plt.figure()
#        plt.title("Feature importances")
#        plt.bar(range(len(importances)), importances[indices],
#                color="r", yerr=std[indices], align="center")
#        plt.xticks(range(len(importances)), labels, rotation='vertical', fontsize=10)
#        plt.subplots_adjust(bottom=0.3)
#        plt.xlim([-1, len(importances)])
#        plt.show()
#
#
##%% 2-4. Get Model 
#        
#def get_raw_model(input_shape=None):
#    return RandomForestEcgModel()
#
#
#def get_saved_model(input_shape=None):
#    model = get_raw_model(input_shape)
#    model.restore()
#    return model


#%% 2-1. Train the model


#def train(subX, subY, fn):
#    X_train, X_val, y_train, y_val = train_test_split(subX, subY, test_size=0.2)
#
#    input_shape = subX.shape[1:]
#    model = get_raw_model(input_shape)
#    model.fit(X_train, y_train, validation=(X_val, y_val))
#    model.evaluate(X_val, y_val)
#
#train(subX, subY, fn)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(subX, subY, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_test, y_test)


# fixing overfitting

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 10, 20, 50, 100]}

grid = GridSearchCV(RandomForestClassifier(n_estimators=1000, 
                                           n_jobs=-1, 
                                           max_features=10),
        param_grid=param_grid, cv=5)
                                            
        
grid.fit(X_train, y_train)


grid.score(X_train, y_train)
grid.score(X_test, y_test)


#balancing, params..

print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))