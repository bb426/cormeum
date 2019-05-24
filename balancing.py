# =============================================================================
# 
# 1-3. Balancing
# 
# =============================================================================

import numpy as np
from collections import Counter
import matlab
from sklearn.utils import shuffle


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