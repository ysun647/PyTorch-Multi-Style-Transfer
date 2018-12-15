import numpy as np

def smooth(arr, smooth_policy=[0.05, 0.2, 0.5, 0.2, 0.05], middle=2):
    assert sum(smooth_policy) == 1
    assert 0 <= middle < len(smooth_policy)
    
    res = [None for _ in range(len(arr))]
    for i, num in enumerate(arr):
        aug_ratio = 0.0
        temp_res = 0
        for j in range(len(smooth_policy)):
            if i + (j-middle) < 0:
                aug_ratio += smooth_policy[j]
            elif i + j - middle >= len(arr):
                aug_ratio += smooth_policy[j]
            else:
                temp_res += arr[i+j-middle] * smooth_policy[j]
        print(i, num, temp_res, aug_ratio)
        temp_res = temp_res / (1 - aug_ratio)
        res[i] = temp_res
    
    return np.array(res)

print(smooth([2,1,2,1.5,2]))
