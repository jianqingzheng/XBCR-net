import numpy as np
import pandas as pd
import glob
import os

# alphabet_dict="ACDEFGHIKLMNPQRSTVWY"
ALPHABET="ACDEFGHIKLMNPQRSTVWY"

def one_hot_encoder(s,  alphabet=ALPHABET):
    # Build dictionary
    d = {a: i for i, a in enumerate(alphabet)}

    # Encode
    x = np.zeros((len(s), len(d)+1))
    x[range(len(s)),[d[c] if c in alphabet else len(d) for c in s]] = 1
    if any(x[:,len(d)]>0):
        print(s)
    return x[:,:len(d)]


# access data
def vec_shift(vec,random_shift=False,seq_shift=20):
    shift = np.random.randint(0, vec[1]) if random_shift else seq_shift
    vec_tmp = np.zeros_like(vec[0])
    vec_tmp[shift:shift + vec[-1], :] = vec[0][:vec[-1], :]
    return vec_tmp

def get_seq_data(data, idx=[[0],[0]],rand_shift=False,repeat=1):
    if repeat <= 1:
        return np.concatenate([np.stack([vec_shift(d[i], rand_shift) for i in id], 0) for d, id in zip(data, idx)], 0)
    else:
        return np.concatenate([np.stack([vec_shift(data[i], rand_shift) for i in idx], 0)] * repeat, 0)

def read_files(folder_dir,suffix='*.xlsx'):
    if 'xlsx' in suffix:
        reader = pd.read_excel
        pths = glob.glob(os.path.join(folder_dir, suffix))
        return [reader(pth, engine='openpyxl') for pth in pths]
    else:
        reader = pd.read_csv
        pths = glob.glob(os.path.join(folder_dir, suffix))
        return [reader(pth, encoding='latin-1') for pth in pths]
