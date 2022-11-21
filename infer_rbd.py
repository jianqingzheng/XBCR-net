import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import pandas as pd

from utils import *

np.random.seed(1)
eps = 1e-5

##############################################################################################
def infer(net_core=None, model_path=None,model_num=None,result_path=None,suffix_save=None,include_light=None, antig_path=None, antib_path=None,batch_size=1):

    def data_process(data, header=[''], seq_length=[300], min_seq_length=10, str_rep=''):
        seq_vecs = [[] for _ in range(len(header))]
        seq_max_length = 0
        out_data = pd.Series({})
        drop_idx = []
        for d in data:
            dtmp = d
            seq_num = len(d[header[0]])
            drop_name = ['rbd']
            for i in range(seq_num):
                seqs = [str(d[h].loc[i]) for h in header]
                rbd_binding = all([dn not in d or d[dn].loc[i] > 0 for dn in drop_name])
                if all([len(s) > min_seq_length for s in seqs]) and rbd_binding:
                    if True:
                        for j, seq in enumerate(seqs):
                            seq = seq.replace(' ', str_rep)
                            seq = seq.replace('\n', str_rep)
                            seq = seq.replace('\t', str_rep)
                            seq_v = np.zeros([seq_length[j], 20])
                            seq_v[0:len(seq), :] = one_hot_encoder(s=seq)
                            seq_vecs[j].append([seq_v, seq_length[j] - len(seq), len(seq)])
                            seq_max_length = max(seq_max_length, len(seq))
                else:
                    drop_idx.append(i)
            dtmp.drop(dtmp.index[drop_idx], inplace=True)
            out_data = pd.concat([out_data, dtmp])
        print(seq_max_length)
        return seq_vecs, out_data

    restore_pre_train = True
    suffix='*.xlsx' #'*.csv'
    reader=pd.read_excel

    shape_heavy = [300, 20]
    shape_light = [300, 20]
    shape_antig = [300, 20]

    # antig_pths = [os.path.join(antig_path, 'train_seq_' + str(0) + '.xlsx')]
    antig_pths = glob.glob(os.path.join(antig_path, suffix))
    print(antig_path)

    antig_data=[reader(pth,engine='openpyxl') for pth in antig_pths]
    antib_pths=glob.glob(os.path.join(antib_path, suffix))
    antib_data=[reader(pth,engine='openpyxl') for pth in antib_pths]

    antig_data = [df.drop_duplicates(subset=['variant_name','variant_seq','rbd'], keep='first').reset_index(drop=False) for df in antig_data]
    [seq_antig], antig_series = data_process(antig_data, ['variant_seq'], seq_length=[shape_antig[0]])
    if include_light:
        [seq_heavy, seq_light], antib_series = data_process(antib_data, ['Heavy', 'Light'], seq_length=[shape_heavy[0], shape_light[0]])
    else:
        [seq_heavy], antib_series = data_process(antib_data, ['Heavy'], seq_length=[shape_heavy[0], shape_light[0]])
        seq_light=[[np.zeros_like(X[0]),0,200] for X in seq_heavy]

    num_heavy_light = len(seq_heavy)
    num_antig = len(seq_antig)
    num_sample = num_heavy_light * num_antig
    combinations = np.array(np.meshgrid(range(num_antig), range(num_heavy_light),indexing='ij')).reshape([2,-1])
    out_data = pd.concat(
        [antig_series.loc[antig_series.index[combinations[0]]].reset_index(), antib_series.loc[antib_series.index[combinations[1]]].reset_index()],
        # keys=['a', 'b'],
        axis=1,
    )
    # ===============================================================================
    input_heavy_seq = tf.placeholder(tf.float32, [None, *shape_heavy])
    input_light_seq = tf.placeholder(tf.float32, [None, *shape_light])
    input_antig_seq = tf.placeholder(tf.float32, [None, *shape_antig])

    net = net_core([shape_heavy, shape_light, shape_antig])


    pred_bind,_=net([input_heavy_seq,input_light_seq,input_antig_seq])

    # restore the trained weights
    # saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if restore_pre_train:
        saver.restore(sess, model_path + "_rbd_" + str(model_num) + ".tf")
    save_path = model_path + "_rbd_" + str(model_num) + ".tf"
    print(save_path)

    print('sample data:',num_sample,' heavy_light:',num_heavy_light,' antig:',num_antig)
    idx_antig = list(range(num_antig))
    idx_heavy_light = list(range(num_heavy_light))


    prob_array = []

    for idx_a in idx_antig:
        for idx_hl in idx_heavy_light:
            inferFeed = {input_heavy_seq: get_seq_data([seq_heavy], [[idx_hl]],rand_shift=False),
                         input_light_seq: get_seq_data([seq_light], [[idx_hl]],rand_shift=False),
                         input_antig_seq: get_seq_data([seq_antig], [[idx_a]], repeat=1,rand_shift=False),
                         }
            prob_bind = sess.run([pred_bind],feed_dict=inferFeed)
            prob_array.append(prob_bind[0])
            # ===========================================================================================================
    prob_array=np.concatenate(prob_array,axis=0)
    out_data['pred_prob'] = prob_array

    if suffix_save=='.csv':
        writer = pd.DataFrame.to_csv
        writer(out_data, path_or_buf=result_path, sep=',', index=False)
    else:
        out_data.to_excel(result_path)
    sess.close()