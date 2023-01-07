import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import datetime
import pandas as pd
import os

from utils import *

np.random.seed(1)
eps = 1e-7
##############################################################################################
def train(net_core=None, model_path=None,model_num=None,include_light=1,pos_path=None, neg_path=None, batch_size=None,nb_epochs1=None):

    def data_process(data, header=[''], seq_length=[300], min_seq_length=10, str_rep='',
                     input_loc=True):
        seq_vecs = [[] for _ in range(len(header))]
        seq_max_length = 0
        if input_loc:
            # loc_names = ['rbd','s1_not_rbd','s2','ntd','spike']
            loc_names = ['rbd', 'not_rbd']
        bind_locs = []

        for d in data:
            seq_num = len(d[header[0]])
            for i in range(seq_num):
                seqs = [str(d[h].loc[i]) for h in header]
                # if True:
                outrange=any([len(s) > length for s,length in zip(seqs,seq_length)])

                if all([len(s) > min_seq_length for s in seqs]) and not outrange:
                    if outrange:
                        # print(seqs,'out of range')
                        print(len(seqs[-1]), 'out of range')

                    if input_loc:
                        out_vec=[d[loc_h].loc[i] for loc_h in loc_names]
                    else:
                        out_vec=[0 for _ in range(2)]

                    if any(out_vec) or not input_loc:
                        bind_locs.append([all([out_vec[0],1-out_vec[-1]])])
                        for j, seq in enumerate(seqs):
                            seq = seq.replace(' ', str_rep)
                            seq = seq.replace('\n', str_rep)
                            seq = seq.replace('\t', str_rep)
                            seq = seq.replace('_', str_rep)
                            seq_v = np.zeros([seq_length[j], 20])
                            seq_v[0:len(seq), :] = one_hot_encoder(s=seq)
                            seq_vecs[j].append([seq_v, seq_length[j] - len(seq), len(seq)])
                            seq_max_length = max(seq_max_length, len(seq))
        if input_loc:
            bind_locs = np.array(bind_locs)
        else:
            bind_locs = None
        print(seq_max_length)
        return seq_vecs, bind_locs

    def accuracy_cls(pred,gt):
        pred_logic=tf.cast(pred>0.5,dtype='float32')
        return tf.reduce_mean(pred_logic*gt+(1-pred_logic)*(1-gt))

    def get_confusion_mat(pred,gt):
        pred_logic = np.round(pred)
        TP = np.mean(pred_logic * gt)
        TN = np.mean((1 - pred_logic) * (1 - gt))
        FP = np.mean((pred_logic) * (1 - gt))
        FN = np.mean((1 - pred_logic) * (gt))
        return [TP,FP,FN,TN]

    eval_name=['acc','sns','spc','ppv','npv','bac']
    def eval_confusion(conf_mat,eps=eps):
        acc = np.sum(conf_mat[0] + conf_mat[3]) / (np.sum(conf_mat) + eps)
        sns = np.sum(conf_mat[0]) / np.sum(conf_mat[0] + conf_mat[2] + eps)
        spc = np.sum(conf_mat[3]) / np.sum(conf_mat[1] + conf_mat[3] + eps)
        ppv = np.sum(conf_mat[0]) / np.sum(conf_mat[0] + conf_mat[1] + eps)
        npv = np.sum(conf_mat[3]) / np.sum(conf_mat[2] + conf_mat[3] + eps)
        bac = (sns+spc)/2
        return [acc,sns,spc,ppv,npv,bac]

    restore_pre_train = True
    # summary_error = False
    summary_error = True
    init_lr = 0.0001
    nb_print_steps = 30
    shape_heavy = [300, 20]
    shape_light = [300, 20]
    shape_antig = [300, 20]

    suffix='*.xlsx' #'*.csv'

    pos_data = read_files(pos_path,suffix)
    neg_data = read_files(neg_path,suffix)

    [seq_heavy_pos, seq_light_pos, seq_antig_pos], rbd_bind = data_process(pos_data, ['Heavy', 'Light', 'variant_seq'], seq_length=[shape_heavy[0], shape_light[0], shape_antig[0]], input_loc=True)
    [seq_heavy_neg, seq_light_neg], _ = data_process(neg_data, ['Heavy', 'Light'], seq_length=[shape_heavy[0], shape_light[0]], input_loc=False)
    if not include_light:
        seq_light_pos=[[np.zeros_like(X[0]),1,200] for X in seq_light_pos]
        seq_light_neg=[[np.zeros_like(X[0]),1,200] for X in seq_light_neg]
    rbd_neg = np.zeros([batch_size, 1], dtype=float)
    # ===============================================================================
    input_heavy_seq = tf.placeholder(tf.float32, [None, *shape_heavy])
    input_light_seq = tf.placeholder(tf.float32, [None, *shape_light])
    input_antig_seq = tf.placeholder(tf.float32, [None, *shape_antig])
    output_bind_lab = tf.placeholder(tf.float32, [None, 1])

    net = net_core([shape_heavy, shape_light, shape_antig])

    loss_func=tf.keras.losses.BinaryCrossentropy(from_logits=False)

    pred_bind,_=net([input_heavy_seq,input_light_seq,input_antig_seq])

    acc_cls = accuracy_cls(pred_bind,output_bind_lab)
    loss_cls = loss_func(output_bind_lab, pred_bind)
    loss_tot = 0.
    loss_tot += loss_cls

    # train_op = tf.train.MomentumOptimizer(learning_rate=init_lr, momentum=0.5).minimize(loss_tot)
    train_op = tf.train.AdamOptimizer(init_lr).minimize(loss_tot)

    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if restore_pre_train:
        saver.restore(sess, model_path + "_rbd_" + str(model_num) + ".tf")
    save_path = model_path + "_rbd_" + str(model_num) + ".tf"
    print(save_path)


    if summary_error:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # time.asctime(time.gmtime())
        train_log_dir = os.path.join(model_path + '_log', current_time, 'train')
        valid_log_dir = os.path.join(model_path + '_log', current_time, 'valid')
        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        valid_summary_writer = tf.summary.FileWriter(valid_log_dir, sess.graph)

    # training
    loss_min = 1e10
    acc_max = 0
    conf_bst = None
    train_rate=.8

    num_pos=len(seq_heavy_pos)
    num_neg=len(seq_heavy_neg)
    num_sample = int(min(num_pos,num_neg)*train_rate)
    num_select = int(min(num_pos* (1-train_rate), num_neg* (1-train_rate)) )
    split_pos = np.floor(num_pos * train_rate)
    split_neg = np.floor(num_neg * train_rate)
    print('sample data:',num_sample,' training positive:',split_pos,' training negative:',split_neg)
    idx_pos = list(range(int(split_pos)))
    idx_neg = list(range(int(split_neg)))
    idx_neg2 = list(range(int(split_neg)))
    idv_pos = list(range(int(split_pos),int(num_pos)))
    idv_neg = list(range(int(split_neg),int(num_neg)))

    for epoch in range(nb_epochs1):
        avg_loss, avg_cls, avg_acc = 0, 0, 0
        avg_loss_val, avg_cls_val, avg_acc_val = 0, 0, 0
        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)
        np.random.shuffle(idx_neg2)
        for step in range(num_sample//batch_size):
            # train_input, train_output = next(train_generator)
            idx=[idx_pos[step*batch_size:(step+1)*batch_size],idx_neg[step*batch_size:(step+1)*batch_size]]
            idx2 = [idx_pos[step * batch_size:(step + 1) * batch_size], idx_neg2[step * batch_size:(step + 1) * batch_size]]
            gt_bind = np.concatenate([rbd_bind[idx[0]], rbd_neg])
            trainFeed = {input_heavy_seq: get_seq_data([seq_heavy_pos,seq_heavy_neg],idx,rand_shift=True),
                         input_light_seq: get_seq_data([seq_light_pos, seq_light_neg], idx2,rand_shift=True),
                         input_antig_seq: get_seq_data(seq_antig_pos,idx[0],repeat=2,rand_shift=True),
                         output_bind_lab: gt_bind,
                         }
            sess.run(train_op, feed_dict=trainFeed)

            if step % nb_print_steps == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                loss, err_cls, acc_cls_train, prob_bind = sess.run(
                    [loss_tot, loss_cls, acc_cls, pred_bind],
                    feed_dict=trainFeed)
                step_val = step // nb_print_steps
                avg_loss = (avg_loss * (step_val) + loss * 1) / (step_val + 1)
                avg_cls = (avg_cls * (step_val) + err_cls * 1) / (step_val + 1)
                avg_acc = (avg_acc * (step_val) + acc_cls_train * 1) / (step_val + 1)
                print(
                    'Epoch %d/%d Step %d [%s]: Loss:%f=%f(cls); acc: %f; avg: loss=%f, cls=%f, acc=%f;' %
                    (epoch, nb_epochs1, step,
                     current_time,
                     loss,
                     err_cls,
                     acc_cls_train,
                     avg_loss,
                     avg_cls,
                     avg_acc,
                     ))
                if summary_error:
                    conf_mat = get_confusion_mat(prob_bind, gt_bind)
                    eval_criteria=eval_confusion(conf_mat)
                    train_summary = tf.compat.v1.Summary()
                    train_summary.value.add(tag='loss', simple_value=loss)

                    [train_summary.value.add(tag=eval_name[id], simple_value=eval_crition) for id,eval_crition in enumerate(eval_criteria)]
                    train_summary_writer.add_summary(train_summary, step)
                    train_summary_writer.flush()
                    del train_summary


                # valid
                np.random.shuffle(idv_pos)
                np.random.shuffle(idv_neg)

                idv = [idv_pos[0 * batch_size:(0 + 1) * batch_size],
                       idv_neg[0 * batch_size:(0 + 1) * batch_size]]
                gt_bind = np.concatenate([rbd_bind[idv[0]], rbd_neg])
                valFeed = {input_heavy_seq: get_seq_data([seq_heavy_pos, seq_heavy_neg], idv,rand_shift=True),
                           input_light_seq: get_seq_data([seq_light_pos, seq_light_neg], idv,rand_shift=True),
                           input_antig_seq: get_seq_data(seq_antig_pos, idv[0], repeat=2,rand_shift=True),
                           output_bind_lab: gt_bind,
                           }
                loss, err_cls, acc_cls_val, prob_bind = sess.run(
                    [loss_tot, loss_cls, acc_cls, pred_bind],
                    feed_dict=valFeed)
                step_val=step//nb_print_steps
                avg_loss_val = (avg_loss_val * (step_val) + loss * 1) / (step_val + 1)
                avg_cls_val = (avg_cls_val * (step_val) + err_cls * 1) / (step_val + 1)
                avg_acc_val = (avg_acc_val * (step_val) + acc_cls_val * 1) / (step_val + 1)
                print(
                    '                                       val: Loss:%f=%f(cls); acc: %f; avg: loss=%f, cls=%f, acc=%f;' %
                    (
                     loss,
                     err_cls,
                     acc_cls_val,
                     avg_loss_val,
                     avg_cls_val,
                     avg_acc_val,
                     ))
                if summary_error:
                    conf_mat = get_confusion_mat(prob_bind, gt_bind)
                    eval_criteria=eval_confusion(conf_mat)
                    valid_summary = tf.compat.v1.Summary()
                    valid_summary.value.add(tag='loss', simple_value=loss)

                    [valid_summary.value.add(tag=eval_name[id], simple_value=eval_crition) for id,eval_crition in enumerate(eval_criteria)]
                    valid_summary_writer.add_summary(valid_summary, step)
                    valid_summary_writer.flush()
                    del valid_summary
                #check the model on the selected data
                if acc_max <= avg_acc_val*1.1:
                    avg_loss_sel, avg_acc_sel = 0, 0
                    avg_conf_mat=[0,0,0,0]
                    for step_sel in range(min(num_select // batch_size,100)):
                        idv = [idv_neg[step_sel * batch_size:(step_sel + 1) * batch_size],
                               idv_pos[step_sel * batch_size:(step_sel + 1) * batch_size]]
                        gt_bind = np.concatenate([rbd_neg, rbd_bind[idv[1]]])
                        valFeed = {input_heavy_seq: get_seq_data([seq_heavy_neg, seq_heavy_pos], idv,rand_shift=True),
                                   input_light_seq: get_seq_data([seq_light_neg, seq_light_pos], idv,rand_shift=True),
                                   input_antig_seq: get_seq_data(seq_antig_pos,idv[1],repeat=2,rand_shift=True),
                                   output_bind_lab: gt_bind,
                                   }
                        loss, err_cls, acc_cls_sel, prob_bind = sess.run(
                            [loss_tot, loss_cls, acc_cls, pred_bind],
                            feed_dict=valFeed)
                        conf_mat=get_confusion_mat(prob_bind,gt_bind)
                        avg_conf_mat=[(avg_val * (step_sel) + val * 1) / (step_sel + 1) for val,avg_val in zip(conf_mat,avg_conf_mat)]
                        avg_loss_sel = (avg_loss_sel * (step_sel) + loss * 1) / (step_sel + 1)
                        avg_acc_sel = (avg_acc_sel * (step_sel) + acc_cls_sel * 1) / (step_sel + 1)
                    if acc_max < avg_acc_sel:
                        loss_min = avg_loss_sel
                        acc_max = avg_acc_sel
                        conf_bst = avg_conf_mat
                        save_path = saver.save(sess, save_path, write_meta_graph=False)
                        print("New model saved in: %s" % save_path)
                    else:
                        print("latest loss: %f, acc cls: %f" % (avg_loss_sel, avg_acc_sel))
                        print(eval_confusion(avg_conf_mat))
                # else:
                print("minimal loss: %f, acc cls: %f" % (loss_min, acc_max))
                print(eval_confusion(conf_bst)) if conf_bst is not None else None
    sess.close()
