import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *

os.getcwd()

from infer_rbd import infer
import networks


name = "binding"

parser = argparse.ArgumentParser()

#=======================================================================================================================
parser.add_argument(
        "--model_name",
        "-mn",
        help="network for training."
        "-mn for ",
        type=str,
        default="XBCR_net",
        required=False,
    )
parser.add_argument(
        "--data_name",
        "-dn",
        help="data name for training."
        "-dn for ",
        type=str,
        default=name,
        required=False,
    )

parser.add_argument(
    "--type",
    help="Training type, full or rbd or multi",
    # default="full",
    default="rbd",
    type=str,
    required=False,
)

parser.add_argument(
    "--model_num",
    help="The model number.",
    type=int,
    default=0,
    # default=1,
)

parser.add_argument(
    "--heavy",
    help="heavy chain",
    type=str,
    default='',
)

parser.add_argument(
    "--light",
    help="light chain",
    type=str,
    default='',
    required=False
)

parser.add_argument(
    "--antig",
    help="antigen chain",
    type=str,
    default='',
)

#=======================================================================================================================
args = parser.parse_args()
#=======================================================================================================================
model_num=args.model_num

model_name=args.model_name
data_name=args.data_name
include_light=int(args.light!='')

# network setting
net_core = networks.get_net(model_name)

os.getcwd()
# print(os.getcwd())
model_path=os.path.join('.','models',data_name,data_name+'-'+model_name,'model')

shape_heavy = [300, 20]
shape_light = [300, 20]
shape_antig = [300, 20]

def seq_proc(str_in,seq_shape=[300,20],shift_loc=10,str_rep=''):
    str_in = str_in.replace(' ', str_rep)
    str_in = str_in.replace('_', str_rep)
    str_in = str_in.replace('.', str_rep)
    str_in = str_in.replace('\n', str_rep)
    str_in = str_in.replace('\t', str_rep)
    # str_lst = str_in.split(';')
    str_lst = str_in.split(',')
    seq_v = np.zeros([len(str_lst)]+seq_shape)
    for i,ss in enumerate(str_lst):
        seq_v[i,shift_loc:shift_loc+len(ss), :] = one_hot_encoder(s=ss)
    return seq_v,str_lst

# ===============================================================================
input_heavy_seq = tf.placeholder(tf.float32, [None, *shape_heavy])
input_light_seq = tf.placeholder(tf.float32, [None, *shape_light])
input_antig_seq = tf.placeholder(tf.float32, [None, *shape_antig])
net = net_core([shape_heavy, shape_light, shape_antig])
pred_bind,_=net([input_heavy_seq,input_light_seq,input_antig_seq])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
saver.restore(sess, model_path + "_rbd_" + str(model_num) + ".tf")

seq_heavy,lst_heavy=seq_proc(args.heavy,shape_heavy)
if include_light:
    seq_light,lst_light=seq_proc(args.light,shape_light)
else:
    seq_light = [seq*0 for seq in seq_heavy]
    lst_light = ['' for element in lst_heavy]
seq_antig,lst_antig=seq_proc(args.antig,shape_antig)
inferFeed = {
    input_heavy_seq: seq_heavy,
    input_light_seq: seq_light,
    input_antig_seq: seq_antig,
}
prob_bind = sess.run([pred_bind],feed_dict=inferFeed)
for ii in range(len(lst_heavy)):
    print('id: ',ii)
    print('heavy: ',lst_heavy[ii])
    print('light: ',lst_light[ii])
    print('antig: ',lst_antig[ii])
    print('score: ',prob_bind[0][ii][0])

