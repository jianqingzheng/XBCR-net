import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Input
import tensorflow as tf

########################################################################################################
eps=1e-4

def get_net(model_name=None):
    if model_name == 'XBCR_net':
        net_core = XBCR_net
    return net_core

########################################################################################################

def res_block(x_in, num_ch=16,dilations=[1,1], layers=None,in_conv=False,act=tf.nn.relu):
    ndims = 1
    if layers==None:
        Conv_nd = getattr(KL, 'Conv%dD' % ndims)
        if in_conv:
            x_in = Conv_nd(num_ch, kernel_size=1, kernel_initializer='he_normal',padding = 'same')(x_in)
        x_out = x_in
        for d in dilations:
            x_out=act(x_out)
            x_out=Conv_nd(num_ch, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=1,dilation_rate=d)(x_out)
    else:
        x_out = x_in
        for ly in layers:
            x_out = act(x_out)
            x_out = ly(x_out)
    return x_out+x_in

def make_layers(num_ch=16,dilations=[1,3,9],layer_type='conv'):
    ndims = 1
    if layer_type == 'conv':
        Conv_nd = getattr(KL, 'Conv%dD' % ndims)
    layers = []
    for d in dilations:
        if layer_type == 'conv':
            layers.append(Conv_nd(num_ch, kernel_size=3,padding='same',kernel_initializer='he_normal',strides=1,dilation_rate=d, activation=None))
        else:
            layers.append(KL.Dense(num_ch))
    return layers

def weight_share_block(inputs,num_ch=32,num_block=4,dilations=[1,3,9],layer_type='conv'):
    X=inputs
    for i in range(num_block):
        rb=make_layers(num_ch, dilations,layer_type=layer_type)
        X=[res_block(x,layers=rb) for x in X]
    return X



def XBCR_net(shapes, seqs=[None,None,None],loc_num=5,use_norm=False,training=False):
    ndims=1
    num_chain=3
    # inputs
    for i in range(num_chain):
        if seqs[i] is None:
            seqs[i] = Input(shape=shapes[i])
    enc_nf = [32, 40, 48, 56, 64]
    dec_nf = [64, 64, 32]
    num_block=[5]*len(enc_nf)
    X=seqs
    Conv_nd = getattr(KL, 'Conv%dD' % ndims)
    Norm_Layer = KL.BatchNormalization
    for i in range(len(num_block)):
        conv0 = Conv_nd(enc_nf[i], kernel_size=3, padding = 'same')
        norm0 = Norm_Layer(momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True) if use_norm else tf.identity
        X = [norm0(conv0(x)) for x in X]
        X=weight_share_block(X,num_ch=enc_nf[i],num_block=num_block[i])

    # pool_layer=KL.GlobalAveragePooling1D()
    pool_layer = KL.GlobalMaxPooling1D()
    X = [pool_layer(x) for x in X]
    X = tf.concat(X, -1)

    for nf in dec_nf:
        X=tf.nn.relu(X)
        X=KL.Dense(nf)(X)
        rb = make_layers(nf, dilations=[1,3],layer_type='dense')
        X=res_block(X,layers=rb)
    X=tf.nn.relu(X)
    rb = make_layers(nf, dilations=[1, 3], layer_type='dense')
    binding = tf.nn.sigmoid(KL.Dense(1)(tf.nn.relu(res_block(X, layers=rb))))
    rb = make_layers(nf, dilations=[1, 3], layer_type='dense')
    location = tf.nn.sigmoid(KL.Dense(loc_num)(tf.nn.relu(res_block(X, layers=rb))))
    return Model(inputs=seqs, outputs=[binding,location])
