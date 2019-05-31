import train_nets as tn
import speck as sp
import numpy as np

from pickle import dump

from os import urandom

from keras import backend as K

WORD_SIZE = sp.WORD_SIZE();

def key_average(ct0a, ct1a, ct0b, ct1b, keys, net):
    n = len(keys);
    pt0a, pt1a = sp.dec_one_round((ct0a, ct1a),keys);
    pt0b, pt1b = sp.dec_one_round((ct0b, ct1b),keys);
    X = sp.convert_to_binary([pt0a, pt1a, pt0b, pt1b]);
    Z = net.predict(X,batch_size=10000);
    Z = Z/(1-Z); v = np.average(Z);
    v = v/(v+1);
    return(v);

def make_testset(n, nr=7, diff = (0x40,0x0)):
  Y = np.frombuffer(urandom(n),dtype=np.uint8); Y = Y & 1;
  pt0a = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt1a = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  ks = sp.expand_key(keys, nr);
  ct0a, ct1a = sp.encrypt((pt0a, pt1a),ks);
  ct0b, ct1b = sp.encrypt((pt0b, pt1b),ks);
  num_rnd = np.sum(Y==0);
  ct0b[Y==0] = np.frombuffer(urandom(2*num_rnd),dtype=np.uint16);
  ct1b[Y==0] = np.frombuffer(urandom(2*num_rnd),dtype=np.uint16);
  return([ct0a, ct1a, ct0b, ct1b], Y);

def make_trainset_with_teacher(n, net, nr=7, diff=(0x40, 0x0), num_keys=1000, keys=None):
    change_keys = (keys is None);
    X,Y = make_testset(n, nr=nr, diff=diff);
    Z = np.zeros(n);
    allkeys = np.arange(0,2**WORD_SIZE,dtype=np.uint16);
    for i in range(n):
        if change_keys: keys = np.random.choice(allkeys, num_keys);
        Z[i] = key_average(X[0][i], X[1][i], X[2][i], X[3][i], keys, net);
    return(X,Y,Z);
