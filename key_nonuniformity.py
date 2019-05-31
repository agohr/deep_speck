#this tests whether the performance of a given neural distinguisher is key dependent
#We generate some fixed keys, then calculate MSE and true positive rate for a test set
#The test set consists solely of real ciphertext pairs encrypted with the key under study

import numpy as np
import train_nets as tn
import speck as sp

from os import urandom

def tpr_fixed_key(net, n, key, nr = 7, diff = (0x40,0x0), batch_size=5000):
    pt0a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
    ks = sp.expand_key(key,nr);
    ct0a, ct1a = sp.encrypt((pt0a, pt1a),ks);
    ct0b, ct1b = sp.encrypt((pt0b, pt1b),ks);
    X = sp.convert_to_binary([ct0a, ct1a, ct0b, ct1b]);
    Z = net.predict(X,batch_size=batch_size).flatten();
    acc = np.sum(Z > 0.5)/n;
    v = 1 - Z; mse = np.mean(v*v);
    return(acc, mse);

def test(net, n=100,sample_size=10000, nr=7, diff=(0x40,0x0), batch_size=5000, filename='key_dependence_7r', test_size=1000000):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(-1,4);
    acc = np.zeros(n); mse = np.zeros(n);
    for i in range(n):
        acc[i],mse[i] = tpr_fixed_key(net, sample_size, keys[i], nr=nr, diff=diff,batch_size=batch_size);
    worst = np.argmax(mse); best = np.argmin(mse);
    #determine mse and tpr again for worst and best index
    mse_worst, acc_worst = tpr_fixed_key(net, test_size, keys[worst], nr=nr,diff=diff, batch_size=batch_size);
    mse_best, acc_best = tpr_fixed_key(net, test_size, keys[best], nr=nr, diff=diff, batch_size=batch_size);
    print("Best mse, tpr: ", mse_best, acc_best);
    print("Worst mse, tpr: ", mse_worst, acc_worst);
    print("Saving keys, mse and tpr data to disk.");
    np.save(filename+'_keys.npy',keys);
    np.save(filename+'_mse.npy', mse);
    np.save(filename+'_tpr.npy', acc);
    
