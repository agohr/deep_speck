from keras.models import model_from_json
from os import urandom

import speck as sp
import numpy as np

json_file = open('simple_7r_model_preproc.json','r');
json_model = json_file.read();
model = model_from_json(json_model);
model.load_weights('simple_7r_model_preproc_weights.h5');

def evaluate_ciphertexts(ct):
    ct4, ct5 = sp.dec_one_round((ct[0], ct[1]),0);
    ct6, ct7 = sp.dec_one_round((ct[2], ct[3]),0);
    X = sp.convert_to_binary([ct[0], ct[1], ct[2], ct[3], ct4, ct5, ct6, ct7]);
    Z = model.predict(X,batch_size=10000);
    return(Z.flatten());

def test(n):
    pt0a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt0b, pt1b = pt0a ^ 0x40, pt1a ^ 0x0;
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    ks = sp.expand_key(keys, 7);
    ct0a, ct1a = sp.encrypt((pt0a, pt1a),ks);
    ct0b, ct1b = sp.encrypt((pt0b, pt1b),ks);
    Z1 = evaluate_ciphertexts([ct0a, ct1a, ct0b, ct1b]);
    r = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    Z0 = evaluate_ciphertexts(r);
    tpr, tnr = np.sum(Z1 > 0.5), np.sum(Z0 < 0.5);
    tpr = tpr / n; tnr = tnr / n;
    acc = (tpr + tnr)/2;
    print("Acc: ", acc, ", TPR: ", tpr, ", TNR: ", tnr); 
    

