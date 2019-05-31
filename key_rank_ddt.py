from os import urandom

import numpy as np
import speck as sp

def key_rank_one_round(nr, dist, n_blocks=1, diff=(0x0040,0x0)):
  pt0a = np.frombuffer(urandom(2*n_blocks),dtype=np.uint16).reshape(n_blocks,-1);
  pt1a = np.frombuffer(urandom(2*n_blocks),dtype=np.uint16).reshape(n_blocks,-1);
  pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a),0);
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b),0);
  key = np.frombuffer(urandom(8),dtype=np.uint16);
  ks = sp.ex_key_test(key, nr); k1 = ks[nr-1];
  ct0a, ct1a = sp.enc_test((pt0a, pt1a), ks);
  ct0b, ct1b = sp.enc_test((pt0b, pt1b), ks);
  trial_keys = np.arange(2**16);
  c0a, c1a = sp.dec_one_round((ct0a, ct1a), trial_keys); c0b, c1b = sp.dec_one_round((ct0b, ct1b), trial_keys);
  d0,d1 = c0a ^ c0b, c1a ^ c1b;
  d = d0 ^ (d1.astype(np.uint32) << 16);
  Z = dist[d]; Z = np.log2(Z); Z = np.sum(Z,axis=0);
  rank0 = np.sum(Z > Z[k1]); rank1 = np.sum(Z >= Z[k1]);
  return(rank0, rank1);

def stats_key_rank(n, nr, dist, n_blocks, diff=(0x0040,0x0)):
  r = np.zeros(n);
  for i in range(n):
    a,b = key_rank_one_round(nr, dist, n_blocks=n_blocks, diff=diff);
    r[i] = randint(a,b);
  return(np.median(r), np.mean(r), r);
