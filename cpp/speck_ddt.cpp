#include<iostream>
#include<cstdlib>
#include<random>
#include<vector>
#include<cstring>
#include<tuple>
#include<unordered_map>
#include<map>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include<vector>
#include<fstream>
#include<unordered_set>
#include<algorithm>
#include<string>
#include<sstream>

using namespace std;

#define WORD_SIZE 16
#define ALPHA 7
#define BETA 2
#define MASK_VAL 0xffff
#define MAX_ROUNDS 50

uint32_t rol(uint32_t a, uint32_t b){
    uint32_t n = ((a << b) & MASK_VAL) | (a >> (WORD_SIZE - b));
    return(n);
}

uint32_t ror(uint32_t a, uint32_t b){
    uint32_t n = (a >> b) | (MASK_VAL & (a << (WORD_SIZE - b)));
    return(n);
}

void round_function(uint32_t a, uint32_t b, uint32_t k, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    c0 = ror(c0, ALPHA);
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA);
    c1 = c1 ^ c0;
    x = c0; y = c1;
}

void inverse_round_function(uint32_t a, uint32_t b, uint32_t k, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA);
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA);
    x = c0; y = c1;
}

uint32_t decrypt_one_round(uint32_t c, uint32_t sk){
    uint32_t x,y;
    uint32_t c0 = c >> 16; uint32_t c1 = c & MASK_VAL;
    inverse_round_function(c0, c1, sk, x, y);
    uint32_t res = (x << 16) ^ y;
    return(res);
}

uint32_t encrypt_one_round(uint32_t p, uint32_t sk){
    uint32_t x,y;
    uint32_t p0 = p >> 16; uint32_t p1 = p & MASK_VAL;
    round_function(p0,p1,sk,x,y);
    uint32_t res = (x << 16) ^ y;
    return(res);
}

uint32_t encrypt(uint32_t p, uint64_t key, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    uint16_t k[4];
    memcpy(k, &key, 8);
    uint32_t sk = k[3]; uint32_t tmp;
    for (uint32_t i = 0; i < rounds; i++){
        round_function(a,b,sk,a,b);
        round_function(k[2 - i%3], sk, i, tmp, sk);
        k[2-i%3] = tmp;
    }
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
}

uint32_t decrypt(uint32_t p, uint64_t key, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    uint16_t k[4];
    memcpy(k, &key, 8);
    uint32_t sk = k[3]; uint32_t tmp;
    uint32_t ks[MAX_ROUNDS];
    for (uint32_t i = 0; i < rounds; i++){
        ks[i] = sk;
        round_function(k[2 - i%3], sk, i, tmp, sk);
        k[2-i%3] = tmp;
    }
    for (int i = rounds-1; i >= 0; i--){
        //cout << dec << i << endl;
        inverse_round_function(a,b,ks[i],a,b);
    }
    //cout << a << " " << b << endl;
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
}

void make_examples(uint32_t nr, uint32_t diff, vector<uint32_t>& v0, vector<uint32_t>& v1, vector<uint32_t>& w){
    random_device rd;
    uniform_int_distribution<uint32_t> rng32(0, 0xffffffff);
    uniform_int_distribution<uint64_t> rng64(0, 0xffffffffffffffffL);
    mt19937 rng(rng64(rd));
    for (int i = 0; i < w.size(); i++)
        w[i] = (rng32(rd)) & 1;
    for (int i = 0; i < v0.size(); i++){
        if (w[i]) {
            uint64_t key = rng64(rd);
            uint32_t plain0 = rng32(rd);
            uint32_t plain1 = plain0 ^ diff;
            uint32_t c0 = encrypt(plain0, key, nr);
            uint32_t c1 = encrypt(plain1, key, nr);
            v0[i] = c0; v1[i] = c1;
        } else {
            v0[i] = rng32(rd); v1[i] = rng32(rd);
            while (v0[i] == v1[i])
                v0[i] = rng32(rd);
        }
    }
}

//the following function calculates the probability of a xor-differential transition of one round of Speck32 according to Lipmaa-Moriai
double diff_prob(uint32_t in, uint32_t out){
    //first, transform the output difference to what it looked like before the modular addition
    //transform also the input difference accordingly
    uint32_t in0 = in >> 16; uint32_t in1 = in & 0xffff;
    uint32_t out0 = out >> 16; uint32_t out1 = out & 0xffff;
    in0 = ror(in0, ALPHA); out1 = out1 ^ out0;
    out1 = ror(out1, BETA);
    if (out1 != in1) return(0);
    uint32_t x = in0 ^ in1 ^ out0; uint32_t y = (in0 ^ out0) | (in1 ^ out0);
    x = (x ^ (x << 1)) & MASK_VAL; y = (y << 1) & MASK_VAL;
    if ((x & y) != x) return(0);
    int weight = __builtin_popcount(y);
    double res = pow(2,-weight);
    return(res);
}

void calc_ddt_update(vector<double>& ddt, vector<double>& tmp){
  uint64_t small = 1L << 32;
  vector<double> sums(1L << WORD_SIZE);
  for (uint32_t i = 1; i != 0; i++)
    sums[i >> WORD_SIZE] += ddt[i];
  #pragma omp parallel for
  for (uint64_t i = 1; i < small; i++){
    uint32_t out = i;
    uint32_t in1 = ror((out >> WORD_SIZE) ^ (out & MASK_VAL), BETA);
    double p = 0; uint32_t ind = in1 << WORD_SIZE;
    if (sums[in1] != 0)
      for (uint32_t in2 = 0; in2 <= MASK_VAL; in2++){
        uint32_t index = ind ^ in2; uint32_t inp = (in2 << WORD_SIZE) ^ in1;
        p += ddt[index] * diff_prob(inp, out);
      }
    uint32_t ind_out = (out >> WORD_SIZE) ^ ((out & MASK_VAL) << WORD_SIZE);
    tmp[ind_out] = p;
  }
  #pragma omp parallel for
  for (uint64_t out = 1; out < small; out++)
    ddt[out] = tmp[out];
}

void calc_ddt(uint32_t in_diff, int num_rounds){
  uint64_t num_diffs = 1L << 32;
  vector<double> ddt(num_diffs); vector<double> tmp(num_diffs);
  uint32_t ind = (in_diff >> WORD_SIZE) ^ ((in_diff & MASK_VAL) << WORD_SIZE);
  ddt[ind] = 1.0; double r = 1.0 / (1L << 32);
  for (int i = 0; i < num_rounds; i++){
    calc_ddt_update(ddt, tmp);
    double tpr = 0.0; double tnr = 0.0;
    for (uint32_t j = 1; j != 0; j++){
      if (ddt[j] > r) tpr += ddt[j];
      if (ddt[j] < r) tnr += r;
    }
    double acc = (tpr + tnr)/2;
    cout << "Rounds: " << dec << (i+1) <<", Acc: " << acc << ", tpr:" << tpr << ", tnr:" << tnr << endl;
    stringstream del; del << hex << in_diff;
    string delta = del.str();
    string rounds = to_string(i); string filename = "ddt_"+ delta +"_" + rounds + "rounds.bin";
    ofstream fout(filename, ios::out | ios::binary);
    fout.write((char*)&ddt[0], ddt.size() * sizeof(double));
    fout.close();
  }
}

int main(){
    cout << "Calculate full ddt explicitly" << endl;
    calc_ddt(0x00400000, 8);
    cout << "Done." << endl;
    return(0);
}

