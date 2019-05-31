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
#include<cmath>
#include<stdlib.h>
#include<vector>
#include<fstream>
#include<unordered_set>

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
        inverse_round_function(a,b,ks[i],a,b);
    }
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
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

//this function, given an input difference, calculates all follow-up states and updates a provided unordered_map accordingly
void update_diff_table(uint32_t diff, double p, unordered_map<uint32_t, double>& dtable){
    uint16_t in0 = (uint16_t)(diff >> 16); uint32_t in1 = (uint16_t)(diff & 0xffff);
    in0 = ror(in0, ALPHA); 
    uint16_t in01 = in0 ^ in1;
    for (int out = 0; out < 65536; out++){
        uint16_t x = in01 ^ out; uint16_t y = (in0 ^ out) | (in1 ^ out);
        x = x ^ (x << 1); y = y << 1;
        if ((x & y) == x){
            int weight = __builtin_popcount(y);
            double p_trans = pow(2,-weight);
            uint32_t out0 = out; uint32_t out1 = rol(in1, BETA);
            out1 = out1 ^ out0;
            uint32_t outdiff = (out0 << 16) ^ out1;
            dtable[outdiff] = dtable[outdiff] + p_trans * p;
        };
    }
}

//starting from a given difference, calculate the differential table n rounds in the future;
unordered_map<uint32_t, double> calc_diff_table(uint32_t in, int n){
    unordered_map<uint32_t, double> inps;
    unordered_map<uint32_t, double> outs;
    inps[in] = 1;
    for (int i = 0; i < n; i++){
        for (auto inp = inps.begin(); inp != inps.end(); inp++){
            uint32_t diff = inp->first; double p = inp->second;
            update_diff_table(diff, p, outs);
        }
        inps = outs; outs.clear();
    }
    return(inps);
}

//we will now calculate the ddt for speck three rounds forward from the difference 0x00400000
//and then use that to probe the ddt for five rounds as needed

double calc_p_two_rounds_back(uint32_t diff, auto& ddt){
    uint16_t out0 = (uint16_t)(diff >> 16); uint16_t out1 = (uint16_t)diff;
    uint16_t in1 = out1 ^ out0; in1 = ror(in1, BETA);
    double p = 0;
    for (auto i = ddt.begin(); i != ddt.end(); i++){
        uint32_t indiff = i->first; double p_lower = i->second;
        uint16_t indiff0 = (uint16_t)(indiff >> 16); uint32_t indiff1 = (uint16_t)indiff;
        uint16_t in0 = rol(indiff1, BETA) ^ in1;
        uint32_t diff_middle = ((uint32_t)in0 << 16) ^ in1;
        double p_trans = diff_prob(indiff, diff_middle);
        double p_upper = diff_prob(diff_middle, diff);
        p += p_trans * p_lower * p_upper;
    }
    return(p);
}

void test_pure_differential(uint32_t num_trials){
    ofstream outfile("testdata.csv");
    uint32_t diff = 0x00400000;
    double threshold = pow(2,32)-1; threshold = 1 / threshold;
    auto ddt3 = calc_diff_table(diff, 3);
    uint32_t good_real = 0; uint32_t good_random = 0;
    random_device rd;
    uniform_int_distribution<uint64_t> rng64(0, 0xffffffffffffffffL);
    uniform_int_distribution<uint32_t> rng32(0, 0xffffffff);
    uint32_t nt = num_trials / 10;
    #pragma omp parallel for
    for (int i = 0; i < num_trials; i++){
        uint64_t key = rng64(rd);
        uint32_t plain0 = rng32(rd);
        uint32_t plain1 = plain0 ^ diff;
        uint32_t enc0 = encrypt(plain0, key, 5);
        uint32_t enc1 = encrypt(plain1, key, 5);
        uint32_t outdiff = enc0 ^ enc1;
        double p = calc_p_two_rounds_back(outdiff, ddt3);
        double bayes = p / (threshold + p);
        if (p > threshold) {
            #pragma omp atomic
            good_real++;
        }
        #pragma omp critical
        {
        outfile << hex << enc0 << " " << enc1 << " " << dec << bayes << " " << 1 << endl;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < num_trials; i++){
        uint32_t enc0 = rng32(rd);
        uint32_t enc1 = rng32(rd);
        uint32_t outdiff = enc0 ^ enc1;
        double p = calc_p_two_rounds_back(outdiff, ddt3);
        if (p < threshold) {
            #pragma omp atomic
            good_random++;
        }
        double bayes = p / (threshold + p);
        #pragma omp critical
        outfile << hex << enc0 << " " << enc1 << " " << dec << bayes << " " << 0 << endl;
    }
    outfile.close();
    double acc_real = (double)(good_real)/num_trials;
    double acc_random = (double)(good_random)/num_trials;
    double acc = 0.5 * acc_real + 0.5 * acc_random;
    cout << "Accuracy: " << dec << acc << endl;
    cout << "True positive rate: " << dec << acc_real << endl;
    cout << "True negative rate: " << dec << acc_random << endl;
}

//the following routine takes as input two-round output data, an input difference, and outputs the number of keys
//that transform input with that difference into the observed output

//solve multiple rounds, given an output pair and a differential characteristic
//that is supposed to be followed
//we assume here that the individual transitions have already been checked to be possible
//if more than limit solutions have been found, return

void solve_multiple_rounds(uint32_t c0, uint32_t c1, uint32_t p0, uint32_t p1, vector<uint32_t>& trail, uint64_t limit, uint64_t& counter, int bit_pos, int depth){
    if (counter > limit) return;
    if (bit_pos == WORD_SIZE){
       if (depth == 0) {counter++; return;};
       uint32_t p0a = p0 >> WORD_SIZE; uint32_t p1a = p0 & MASK_VAL;
       uint32_t p0b = p1 >> WORD_SIZE; uint32_t p1b = p1 & MASK_VAL;
       p0a = rol(p0a, ALPHA); p0b = rol(p0b, ALPHA);
       p0 = (p0a << WORD_SIZE) + p1a; p1 = (p0b << WORD_SIZE) + p1b;
       solve_multiple_rounds(p0, p1, ror(p0a ^ p1a, BETA), ror(p0b ^ p1b, BETA), trail, limit,  counter, 0, depth-1);
       return;
    };
    uint32_t outdiff = c0 ^ c1;
    uint32_t o0 = outdiff >> WORD_SIZE; uint32_t o1 = outdiff & MASK_VAL;
    uint32_t i0 = trail[depth] >> WORD_SIZE; uint32_t i1 = trail[depth] & MASK_VAL;
    i0 = ror(i0, ALPHA);
    uint32_t m = (1 << (bit_pos + 1)) - 1; 
    for (int i = 0; i <= 1; i++){
        uint32_t p0a = p0 >> WORD_SIZE; uint32_t p1a = p0 & MASK_VAL;
        uint32_t p0b = p1 >> WORD_SIZE; uint32_t p1b = p1 & MASK_VAL;
        p0a = p0a ^ (i * (1 << bit_pos)); p0b = p0a ^ i0;
        uint32_t t0a = (p0a + p1a) & m; uint32_t t0b = (p0b + p1b) & m;
        uint32_t t = t0a ^ t0b;
        if (t == (o0 & m)) solve_multiple_rounds(c0, c1, (p0a << WORD_SIZE) + p1a, (p0b << WORD_SIZE) + p1b, trail, limit, counter, bit_pos+1, depth);
    }
}

double p_trail(vector<uint32_t>& trail){
    double p = 1;
    for (auto it = trail.begin(); it != trail.end()-1; it++){
        p = p * diff_prob(*it, *(it+1));
    }
    return(p);
}

//complete the middle difference in a two-round trail where input and output difference is given
uint32_t complete_subtrail(uint32_t diff_in, uint32_t diff_out){
    uint32_t i1 = diff_in & MASK_VAL; 
    uint32_t o0 = diff_out >> WORD_SIZE; uint32_t o1 = diff_out & MASK_VAL;
    //uint32_t m1 = ror(o0 ^ o1, BETA); uint32_t m0 = rol(i1, BETA) ^ m1;
    uint32_t m1 = decrypt_one_round(diff_out,0) & MASK_VAL; uint32_t m0 = rol(i1, BETA) ^ m1;
    uint32_t middle = (m0 << WORD_SIZE) + m1;
    return(middle);
}


uint64_t count_sols_2_rounds(uint32_t c0, uint32_t c1, uint32_t in_diff){
    vector<uint32_t> trail(3);
    trail[0] = in_diff; trail[2] = c0 ^ c1;
    trail[1] = complete_subtrail(trail[0], trail[2]);
    uint64_t counter = 0; uint64_t limit = 1L << 32;
    uint32_t p0 = decrypt_one_round(c0, 0) & MASK_VAL; uint32_t p1 = decrypt_one_round(c1, 0) & MASK_VAL;
    solve_multiple_rounds(c0, c1, p0, p1, trail, limit, counter, 0, 1);
    return(counter);
}

void complete_5r_subtrail(vector<uint32_t>& trail){
    trail[1] = complete_subtrail(trail[0], trail[2]);
    trail[3] = complete_subtrail(trail[2], trail[4]);
    /*for (auto it = trail.begin(); it != trail.end(); it++){
        cout << hex << (*it) << " ";
    }
    cout << endl;*/
}

//count number of solutions for four rounds up to a given limit
//needs a DDT of mid-round differences as input
uint64_t count_sols_4r(uint32_t c0, uint32_t c1, uint32_t in_diff, auto& mid_table, uint64_t limit){
    vector<uint32_t> trail(5);
    trail[0] = in_diff; trail[4] = c0 ^ c1;
    uint64_t count = 0;
    for (auto it = mid_table.begin(); it != mid_table.end(); it++){
        trail[2] = it->first;
        complete_5r_subtrail(trail);
        double p = p_trail(trail);
        if (p > 0){
            uint32_t p0 = decrypt_one_round(c0, 0) & MASK_VAL;
            uint32_t p1 = decrypt_one_round(c1, 0) & MASK_VAL;
            solve_multiple_rounds(c0, c1, p0, p1, trail, limit, count, 0, 3);
        };
        if (count > limit) return(count);
    }
    //cout << count << endl;
    return(count);
}

bool classify_4r(uint32_t c0, uint32_t c1, uint32_t in_diff, auto& mid_table){
    vector<uint32_t> trail(5);
    trail[0] = in_diff; trail[4] = c0 ^ c1;
    uint64_t limit = 1L << 32;
    uint64_t count = count_sols_4r(c0, c1, in_diff, mid_table, limit);
    return(count > limit);
}

double estimate_number_solutions(uint32_t c0, uint32_t c1, auto& mid_table){
    double counter = 0; double cx = pow(2,32);
    vector<uint64_t> sols;
    for (auto i = mid_table.begin(); i != mid_table.end(); i++){
        uint32_t diff = i->first; double p = i->second;
        uint32_t n = count_sols_2_rounds(c0, c1, diff);
        counter += n * p * cx;
    }
    return(counter);
}

//estimate number of solutions by key search on the final two rounds 
bool classify_estimate(uint32_t c0, uint32_t c1, uint32_t in_diff, auto& mid_table){
    double num_sols = estimate_number_solutions(c0, c1, mid_table);
    return(num_sols > (1L << 32));
}

//distinguish 5-round Speck near-perfectly, i.e. use ciphertext pair distribution by partial key search
void test_perfect_distinguisher(uint32_t num_trials, uint32_t diff=0x00400000, int rounds=5){
    auto mid_table = calc_diff_table(diff, rounds - 2);
    random_device local_rng; int mx_rounds = 32;
    uniform_int_distribution<uint64_t> rng64(0, 0xffffffffffffffffL);
    uniform_int_distribution<uint64_t> rng32(0, 0xffffffff);
    vector<uint32_t> c0(num_trials); vector<uint32_t> c1(num_trials); vector<bool> labels(num_trials);
    int ex_real = 0;
    for (int i = 0; i < num_trials; i++){
        //bool t = true;
        bool t = rng32(local_rng)&1; labels[i] = t; ex_real = ex_real + t;
        uint32_t p0 = rng32(local_rng);
        uint32_t p1 = p0 ^ diff;
        uint64_t key = rng64(local_rng);
        c0[i] = encrypt(p0, key, rounds); c1[i] = encrypt(p1, key, rounds);
        if (!t){c0[i] = encrypt(c0[i], key, mx_rounds); c1[i] = encrypt(c1[i], key, mx_rounds);};
    }
    int counter = 0; int counter0 = 0; int counter1 = 0;
    int counter_est = 0; int counter0_est = 0; int counter1_est = 0;
    int counter_dis = 0; int full_search_right = 0;
    #pragma omp parallel for
    for (int i = 0; i < num_trials; i++){
        bool c = classify_estimate(c0[i], c1[i], diff, mid_table);
        bool c2 = classify_4r(c0[i], c1[i], 0x80008000, mid_table);
        if (c == labels[i]) counter_est++;
        if ((c == labels[i]) && (c == 0)) counter0_est++;
        if ((c == labels[i]) && (c == 1)) counter1_est++;
        if (c2 == labels[i]) counter++;
        if ((c2 == labels[i]) && (c2 == 0)) counter0++;
        if ((c2 == labels[i]) && (c2 == 1)) counter1++;
        if (c != c2) {counter_dis++; if (c2 == labels[i]) full_search_right++;};
    }
    double acc = (double)counter / num_trials;
    double acc_random = (double)counter0 / (num_trials - ex_real);
    double acc_real = (double)counter1 / ex_real;
    cout << "Full key search:" << endl;
    cout << "Acc: " << acc << ", TPR: " << acc_real << ", TNR: " << acc_random << endl;
    double acc_est = (double)counter_est / num_trials;
    double acc_random_est = (double)counter0_est / (num_trials - ex_real);
    double acc_real_est = (double)counter1_est / ex_real;
    cout << "Partial key search with estimation of lower 2 rounds:" << endl;
    cout << "Acc: " << acc_est << ", TPR: " << acc_real_est << ", TNR: " << acc_random_est << endl;
    cout << "Cases of disagreement between both distinguishers: " << dec << counter_dis << endl;
    cout << "Of these, full search right: " << full_search_right << endl;
}

//estimate a given transition probability empirically
double empirical_p_estimate(uint64_t num_trials, uint32_t diff_in, uint32_t diff_out, int n_rounds){
    uniform_int_distribution<uint64_t> rng64(0,0xffffffffffffffffL);
    uniform_int_distribution<uint32_t> rng32(0,0xffffffff);
    uint64_t counter = 0;
    #pragma omp parallel reduction(+:counter)
    {
        int num_threads = omp_get_num_threads();
        int thread_num = omp_get_thread_num();
        uint64_t nt = num_trials / num_threads;
        if (thread_num == num_threads - 1) nt = nt + num_trials % num_threads;
        random_device rng; mt19937 rd(rng64(rng));
        for (uint64_t i = 0; i < nt; i++){
            uint32_t pt0 = rng32(rd);
            uint32_t pt1 = pt0 ^ diff_in;
            uint64_t key = rng64(rd);
            uint32_t ct0 = encrypt(pt0, key, n_rounds);
            uint32_t ct1 = encrypt(pt1, key, n_rounds);
            uint32_t d = ct0 ^ ct1;
            if (d == diff_out) counter++;
        }
    }
    double res = (double)counter / num_trials;
    res = log2(res);
    return(res);
}

int main(int argc, char* argv[]){
    int n_trials = 100;
    uint64_t n = n_trials * 500;
    uint32_t p = 0x6574694c;
    uint64_t key = 0x0100090811101918L;
    uint32_t c;
    c = encrypt(p, key, 22);
    if (c == 0xa86842f2) cout << "Testvector verified." << endl;
    
    auto ddt3 = calc_diff_table(0x00400000, 3);
    
    cout << "Testing accuracy of Markov model distinguisher against five-round Speck32/64" << endl;
    cout << "Writing whole test set including bayesian predictions to testdata.csv" << endl;
    cout << "Test set size: " << 2*n << endl;
    
    test_pure_differential(n);
    
    cout << endl << "Now testing accuracy of a distinguisher based on DDT and two rounds key search" << endl;
    cout << "Test set size: " << 2*n/10 << endl;
    
    //test_perfect_distinguisher(2*n/10);
    test_perfect_distinguisher(10000);
    
    uint32_t ct0 = 0xc65d2696; uint32_t ct1 = 0xa6a37b2a;
    uint32_t d = ct0 ^ ct1;
    cout << endl << "In the paper, the random ciphertext pair (" << hex << ct0 << ", " << ct1 << ") is claimed to be an impossible output ciphertext pair for five-round Speck with input difference 0x0040/0000. We check this here." << endl;
    double p_markov = calc_p_two_rounds_back(d, ddt3);
    double num_keys = estimate_number_solutions(ct0, ct1, ddt3);
    cout << "Probability of output difference under Markov assumption: " << log2(p_markov) << endl;
    cout << "Estimated number of keys decrypting seen output pair to desired input difference: " << num_keys << endl;
    
    ct0 = 0x58e0bc4; ct1 = 0x85a4ff6c;
    d = ct0 ^ ct1;
    cout << endl << "Now doing the same for the ciphertext pair (" << hex << ct0 << ", " << ct1 << ") mentioned at the end of section 5." << endl;
    
    p_markov = calc_p_two_rounds_back(d, ddt3);
    num_keys = count_sols_4r(ct0, ct1, 0x80008000, ddt3, 1L << 33); 
    double p_test_markov = empirical_p_estimate(10000000, 0x00400000, d, 5);
    
    cout << "Probability of output difference under Markov assumption: " << log2(p_markov) << endl;
    cout << "Empirical probability of the transition (10**7 trials): " << p_test_markov << endl;
    cout << "Probability of decrypting seen output pair to desired input difference (log2, uniformly distributed keys): " << (log2(num_keys) - 64) << endl;
    
    uint64_t trials = 10000000000;
    cout << endl << "According to the paper, the most likely differential transition from input 0x0040/0000 after 8 rounds should be to 0x0280/0080 with log2(p) = -26.015." << endl;
    cout << "Test this prediction in " << dec << trials << " trials." << endl;
    double p_emp = empirical_p_estimate(trials, 0x00400000, 0x02800080,8);
    cout << "Empirical result: " << p_emp << endl;

    return(0);
}

