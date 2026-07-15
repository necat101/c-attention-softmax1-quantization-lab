#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <errno.h>
#include <stdio.h>

static double dot_product(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static int stable_softmax(const double *x, const int *mask, size_t n, double *out) {
    int allowed = 0;
    double maxv = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        if (!isfinite(x[i])) return -2;
        allowed++;
        if (x[i] > maxv) maxv = x[i];
    }
    if (allowed == 0) return -1;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) { out[i] = 0.0; continue; }
        double e = exp(x[i] - maxv);
        out[i] = e;
        sum += e;
    }
    if (!isfinite(sum) || sum <= 0) return -3;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        out[i] /= sum;
    }
    return 0;
}

static int stable_softmax1(const double *x, const int *mask, size_t n, double *out, double *abstention) {
    int allowed = 0;
    double maxv = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        if (!isfinite(x[i])) return -2;
        allowed++;
        if (allowed == 1) maxv = x[i] > 0.0 ? x[i] : 0.0;
        else if (x[i] > maxv) maxv = x[i];
    }
    if (allowed == 0) { maxv = 0.0; }
    double implicit_zero = exp(-maxv);
    double sum = implicit_zero;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) { out[i] = 0.0; continue; }
        double e = exp(x[i] - maxv);
        out[i] = e;
        sum += e;
    }
    if (!isfinite(sum) || sum <= 0) return -3;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        out[i] /= sum;
    }
    *abstention = implicit_zero / sum;
    return 0;
}

static void weighted_sum(const double *weights, const double *values, size_t n, size_t dim, double *out) {
    for (size_t d = 0; d < dim; d++) out[d] = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < dim; d++) {
            out[d] += weights[i] * values[i*dim + d];
        }
    }
}

typedef struct {
    float max_abs;
    float scale;
    int zero_block;
} quant_info_t;

static quant_info_t quantize_int8_block(const float *in, int8_t *out, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    quant_info_t info = {max_abs, 0.0f, 0};
    if (max_abs == 0.0f) {
        info.zero_block = 1;
        for (size_t i = 0; i < n; i++) out[i] = 0;
        return info;
    }
    info.scale = max_abs / 127.0f;
    for (size_t i = 0; i < n; i++) {
        float v = roundf(in[i] / info.scale);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        out[i] = (int8_t)v;
    }
    return info;
}

static void dequantize_int8_block(const int8_t *in, float scale, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = scale * (float)in[i];
}

int main(void) {
    printf("{\n");
    printf("  \"toolchain\": {\n");
    printf("    \"STDC_VERSION\": %ld,\n", (long)__STDC_VERSION__);
    printf("    \"CHAR_BIT\": %d,\n", CHAR_BIT);
    printf("    \"sizeof_char\": %zu,\n", sizeof(char));
    printf("    \"sizeof_int8_t\": %zu,\n", sizeof(int8_t));
    printf("    \"sizeof_int\": %zu,\n", sizeof(int));
    printf("    \"sizeof_float\": %zu,\n", sizeof(float));
    printf("    \"sizeof_double\": %zu,\n", sizeof(double));
    printf("    \"sizeof_void_p\": %zu,\n", sizeof(void*));
    printf("    \"sizeof_size_t\": %zu,\n", sizeof(size_t));
    printf("    \"FLT_RADIX\": %d,\n", FLT_RADIX);
    printf("    \"FLT_MANT_DIG\": %d,\n", FLT_MANT_DIG);
    printf("    \"DBL_MANT_DIG\": %d,\n", DBL_MANT_DIG);
    printf("    \"INT8_MIN\": %d,\n", INT8_MIN);
    printf("    \"INT8_MAX\": %d\n", INT8_MAX);
    printf("  },\n");

    /* math api check */
    double (*exp_p)(double) = exp;
    double (*sqrt_p)(double) = sqrt;
    double (*fabs_p)(double) = fabs;
    double (*round_p)(double) = round;
    double e0 = exp_p(0.0);
    double s4 = sqrt_p(4.0);
    double f = fabs_p(-3.5);
    double r1 = round_p(2.5);
    double r2 = round_p(-2.5);
    int isfin = isfinite(1.0);
    int isn = isnan(nan(""));
    int isinfv = isinf(INFINITY);
    int sb = signbit(-0.0);
    printf("  \"math_api\": {\n");
    printf("    \"exp_0\": %.17g,\n", e0);
    printf("    \"sqrt_4\": %.17g,\n", s4);
    printf("    \"fabs_m3p5\": %.17g,\n", f);
    printf("    \"round_2p5\": %.17g,\n", r1);
    printf("    \"round_m2p5\": %.17g,\n", r2);
    printf("    \"isfinite_1\": %d,\n", isfin);
    printf("    \"isnan_nan\": %d,\n", isn);
    printf("    \"isinf_inf\": %d,\n", isinfv);
    printf("    \"signbit_negzero\": %d\n", sb);
    printf("  },\n");

    /* dot product */
    double a_dp[4] = {1.0, -2.0, 3.0, 0.5};
    double b_dp[4] = {4.0, 0.25, -1.0, 2.0};
    double dp_products[4];
    for (int i=0;i<4;i++) dp_products[i]=a_dp[i]*b_dp[i];
    double dp_sum = dot_product(a_dp, b_dp, 4);
    printf("  \"dot_product\": {\"a\":[1.0,-2.0,3.0,0.5],\"b\":[4.0,0.25,-1.0,2.0],\"products\":[%.17g,%.17g,%.17g,%.17g],\"sum\":%.17g},\n", dp_products[0],dp_products[1],dp_products[2],dp_products[3],dp_sum);

    /* scaled attention scores */
    double query1[2] = {1.0, 0.0};
    double keys1[3][2] = {{1.0,0.0},{0.0,1.0},{-1.0,0.0}};
    double dots1[3];
    for (int i=0;i<3;i++) dots1[i]=dot_product(query1, keys1[i], 2);
    double scale = 1.0 / sqrt(2.0);
    double scores1[3];
    for (int i=0;i<3;i++) scores1[i]=dots1[i]*scale;
    printf("  \"scaled_scores\": {\"dots\":[%.17g,%.17g,%.17g],\"scale\":%.17g,\"scores\":[%.17g,%.17g,%.17g]},\n", dots1[0],dots1[1],dots1[2],scale,scores1[0],scores1[1],scores1[2]);

    /* stable softmax */
    double weights_sm[3];
    stable_softmax(scores1, NULL, 3, weights_sm);
    double sum_sm = weights_sm[0]+weights_sm[1]+weights_sm[2];
    printf("  \"softmax\": {\"weights\":[%.17g,%.17g,%.17g],\"sum\":%.17g},\n", weights_sm[0],weights_sm[1],weights_sm[2],sum_sm);

    /* shift invariance */
    double scores_shifted[3] = {scores1[0]+100.0, scores1[1]+100.0, scores1[2]+100.0};
    double w_shift[3];
    stable_softmax(scores_shifted, NULL, 3, w_shift);
    printf("  \"softmax_shift\": {\"base_weights\":[%.17g,%.17g,%.17g],\"shifted_weights\":[%.17g,%.17g,%.17g]},\n", weights_sm[0],weights_sm[1],weights_sm[2],w_shift[0],w_shift[1],w_shift[2]);

    /* softmax1 */
    double weights_s1[3]; double abstention;
    stable_softmax1(scores1, NULL, 3, weights_s1, &abstention);
    double sum_s1 = weights_s1[0]+weights_s1[1]+weights_s1[2];
    printf("  \"softmax1\": {\"weights\":[%.17g,%.17g,%.17g],\"sum\":%.17g,\"abstention\":%.17g},\n", weights_s1[0],weights_s1[1],weights_s1[2],sum_s1,abstention);

    /* all negative */
    double neg_scores[3] = {-20.0,-20.0,-20.0};
    double w_neg[3], w_neg1[3], abst_neg;
    stable_softmax(neg_scores, NULL, 3, w_neg);
    stable_softmax1(neg_scores, NULL, 3, w_neg1, &abst_neg);
    printf("  \"all_negative\": {\"softmax\":[%.17g,%.17g,%.17g],\"softmax1\":[%.17g,%.17g,%.17g],\"abstention\":%.17g},\n", w_neg[0],w_neg[1],w_neg[2],w_neg1[0],w_neg1[1],w_neg1[2],abst_neg);

    /* attention weighted sum */
    double values_aw[3][2] = {{1.0,2.0},{3.0,4.0},{5.0,6.0}};
    double out_aw[2];
    weighted_sum(weights_sm, (double*)values_aw, 3, 2, out_aw);
    printf("  \"attention_output\": [%.17g,%.17g],\n", out_aw[0], out_aw[1]);

    /* quiet attention */
    double out_qa[2];
    weighted_sum(weights_s1, (double*)values_aw, 3, 2, out_qa);
    printf("  \"quiet_attention_output\": [%.17g,%.17g],\n", out_qa[0], out_qa[1]);

    /* causal mask */
    double causal_scores[3][3] = {{1.0,2.0,3.0},{0.0,1.0,2.0},{-1.0,0.0,1.0}};
    double causal_values[3][2] = {{1.0,0.0},{0.0,1.0},{1.0,1.0}};
    double causal_out[3][2];
    double causal_w[3][3] = {{0}};
    for (int row=0; row<3; row++) {
        int mask[3] = {0,0,0};
        for (int c=0;c<=row;c++) mask[c]=1;
        double w[3]={0};
        stable_softmax(causal_scores[row], mask, 3, w);
        for (int c=0;c<3;c++) causal_w[row][c]=mask[c]?w[c]:0.0;
        weighted_sum(w, (double*)causal_values, 3, 2, causal_out[row]);
    }
    printf("  \"causal\": {\"outputs\":[[%.17g,%.17g],[%.17g,%.17g],[%.17g,%.17g]]},\n", causal_out[0][0],causal_out[0][1],causal_out[1][0],causal_out[1][1],causal_out[2][0],causal_out[2][1]);

    /* all masked */
    int mask_none[3] = {0,0,0};
    double w_allmask[3] = {9,9,9};
    int status_allmask = stable_softmax(scores1, mask_none, 3, w_allmask);
    printf("  \"all_masked\": {\"status\":%d},\n", status_allmask);

    /* nonfinite */
    double nf1[3] = {0.0, NAN, 1.0};
    double nf2[3] = {0.0, INFINITY, 1.0};
    double nf3[3] = {0.0, -INFINITY, 1.0};
    double w_nf[3];
    int st_nf1 = stable_softmax(nf1, NULL, 3, w_nf);
    int st_nf2 = stable_softmax(nf2, NULL, 3, w_nf);
    int st_nf3 = stable_softmax(nf3, NULL, 3, w_nf);
    printf("  \"nonfinite\": {\"nan_status\":%d,\"inf_status\":%d,\"ninf_status\":%d},\n", st_nf1, st_nf2, st_nf3);

    /* symmetric int8 block */
    float block_vals[32] = {-1.000f,-0.750f,-0.500f,-0.333f,-0.250f,-0.125f,-0.100f,-0.030f,0.000f,0.030f,0.100f,0.125f,0.250f,0.333f,0.500f,0.750f,1.000f,0.820f,-0.820f,0.670f,-0.670f,0.420f,-0.420f,0.210f,-0.210f,0.070f,-0.070f,0.015f,-0.015f,0.950f,-0.950f,0.580f};
    int8_t q_block[32];
    quant_info_t qi = quantize_int8_block(block_vals, q_block, 32);
    float deq_block[32];
    dequantize_int8_block(q_block, qi.scale, deq_block, 32);
    printf("  \"int8_block\": {\"max_abs\":%.9g,\"scale\":%.9g,\"q\":[", qi.max_abs, qi.scale);
    for(int i=0;i<32;i++) printf("%d%s", q_block[i], i<31?",":"");
    printf("],\"deq\":[");
    for(int i=0;i<32;i++) printf("%.9g%s", deq_block[i], i<31?",":"");
    printf("]},\n");

    /* zero block */
    float zero_block[32] = {0};
    int8_t q_zero[32];
    quant_info_t qz = quantize_int8_block(zero_block, q_zero, 32);
    printf("  \"zero_block\": {\"max_abs\":%.9g,\"scale\":%.9g,\"zero_block\":%d},\n", qz.max_abs, qz.scale, qz.zero_block);

    /* dequantized dot product */
    float v1[16] = {0.35f,-0.80f,0.10f,0.55f,-0.25f,0.90f,-0.45f,0.05f,0.72f,-0.18f,0.33f,-0.61f,0.47f,0.12f,-0.29f,0.84f};
    float v2[16] = {-0.40f,0.65f,0.20f,-0.75f,0.50f,0.15f,-0.30f,0.95f,-0.62f,0.44f,0.08f,-0.57f,0.31f,-0.22f,0.76f,-0.11f};
    float vb1[32]={0}, vb2[32]={0};
    for(int i=0;i<16;i++){ vb1[i]=v1[i]; vb2[i]=v2[i];}
    int8_t qb1[32], qb2[32];
    quant_info_t q1 = quantize_int8_block(vb1, qb1, 32);
    quant_info_t q2 = quantize_int8_block(vb2, qb2, 32);
    float dq1[32], dq2[32];
    dequantize_int8_block(qb1, q1.scale, dq1, 32);
    dequantize_int8_block(qb2, q2.scale, dq2, 32);
    double ref_dot=0, deq_dot=0;
    for(int i=0;i<16;i++){ ref_dot += (double)v1[i]*v2[i]; deq_dot += (double)dq1[i]*dq2[i]; }
    printf("  \"deq_dot\": {\"ref\":%.17g,\"deq\":%.17g,\"scale1\":%.9g,\"scale2\":%.9g},\n", ref_dot, deq_dot, q1.scale, q2.scale);

    /* quantized attention error */
    double qa_q[2] = {0.35, -0.80};
    double qa_k[3][2] = {{0.90,-0.10},{-0.25,0.70},{0.45,0.55}};
    double qa_v[3][2] = {{0.20,-0.40},{1.10,0.30},{-0.60,0.90}};
    double qa_scores[3];
    for(int i=0;i<3;i++) qa_scores[i]=dot_product(qa_q, qa_k[i], 2) / sqrt(2.0);
    double qa_w[3];
    stable_softmax(qa_scores, NULL, 3, qa_w);
    double qa_out[2];
    weighted_sum(qa_w, (double*)qa_v, 3, 2, qa_out);
    /* pack block: q,k,v then zeros */
    float qa_block[32]={0};
    qa_block[0]=(float)qa_q[0]; qa_block[1]=(float)qa_q[1];
    qa_block[2]=(float)qa_k[0][0]; qa_block[3]=(float)qa_k[0][1];
    qa_block[4]=(float)qa_k[1][0]; qa_block[5]=(float)qa_k[1][1];
    qa_block[6]=(float)qa_k[2][0]; qa_block[7]=(float)qa_k[2][1];
    qa_block[8]=(float)qa_v[0][0]; qa_block[9]=(float)qa_v[0][1];
    qa_block[10]=(float)qa_v[1][0]; qa_block[11]=(float)qa_v[1][1];
    qa_block[12]=(float)qa_v[2][0]; qa_block[13]=(float)qa_v[2][1];
    int8_t qa_qb[32];
    quant_info_t qaqi = quantize_int8_block(qa_block, qa_qb, 32);
    float qa_deq[32];
    dequantize_int8_block(qa_qb, qaqi.scale, qa_deq, 32);
    double qa_qd[2] = {qa_deq[0], qa_deq[1]};
    double qa_kd[3][2] = {{qa_deq[2],qa_deq[3]},{qa_deq[4],qa_deq[5]},{qa_deq[6],qa_deq[7]}};
    double qa_vd[3][2] = {{qa_deq[8],qa_deq[9]},{qa_deq[10],qa_deq[11]},{qa_deq[12],qa_deq[13]}};
    double qa_scores_d[3];
    for(int i=0;i<3;i++) qa_scores_d[i]=dot_product(qa_qd, qa_kd[i], 2) / sqrt(2.0);
    double qa_wd[3];
    stable_softmax(qa_scores_d, NULL, 3, qa_wd);
    double qa_out_d[2];
    weighted_sum(qa_wd, (double*)qa_vd, 3, 2, qa_out_d);
    printf("  \"quant_attn\": {\"ref_out\":[%.17g,%.17g],\"deq_out\":[%.17g,%.17g],\"scale\":%.9g},\n", qa_out[0], qa_out[1], qa_out_d[0], qa_out_d[1], qaqi.scale);

    /* multihead */
    double mh0_q[2] = {1.0,0.0};
    double mh0_k[3][2] = {{1.0,0.0},{0.0,1.0},{-1.0,0.0}};
    double mh0_v[3][2] = {{1.0,2.0},{3.0,4.0},{5.0,6.0}};
    double mh0_scores[3]; for(int i=0;i<3;i++) mh0_scores[i]=dot_product(mh0_q, mh0_k[i],2)/sqrt(2.0);
    double mh0_w[3]; stable_softmax(mh0_scores, NULL, 3, mh0_w);
    double mh0_out[2]; weighted_sum(mh0_w, (double*)mh0_v,3,2,mh0_out);
    double mh1_q[2] = {0.0,1.0};
    double mh1_k[3][2] = {{1.0,0.0},{0.0,1.0},{1.0,1.0}};
    double mh1_v[3][2] = {{-1.0,1.0},{2.0,-2.0},{0.5,0.25}};
    double mh1_scores[3]; for(int i=0;i<3;i++) mh1_scores[i]=dot_product(mh1_q, mh1_k[i],2)/sqrt(2.0);
    double mh1_w[3]; stable_softmax(mh1_scores, NULL, 3, mh1_w);
    double mh1_out[2]; weighted_sum(mh1_w, (double*)mh1_v,3,2,mh1_out);
    printf("  \"multihead\": {\"head0\":[%.17g,%.17g],\"head1\":[%.17g,%.17g],\"concat\":[%.17g,%.17g,%.17g,%.17g]}\n", mh0_out[0],mh0_out[1],mh1_out[0],mh1_out[1],mh0_out[0],mh0_out[1],mh1_out[0],mh1_out[1]);
    printf("}\n");
    return 0;
}
