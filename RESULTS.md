# Results

zig: /portable-zig/.local/zig/zig 0.14.0

target: x86_64-unknown-linux-musl

python: 3.12.3 Linux-6.17.0-1009-aws-x86_64-with-glibc2.39

cases: 20, methods: 5, rows: 100

Classifications:
- pass: 13
- expected_error: 2
- local_observation: 4
- implementation_skip: 0
- toolchain_skip: 0
- context_only: 1
- not_applicable: 80
- fail: 0

dot_product_sum: 1.5

scaled_scores: [0.7071067811865475, 0, -0.7071067811865475]

softmax_weights: [0.575975345215362, 0.28399540974126003, 0.14002924504337802] sum=1

softmax1_weights: [0.44858053295644384, 0.22118101637021303, 0.10905743430313006] abstention=0.22118101637021303

attention_output: [2.128107799656032, 3.128107799656032]
quiet_attention_output: [1.6574107535827332, 2.4362297372125203]

causal_outputs: [[1, 0], [0.2689414213699951, 0.7310585786300049], [0.7552715289452022, 0.9099694268296195]]

all_masked_status: -1
nonfinite_statuses: nan=-2 inf=-2 ninf=-2

int8_scale: 0.00787401572

zero_block: max_abs=0 scale=0

deq_dot ref=-1.2049999986961488 deq=-1.2111749136536132

quant_attn ref=[0.16907972867125676, 0.1434658560683803] deq=[0.1701235243346652, 0.1447201803082028]

multihead_concat: [2.128107799656032, 3.128107799656032, 0.8050044170590367, -0.5041703475491972]

elapsed: 0.345s

## Disclaimers

The repository does NOT prove:

- ordinary transformer attention contains a literal implementation bug
- softmax1 improves a trained model
- softmax1 reduces loss
- softmax1 reduces perplexity
- softmax1 eliminates activation outliers
- softmax1 improves quantization
- softmax1 reduces model size
- softmax1 reduces kv-cache size
- softmax1 reduces RAM use
- softmax1 improves inference speed
- softmax1 is equivalent to every add-zero-attention implementation
- softmax1 can be inserted into an existing checkpoint without retraining
- the implicit zero mass has learned semantic meaning
- a lower attention-output norm is better
- ordinary softmax cannot approximate an opt-out through other learned components
- uniform attention always produces a near-zero value vector
- the article's whitespace and punctuation observation applies to every model or tokenizer
- outlier behavior begins at one universal model size
- small synthetic vectors predict large-model behavior
- eight-bit integer quantization is better than eight-bit floating point
- uniform quantization is appropriate for every distribution
- the repository block format is llama.cpp q8_0
- the repository is bit-exact with ggml
- the repository validates a gguf model
- the repository validates llama.cpp
- the repository validates pytorch
- the repository implements flash attention
- the repository implements causal transformer decoding
- the repository implements a kv cache
- the repository implements rotary embeddings
- the repository implements normalization layers
- the repository implements gradient calculation
- the repository trains a model
- the repository reads a dataset
- the repository calculates model quality
- the repository establishes an acceptable quantization threshold
- the repository establishes numerical reproducibility across all compilers, libm implementations, processors, or floating-point modes
- the selected local libc or libm is one specific implementation unless independently identified
- passing scalar tests validates neural-network semantics
- passing vector tests validates a production inference pipeline
- the lab is statistically validated
- the lab is machine-learning validated
- the lab is secure
- the lab is production-ready
