# c-attention-softmax1-quantization-lab

A very small deterministic C vector-math correctness lab about dot products, scaled attention scores, numerically stable softmax, numerically stable softmax1, attention weighting, causal masking, symmetric int8 block quantization, and multi-head concatenation.

No models. No datasets. No training. No gradients. No GPU.

## Hacker News thread access

Thread: https://news.ycombinator.com/item?id=36851494 — “Attention Is Off By One”

Read via the bundled Hacker News API CLI (`python3 ./hackernews get-item --id 36851494`, Firebase API backend), 2026-07-15. Relevant public comment evidence was captured before preparing the sentiment summary, stored in `hn_comments_sanitized.json` / `hn_thread_evidence.md`.

Summary of the real discussion:

- **tylerneylon** described the proposal as adding one to the internal attention-softmax denominator and questioned whether ordinary attention can already approximate an opt-out through uniform averaging, value projections, and later layers.
- **nazgul17** interpreted the proposal as potentially reducing large activation values so lower-precision representations become more useful.
- **AlanSE** questioned how changing values without changing matrix dimensions would reduce memory.
- **Zamalek** explained the loss of useful resolution when a few large outliers stretch a simple uniform quantization range.
- **Sampo** asked whether an eight-bit floating representation would be more appropriate than a uniform integer representation.
- **TimPC** argued that practical quantization should account for the observed distribution rather than blindly divide the entire range into equal intervals.
- **cycomanic** warned that moving values from a range such as 1 through 5000 into a range such as 0.0002 through 1 may leave the dynamic-range ratio essentially unchanged.
- **numeri** argued that activation-outlier behavior may emerge differently as model scale increases and that success or failure on a tiny model cannot settle every large-model claim.
- **quickthrower2** said a fair experiment might require retuning learning rate, dropout, or other training settings.
- **sweezyjeezy** argued that explicitly providing a useful transformation can matter even when the network is theoretically expressive enough to approximate it.
- **PartiallyTyped** emphasized that practical softmax is evaluated with a maximum-subtraction stability transformation.
- **mrfox321** explained that the additional zero logit still changes the mathematical normalization even after numerical-stability transformations.
- **uoaei** connected the argument to the number of bits needed to distinguish values after normalization.
- **Piezoid** explained that a stable implementation of the added denominator term becomes exp(-max) after shifting by the maximum.
- **ersiees** connected the idea to adding a zero key and value entry in multi-head attention.
- **babel_** pointed to pytorch's add_zero_attn option and a related flaxformer variant while noting that their existence does not prove the new proposal improves modern models.
- **thesausageking** said related mechanisms were already known and available but were not widely used because prior experiments had not made them a default choice.
- **civilized** questioned whether an existing option being disabled by default indicates limited practical benefit.

The thread contains disagreement rather than a settled empirical conclusion. A local C arithmetic lab cannot answer the model-training question. Attention and quantization are directly neural-network-relevant, while scalar correctness does not validate a neural network.

## What this repo covers

Article (evanmiller.org/attention-is-off-by-one): attention outliers, whitespace/punctuation positions, ordinary softmax, forced choice among value vectors, proposed softmax1 denominator, implicit zero-valued option, relative-weight preservation, total attention mass below one, negative-logit limit, quietattention name, quantization motivation, need to retrain.

HN commenters: as summarized above.

Transformer paper (Vaswani et al. 2017): scaled dot-product attention.

Quantizable Transformers (Qualcomm, 2023): reports activation outliers and attention heads.

PyTorch MultiheadAttention: documents `add_zero_attn` – existence is not proof of benefit.

C11 / POSIX: exp(), sqrt(), fabs(), round(), isfinite(), isnan(), isinf(), signbit().

Zig CC: portable C compilation.

Local observations: exp(0)=1, sqrt(4)=2, etc., from the linked libm.

Python recomputation: independent double-precision checks.

Quantizer: repository symmetric int8 block, scale = max_abs/127, round-away-from-zero, inspired by common scale×int quantization, NOT q8_0 / llama.cpp bit-exact.

Not attempted: model weights, training, gradients, GPU, framework integration, model quality claims.

Policies: stable softmax with max-subtraction, softmax1 with implicit zero exp(-m), explicit boolean masks, all-masked → zero output + error status, non-finite rejection before exp.

Relevance: query/key/value vectors, attention scores, causal masks, activation quantization, kv-cache representations, multi-head concatenation, deterministic vector math – all directly relevant to NN inference.

Not validated: transformer architecture, pretrained model, quantization format, gguf, llama.cpp, tokenizer, dataset, training, gradients, loss, perplexity, benchmark, GPU backend, production inference.

## Disclaimers

The repository does NOT prove:

- ordinary transformer attention contains a literal implementation bug;
- softmax1 improves a trained model;
- softmax1 reduces loss;
- softmax1 reduces perplexity;
- softmax1 eliminates activation outliers;
- softmax1 improves quantization;
- softmax1 reduces model size;
- softmax1 reduces kv-cache size;
- softmax1 reduces RAM use;
- softmax1 improves inference speed;
- softmax1 is equivalent to every add-zero-attention implementation;
- softmax1 can be inserted into an existing checkpoint without retraining;
- the implicit zero mass has learned semantic meaning;
- a lower attention-output norm is better;
- ordinary softmax cannot approximate an opt-out through other learned components;
- uniform attention always produces a near-zero value vector;
- the article's whitespace and punctuation observation applies to every model or tokenizer;
- outlier behavior begins at one universal model size;
- small synthetic vectors predict large-model behavior;
- eight-bit integer quantization is better than eight-bit floating point;
- uniform quantization is appropriate for every distribution;
- the repository block format is llama.cpp q8_0;
- the repository is bit-exact with ggml;
- the repository validates a gguf model;
- the repository validates llama.cpp;
- the repository validates pytorch;
- the repository implements flash attention;
- the repository implements causal transformer decoding;
- the repository implements a kv cache;
- the repository implements rotary embeddings;
- the repository implements normalization layers;
- the repository implements gradient calculation;
- the repository trains a model;
- the repository reads a dataset;
- the repository calculates model quality;
- the repository establishes an acceptable quantization threshold;
- the repository establishes numerical reproducibility across all compilers, libm implementations, processors, or floating-point modes;
- the selected local libc or libm is one specific implementation unless independently identified;
- passing scalar tests validates neural-network semantics;
- passing vector tests validates a production inference pipeline;
- the lab is statistically validated;
- the lab is machine-learning validated;
- the lab is secure;
- the lab is production-ready.

## Build

```
$ZIG_BIN cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror attention_lab.c -lm -o attention_lab
python3 run_lab.py
python3 -m unittest -v
```

See RESULTS.md (generated).
