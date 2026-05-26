# Addition Transformer

Karpathy GPT homework notebook for learning decoder-only Transformers by building a small character-level model that learns addition.

The main notebook is [notebooks/gpt-hw.ipynb](notebooks/gpt-hw.ipynb).

## What This Covers

- **EX1: Parallel multi-head attention**
  - Starts from the lecture-style separate `Head` and `MultiHeadAttention` classes.
  - Merges the attention computation so all heads are computed together with tensor operations.
  - Practices the core tensor shape flow: `(B, T, C) -> (B, n_head, T, head_size)`.

- **EX2: Addition as language modeling**
  - Generates synthetic addition examples up to 3-digit numbers.
  - Hard-codes a tiny character vocabulary: digits, `+`, `=`, and space.
  - Trains a decoder-only Transformer to predict reversed answer digits, e.g. `123+456=9750 ` represents answer `0579`.

## Model Sketch

The model is a compact GPT-style decoder:

```text
token ids
-> token embeddings + learned position embeddings
-> repeated Transformer blocks
   -> pre-norm causal multi-head self-attention
   -> residual connection
   -> pre-norm feed-forward network
   -> residual connection
-> final layer norm
-> vocab logits
```

Current notebook configuration:

```text
vocab_size = 13
n_embd = 64
n_head = 4
n_layer = 4
parameters ~= 0.2M
```

## Result

The saved notebook run reaches roughly:

```text
train loss ~= 0.0003
val loss   ~= 0.0003
```

The inference cells then prompt the model with examples like:

```text
123+456=
```

and autoregressively generate the answer digits.

## Notes

This is intentionally an exploratory learning notebook, not a polished training library. The goal is to make the Transformer mechanics visible: causal masking, attention heads, tensor reshaping, residual connections, loss masking, and autoregressive generation.
