# Model Basics Notes

> Discussion notes on the fundamental concepts of a language model.

---

## Vocabulary

A fixed set of all tokens the model knows about. Each token has an integer ID.

```
ID 0 = "Hello"
ID 1 = "world"
ID 2 = "!"
ID 3 = "goodbye"
ID 4 = "<end>"
```

Nanochat's vocabulary has 32,768 tokens. The vocabulary is fixed after tokenizer training — the model never sees a token outside this set.

---

## Input tokens

Text is converted to a sequence of token IDs before being fed to the model:

```
"Hello world" → [0, 1]
```

This sequence is the only thing the model receives. It never sees raw text — only integers.

---

## Logits

The table the model produces after processing the input. It has:

- **One row per input token** (position in the sequence)
- **One column per vocabulary token** (all possible next tokens)

Each row is the model's answer to: *"given everything I've seen up to this position, how likely is each vocabulary token to come next?"*

For input `["Hello", "world"]` with a 5-token vocabulary:

```
                col 0    col 1   col 2    col 3     col 4
              "Hello"  "world"    "!"   "goodbye"  "<end>"
              ───────  ───────  ──────  ─────────  ───────
position 0:  [ -1.2,    3.4,    0.8,    -2.1,      0.1  ]  ← after seeing "Hello"
position 1:  [ -0.5,   -1.1,    4.7,    -3.2,      2.1  ]  ← after seeing "Hello world"
```

- **Row (position)**: which prefix the model has seen so far. Position 0 has only seen `"Hello"`. Position 1 has seen `"Hello world"`.
- **Column**: which vocabulary token the score is for. Col 0 = score for `"Hello"` being next, col 1 = score for `"world"` being next, col 2 = score for `"!"` being next, and so on.

So reading position 0 across its columns: after seeing `"Hello"`, the model gives `"world"` (col 1) the highest score of 3.4 — meaning it thinks `"world"` is the most likely next token.

Reading position 1 across its columns: after seeing `"Hello world"`, the model gives `"!"` (col 2) the highest score of 4.7 — meaning it thinks `"!"` is the most likely next token.

These are raw scores, not probabilities yet. To turn them into probabilities, apply **softmax**:

```
softmax(x)_i = exp(x_i) / sum(exp(x_j))
```

Applied to position 1's row `[-0.5, -1.1, 4.7, -3.2, 2.1]`:

```
"Hello"   →  0.5%
"world"   →  0.3%
"!"       → 92.3%  ← model is almost certain
"goodbye" →  0.0%
"<end>"   →  6.8%
```

---

## The Hello World example

Input: `["Hello", "world"]` → token IDs: `[0, 1]`

**At inference** you only care about the last row — it tells you what token most likely follows the full input sequence. Here the model is 92.3% confident `"!"` comes next.

**At training** you keep every row because every position gives a learning signal. The targets are the shifted sequence — each token predicts the one after it:

```
Input:   ["Hello",  "world",  "!"    ]
Targets: ["world",  "!",      "<end>"]
```

- Position 0: after `"Hello"` → should predict `"world"`
- Position 1: after `"Hello world"` → should predict `"!"`
- Position 2: after `"Hello world !"` → should predict `"<end>"`

Three learning signals from one forward pass.

---

## What the model is made of

The model is not a lookup table of sequences → next token. That would be impossible — the number of possible input sequences is infinite.

Instead the model is a **function** — a fixed set of matrices (tensors) that transforms any input token sequence into logits. The knowledge of what follows what is encoded implicitly in the values of those matrices, spread across all of them. No single number says `"Hello world → !"`. That answer emerges from running the full computation.

---

## Weights and parameters

The model's matrices are called **weights** or **parameters** — two words for the same thing.

| Term | Meaning |
|---|---|
| Weight / parameter | A single float number inside a matrix |
| Tensor / matrix | A named array of weights with a specific shape |
| Model | The full collection of all tensors |

Each tensor has a name and a shape:

```
"transformer.wte.weight"          shape (32768, 768)   ← embedding lookup table
"transformer.h.0.attn.c_q.weight" shape (768, 768)     ← attention projection
"transformer.h.0.mlp.c_fc.weight" shape (3072, 768)    ← MLP expansion
"lm_head.weight"                  shape (768, 32768)   ← output projection
... hundreds more tensors ...
```

Nanochat's model has roughly **1.3 billion parameters** total — 1.3 billion individual float numbers stored in these matrices.

Saved to disk as a dictionary of named tensors — just binary arrays of floats. No sequences, no mappings, no lookup tables. When you want a prediction, you load these matrices and run the forward pass.

---

## Tensors vs matrices

A tensor is the general term for an array of numbers of any number of dimensions:

```
scalar     shape ()              a single number
vector     shape (768,)          a 1D list
matrix     shape (768, 768)      a 2D grid  ← most weight tensors are this
3D tensor  shape (32, 2048, 768) batch × sequence × embedding
```

In practice: weight tensors are matrices (2D). The activations flowing through the model during a forward pass are 3D tensors (batch × sequence × embedding). "Tensor" is just the word that covers all cases.

---

## The forward pass

The computation that turns input tokens into logits:

```
input token IDs  [0, 1]
      ↓
  forward pass through all weight matrices
      ↓
logits  (2 rows × 5 columns)
```

The weights are **fixed** during inference — they don't change. The same weights produce different logits for different inputs. The weights are the learned function; the logits are its output for a specific input.

All positions are computed **simultaneously** in one pass — the transformer processes the full sequence at once, not token by token.
