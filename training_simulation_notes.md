# Training Simulation Notes

> A self-contained walkthrough of one training loop iteration — using concrete numbers to show how embeddings and weights update during learning.

---

## Definitions

### Vocabulary

A fixed set of all tokens the model knows about. Each token has an integer ID. These IDs never change after the tokenizer is trained.

```
ID 0 = "hello"
ID 1 = "world"
ID 2 = "hi"
ID 3 = "cat"
```

---

### Token embedding

Each token in the vocabulary gets a vector of floats — its **embedding**. This vector is the token's learned representation. Tokens that appear in similar contexts end up with similar vectors after training.

The embedding table has shape **(vocab_size × embedding_size)**:

- One row per token in the vocabulary
- Each row is the token's embedding vector

The embedding size (768 in nanochat, 2 in our example) is a hyperparameter chosen before training. All embeddings start as random numbers and are updated during training.

```
           dim 1    dim 2
"hello"  [ 0.5,    0.1  ]
"world"  [ 0.0,    0.6  ]
"hi"     [ 0.1,    0.8  ]
"cat"    [ 0.7,   -0.5  ]
```

When the model sees an input token, it looks up that token's row in this table. This is the only information the token carries into the forward pass.

---

### Weight matrix

A second set of learned numbers — one vector per vocabulary token, same size as the embeddings.

```
           dim 1    dim 2
"hello"  [ 0.3,    0.2  ]
"world"  [ 0.1,    0.4  ]
"hi"     [ 0.5,   -0.1  ]
"cat"    [-0.2,    0.3  ]
```

The weight matrix is what the model uses to *score* how likely each vocabulary token is to come next. Where embeddings answer "what is this token?", weight vectors answer "how well does something match this token?".

Like embeddings, all weight vectors start as **random numbers** and are adjusted during training. Neither the embeddings nor the weights have any meaningful values before training begins — they are noise, and training is the process of turning that noise into something useful.

---

### Dot product

The operation that turns an embedding vector into a score for a candidate token.

```
dot(a, b) = a[0]×b[0] + a[1]×b[1] + ... + a[n]×b[n]
```

To score how likely `"world"` follows `"hello"`:

```
dot(e_hello, w_world) = 0.5×0.1 + 0.1×0.4 = 0.05 + 0.04 = 0.09
```

A higher dot product means the model thinks that pairing is more likely.

---

### Logits

The full table of scores — one dot product per (input position, vocabulary token) pair.

- **Rows**: one per input position (how many tokens the model has seen so far)
- **Columns**: one per vocabulary token (all possible next tokens)

At inference you only need the last row — the scores for what follows the entire input sequence. At training you use every row as a learning signal (each token predicts the one after it).

---

### Softmax

Converts raw scores (logits) into probabilities that sum to 1:

```
softmax(x)_i = exp(x_i) / sum(exp(x_j) for all j)
```

Larger scores get amplified; smaller scores get suppressed. The result is a probability distribution over the vocabulary.

---

### Cross-entropy loss

Measures how wrong the model's prediction was. Defined as the negative log probability assigned to the correct answer:

```
loss = -log(probability of correct token)
```

- If the model gave the correct token 90% probability → loss = -log(0.90) ≈ 0.105  (low, good)
- If the model gave the correct token 10% probability → loss = -log(0.10) ≈ 2.303  (high, bad)

The goal of training is to minimize this loss across all training examples.

---

### Error signal (gradient)

How much the loss would decrease if a weight or embedding value increased by a tiny amount. This is computed by backpropagation — running the chain rule backwards through the computation.

**Key property**: the gradient is not a single scalar applied equally to all dimensions. Each dimension gets a different update because each dimension contributed differently to the wrong answer.

For an embedding update, the gradient in each dimension is proportional to the **corresponding weight value** for the predicted token:

```
gradient for dim d = (predicted_prob - 1) × w_predicted[d]   ← for the input token embedding
gradient for dim d = (predicted_prob - 1) × e_input[d]       ← for the predicted token's weight
```

The `(predicted_prob - 1)` factor is the error: it is 0 if the model was perfect (probability = 1), and more negative the worse the model was.

**Why this formula? The chain rule through a multiplication.**

The score is computed as a dot product:

```
score_world = e_hello[0] × w_world[0] + e_hello[1] × w_world[1]
```

If you increase `w_world[0]` by a tiny amount, `score_world` changes in proportion to `e_hello[0]` — because that is what `w_world[0]` is multiplied by. The chain rule captures this:

```
gradient of w_world[0] = error × e_hello[0]
gradient of w_world[1] = error × e_hello[1]
```

And symmetrically, `e_hello[0]` is multiplied by `w_world[0]`, so:

```
gradient of e_hello[0] = error × w_world[0]
gradient of e_hello[1] = error × w_world[1]
```

The pattern: **each value's gradient = error × the other value it was multiplied with.**

The error is negative when the model was wrong (e.g. `-0.756`). Subtracting a negative gradient means the values increase — which increases the score for the correct token. That is how the model corrects itself.

---

### Learning rate

A small scalar (e.g. 0.3) that controls how large each update step is. Prevents overshooting the minimum.

```
new_value = old_value - learning_rate × gradient
```

A learning rate that is too large causes the model to overshoot and diverge. Too small and training is very slow.

---

### Cosine similarity

Measures how similar two vectors are in *direction*, ignoring their magnitude. Ranges from -1 (opposite) to +1 (identical direction):

```
cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b||)
```

Used to track whether token embeddings are converging toward each other during training. Tokens that appear in similar contexts should end up with similar embeddings.

---

## Training Simulation

### Setup

- **Vocabulary**: hello(0), world(1), hi(2), cat(3)
- **Embedding size**: 2 (two floats per token)
- **Learning rate**: 0.3
- **Training examples**: `hello → world`, `hi → world`, `cat → hi`

**Initial embeddings** (random starting values):

```
e_hello = [0.5,  0.1]
e_world = [0.0,  0.6]
e_hi    = [0.1,  0.8]
e_cat   = [0.7, -0.5]
```

**Initial weight matrix** (random starting values):

```
w_hello = [ 0.3,  0.2]
w_world = [ 0.1,  0.4]
w_hi    = [ 0.5, -0.1]
w_cat   = [-0.2,  0.3]
```

---

### Step 1: Train on "hello → world"

**Goal**: given `"hello"`, predict `"world"`.

**Forward pass** — compute dot product of `e_hello` with every weight vector:

```
score_hello = dot([0.5, 0.1], [ 0.3,  0.2]) =  0.5×0.3 + 0.1×0.2 =  0.17
score_world = dot([0.5, 0.1], [ 0.1,  0.4]) =  0.5×0.1 + 0.1×0.4 =  0.09
score_hi    = dot([0.5, 0.1], [ 0.5, -0.1]) =  0.5×0.5 + 0.1×(-0.1) = 0.24
score_cat   = dot([0.5, 0.1], [-0.2,  0.3]) =  0.5×(-0.2) + 0.1×0.3 = -0.07
```

**Softmax** (converting scores to probabilities):

```
exp values: [e^0.17, e^0.09, e^0.24, e^-0.07] = [1.185, 1.094, 1.271, 0.932]
sum = 4.482

P(hello) = 1.185 / 4.482 = 0.264
P(world) = 1.094 / 4.482 = 0.244   ← correct token
P(hi)    = 1.271 / 4.482 = 0.284
P(cat)   = 0.932 / 4.482 = 0.208
```

**Loss** (model gave 24.4% probability to the correct answer):

```
loss = -log(0.244) ≈ 1.410
```

**Error signal** for `"world"` (the correct target):

```
error = P(world) - 1 = 0.244 - 1 = -0.756
```

**Gradient for `e_hello`** (how to adjust the input embedding):

```
grad_e_hello[dim1] = error × w_world[dim1] = -0.756 × 0.1 = -0.076
grad_e_hello[dim2] = error × w_world[dim2] = -0.756 × 0.4 = -0.302
```

**Gradient for `w_world`** (how to adjust the target weight vector):

```
grad_w_world[dim1] = error × e_hello[dim1] = -0.756 × 0.5 = -0.378
grad_w_world[dim2] = error × e_hello[dim2] = -0.756 × 0.1 = -0.076
```

**Weight update** (subtract gradient × learning_rate):

```
e_hello[dim1] = 0.5   - 0.3 × (-0.076) = 0.5   + 0.023 = 0.523
e_hello[dim2] = 0.1   - 0.3 × (-0.302) = 0.1   + 0.091 = 0.191

w_world[dim1] = 0.1   - 0.3 × (-0.378) = 0.1   + 0.113 = 0.213
w_world[dim2] = 0.4   - 0.3 × (-0.076) = 0.4   + 0.023 = 0.423
```

**State after Step 1:**

```
e_hello = [0.523, 0.191]   ← updated (was [0.5, 0.1])
w_world = [0.213, 0.423]   ← updated (was [0.1, 0.4])
```

All other embeddings and weights are unchanged.

---

### Step 2: Train on "hi → world"

**Goal**: given `"hi"`, predict `"world"`.

Notice that `w_world` was already updated in Step 1 — the lesson from `"hello → world"` carries over.

**Forward pass** — compute dot products of `e_hi = [0.1, 0.8]` with each weight:

```
score_hello = dot([0.1, 0.8], [ 0.3,  0.2]) = 0.03 + 0.16 =  0.19
score_world = dot([0.1, 0.8], [ 0.213, 0.423]) = 0.0213 + 0.3384 = 0.360   ← uses updated w_world
score_hi    = dot([0.1, 0.8], [ 0.5,  -0.1]) = 0.05 - 0.08 = -0.03
score_cat   = dot([0.1, 0.8], [-0.2,   0.3]) = -0.02 + 0.24 =  0.22
```

**Softmax**:

```
exp values: [e^0.19, e^0.360, e^-0.03, e^0.22] = [1.209, 1.433, 0.970, 1.246]
sum = 4.858

P(hello) = 1.209 / 4.858 = 0.249
P(world) = 1.433 / 4.858 = 0.295   ← correct token, already higher than Step 1's 0.244!
P(hi)    = 0.970 / 4.858 = 0.200
P(cat)   = 1.246 / 4.858 = 0.256
```

**Loss** (model gave 29.5% probability to the correct answer — already lower loss than Step 1):

```
loss = -log(0.295) ≈ 1.214
```

The loss dropped from 1.410 to 1.214 without ever training on `"hi → world"` directly. The update to `w_world` from Step 1 immediately helped `"hi"` too, because `"hi"` and `"hello"` had similar starting embeddings in dim 2.

**Error signal**:

```
error = P(world) - 1 = 0.295 - 1 = -0.705
```

**Gradient for `e_hi`**:

```
grad_e_hi[dim1] = -0.705 × 0.213 = -0.150
grad_e_hi[dim2] = -0.705 × 0.423 = -0.298
```

**Gradient for `w_world`**:

```
grad_w_world[dim1] = -0.705 × 0.1 = -0.071
grad_w_world[dim2] = -0.705 × 0.8 = -0.564
```

**Updates**:

```
e_hi[dim1] = 0.1  - 0.3 × (-0.150) = 0.1  + 0.045 = 0.145
e_hi[dim2] = 0.8  - 0.3 × (-0.298) = 0.8  + 0.089 = 0.889

w_world[dim1] = 0.213 - 0.3 × (-0.071) = 0.213 + 0.021 = 0.234
w_world[dim2] = 0.423 - 0.3 × (-0.564) = 0.423 + 0.169 = 0.592
```

**State after Step 2:**

```
e_hello = [0.523, 0.191]   ← unchanged since Step 1
e_hi    = [0.145, 0.889]   ← updated (was [0.1, 0.8])
w_world = [0.234, 0.592]   ← updated again (was [0.213, 0.423])
```

---

### Step 3: Train on "cat → hi"

**Goal**: given `"cat"`, predict `"hi"`.

**Forward pass** — compute dot products of `e_cat = [0.7, -0.5]` with each weight:

```
score_hello = dot([0.7, -0.5], [ 0.3,  0.2]) =  0.21 - 0.10 =  0.11
score_world = dot([0.7, -0.5], [ 0.234, 0.592]) =  0.164 - 0.296 = -0.132
score_hi    = dot([0.7, -0.5], [ 0.5,  -0.1]) =  0.35 + 0.05 =  0.40   ← highest score
score_cat   = dot([0.7, -0.5], [-0.2,   0.3]) = -0.14 - 0.15 = -0.29
```

**Softmax**:

```
exp values: [e^0.11, e^-0.132, e^0.40, e^-0.29] = [1.116, 0.876, 1.492, 0.748]
sum = 4.232

P(hello) = 1.116 / 4.232 = 0.264
P(world) = 0.876 / 4.232 = 0.207
P(hi)    = 1.492 / 4.232 = 0.353   ← correct token, highest probability
P(cat)   = 0.748 / 4.232 = 0.177
```

**Loss**:

```
loss = -log(0.353) ≈ 1.041
```

**Error signal**:

```
error = P(hi) - 1 = 0.353 - 1 = -0.647
```

**Gradient for `e_cat`**:

```
grad_e_cat[dim1] = -0.647 × 0.5  = -0.324
grad_e_cat[dim2] = -0.647 × (-0.1) = +0.065
```

**Gradient for `w_hi`**:

```
grad_w_hi[dim1] = -0.647 × 0.7  = -0.453
grad_w_hi[dim2] = -0.647 × (-0.5) = +0.324
```

**Updates**:

```
e_cat[dim1] = 0.7   - 0.3 × (-0.324) = 0.7   + 0.097 = 0.797
e_cat[dim2] = -0.5  - 0.3 × (+0.065) = -0.5  - 0.019 = -0.519

w_hi[dim1] = 0.5  - 0.3 × (-0.453) = 0.5  + 0.136 = 0.636
w_hi[dim2] = -0.1 - 0.3 × (+0.324) = -0.1 - 0.097 = -0.197
```

**State after Step 3:**

```
e_cat = [0.797, -0.519]   ← updated, moved away from hello/hi space
w_hi  = [0.636, -0.197]   ← updated
```

---

## Convergence Summary

**Cosine similarity before any training:**

| Pair          | Similarity | Interpretation                  |
|---------------|------------|---------------------------------|
| hello ↔ hi   | 0.316      | moderately different direction  |
| hello ↔ cat  | 0.683      | accidentally similar at start   |

**Cosine similarity after 3 steps:**

| Pair          | Before | After | Change |
|---------------|--------|-------|--------|
| hello ↔ hi   | 0.316  | 0.455 | +0.139 ↑ converging |
| hello ↔ cat  | 0.683  | 0.546 | -0.137 ↓ diverging  |

`"hello"` and `"hi"` both predicted `"world"` → their embeddings pulled toward the same direction → cosine similarity increased.

`"cat"` predicted `"hi"` (a different target) → its embedding pulled in a different direction → cosine similarity with `"hello"` decreased.

This is the core mechanism by which tokens that appear in similar contexts end up in the same region of embedding space, purely as a side effect of gradient updates — no explicit rule says "hello and hi are synonyms."

---

## What real training adds

This simulation used a 2-dimensional embedding and 4 tokens. Real nanochat uses:

- **32,768 tokens** in the vocabulary
- **768 dimensions** per embedding (giving the model much more capacity to encode meaning)
- **Hundreds of weight matrices** (not just one) — the model is a stack of transformer blocks
- **Millions of training examples** — gradients accumulate over many steps
- All positions in the sequence contribute learning signals simultaneously (not one token at a time)

The mechanics are identical. Scale is what differs.
