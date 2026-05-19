# Tokenizer Training Notes

> Discussion notes on how BPE tokenizer training works in nanochat, written as a from-scratch explanation.

---

## What is a tokenizer and why do we need one?

The model works with integers, not text. The tokenizer converts any string into a list of integer IDs and back. Three options exist, each with trade-offs:

| Scheme | "Hello world" | Problem |
|---|---|---|
| Raw bytes | `[72,101,108,108,111,32,119,...]` | Sequences ~4× too long; model wastes capacity on trivial patterns |
| Words | `[1024, 8532]` | Vocabulary explodes; `"running"` and `"runner"` are unrelated |
| BPE | `[15496, 995]` | Compressed but recoverable; unseen text falls back to bytes |

BPE is a compression algorithm. Common chunks get one slot in the vocabulary table; rare text is spelled out byte by byte.

---

## What does "learns vocabulary from the pretraining corpus" mean?

The vocabulary is a dictionary: `{bytes → integer_id}`. BPE builds it by discovering which byte sequences appear together most often in the training data and promoting them to their own token. It is not hand-crafted — it is derived entirely from the corpus.

---

## BPE — Byte Pair Encoding

The name comes from the algorithm's core operation: find the most frequent *pair* of *bytes*, merge them into one. Repeat. Originally a 1994 data compression algorithm by Philip Gage, adapted for NLP by Sennrich et al. in 2015. Used in GPT-2, GPT-4, Llama, and nanochat.

### Step 0 — Seed the vocabulary with all 256 bytes

```python
vocab = {bytes([i]): i for i in range(256)}
# b'\x00'→0, b'\x01'→1, ... b'H'→72, b'e'→101, b'l'→108 ...
```

Every possible byte sequence can already be encoded at this point, just slowly. The algorithm will add at most `vocab_size - 256` tokens by merging pairs.

### Step 1 — Pre-tokenize with a regex (before BPE runs)

From `nanochat/tokenizer.py:30`:

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

This runs before BPE and splits text into chunks that BPE cannot merge across:

```
"Hello, world! 42" → ["Hello", ",", " world", "!", " 4", "2"]
```

BPE merges only happen *inside* a chunk, never across boundaries. This means `"end"` and `"ending"` can never accidentally fuse. The leading space before `" world"` is intentionally part of that chunk — so `" world"` (with space) can become one token.

### Step 2 — Convert each chunk to raw bytes

```
"Hello" → [b'H', b'e', b'l', b'l', b'o']
```

BPE works purely on sequences of byte tokens from here on.

### Step 3 — Count all adjacent pairs across the entire corpus

Scan every document in the training slice. For each document, split into chunks, convert to bytes, and count every adjacent pair:

```
(b't', b'h') → 6,800,000 occurrences
(b'l', b'l') → 4,200,000 occurrences
(b'e', b' ') → 3,100,000 occurrences
...
```

### Step 4 — Merge the most frequent pair into a new token

Most frequent pair: `(b't', b'h')`. Create a new token:

```python
vocab[b'th'] = 256   # next available id
```

Replace every `[... b't', b'h' ...]` in the corpus with `[... b'th' ...]`.

### Step 5 — Repeat until the vocabulary is full

```
iteration 1:  b'th'   → id 256
iteration 2:  b'he'   → id 257
iteration 3:  b'the'  → id 258
iteration 4:  b' the' → id 259   (space included!)
...
iteration N:  b' information' → id 32751
```

Nanochat trains for `vocab_size - len(SPECIAL_TOKENS) = 32768 - 8 = 32760` merges (`tokenizer.py:175-176`).

**The corpus pass cost.**

> **number of corpus passes = number of merges = vocab_size − 256**

Each iteration (Steps 3–4) requires a full scan of the entire corpus to recount pairs after the previous merge. For nanochat that is **32,760 full passes** over 2B characters. This is why tokenizer training is its own dedicated step run once before pretraining, and why training it on the full 400B corpus would be prohibitively slow. The 2B character budget (`--max-chars`) is a direct trade-off against this cost.

### What the output looks like

The result is `mergeable_ranks` — a dict from `bytes → int` where the int encodes merge priority (lower = merged earlier = more common):

```python
mergeable_ranks = {
    b'\x00': 0,
    ...
    b'H': 72,
    b'e': 101,
    b'l': 108,
    ...
    b'th':  256,      # first ever merge
    b'the': 258,      # merged shortly after
    b' the': 259,     # even more common (includes space)
    ...
    b' information': 32751,
}
```

This dict is the entire vocabulary. Nanochat saves it via `tiktoken.Encoding` and pickles it to disk (`tokenizer.py:184-189`, `tokenizer.py:258-263`).

---

## Inference tokenization

### Two libraries, two jobs

Training BPE and running BPE at inference are fundamentally different operations:

- **Training**: scan the entire corpus, count every adjacent pair, find the max, merge. Slow, write-heavy, stateful — runs once.
- **Inference**: take a single string, apply the learned merges as fast as possible. Read-only, latency-sensitive — runs on every prompt.

No single library is best at both, so nanochat splits the job:

| Job | Library | Why |
|---|---|---|
| Training | `rustbpe` | Purpose-built for fast pair counting in Rust |
| Inference | `tiktoken` | OpenAI's library, purpose-built for fast encoding throughput |

After rustbpe trains, it hands its entire learned vocabulary — `mergeable_ranks` — directly to tiktoken (`tokenizer.py:179-189`). tiktoken does not retrain anything. The two libraries share the exact same vocabulary; rustbpe builds it, tiktoken consumes it.

The `HuggingFaceTokenizer` class also exists in the codebase and implements both training and inference, but it is commented out in production (`tokenizer.py:394`). It adds a byte-to-unicode mapping layer (`ByteLevel` pre-tokenizer) where every raw byte is mapped to a printable Unicode character before BPE runs (e.g. space `0x20` → `Ġ`). rustbpe/tiktoken skip this indirection and work directly with raw `bytes` objects. For equivalent vocabularies on the same text, both produce identical token IDs — the intermediate representation differs but the final integer sequence is the same.

---

### The inference algorithm

At inference time you have one thing: `mergeable_ranks`, a `dict[bytes, int]` where the value is the merge rank (lower = merged earlier = more common = higher priority).

#### Step 1 — Pre-tokenize with the same regex used during training

```python
import regex
chunks = regex.findall(SPLIT_PATTERN, "Hello world")
# → ["Hello", " world"]
```

The same pattern is used at both training and inference. BPE merges never cross chunk boundaries.

#### Step 2 — Convert each chunk to a list of single-byte tokens

```python
parts = [bytes([b]) for b in "Hello".encode('utf-8')]
# → [b'H', b'e', b'l', b'l', b'o']
```

#### Step 3 — Repeatedly merge the lowest-rank adjacent pair

On every iteration:
1. Check every adjacent pair in `parts`
2. Concatenate them and look up in `mergeable_ranks`
3. Find the pair with the **lowest rank**
4. Merge it (replace the two entries with one)
5. Repeat until no adjacent pair exists in the vocabulary

Concrete trace for `[b'H', b'e', b'l', b'l', b'o']`:

```
Initial:  [b'H', b'e', b'l', b'l', b'o']

Adjacent pairs:
  b'He'  rank=312
  b'el'  rank=501
  b'll'  rank=256  ← lowest
  b'lo'  rank=743

Merge b'll':  [b'H', b'e', b'll', b'o']

Adjacent pairs:
  b'He'   rank=312  ← lowest
  b'ell'  rank=∞   (not in vocab)
  b'llo'  rank=891

Merge b'He':  [b'He', b'll', b'o']

Adjacent pairs:
  b'Hell'  rank=∞  (not in vocab)
  b'llo'   rank=891 ← lowest

Merge b'llo':  [b'He', b'llo']

Adjacent pairs:
  b'Hello'  rank=4821 ← only option

Merge b'Hello':  [b'Hello']

No more pairs. Done.
```

#### Step 4 — Map each remaining token to its ID

```python
ids = [mergeable_ranks[token] for token in parts]
# → [4821]
```

#### A minimal Python implementation

```python
def encode_chunk(chunk: str, mergeable_ranks: dict) -> list[int]:
    parts = [bytes([b]) for b in chunk.encode('utf-8')]

    while True:
        best_rank = float('inf')
        best_idx = -1
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i + 1], float('inf'))
            if rank < best_rank:
                best_rank = rank
                best_idx = i

        if best_idx == -1:
            break

        parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
        parts.pop(best_idx + 1)

    return [mergeable_ranks[token] for token in parts]


def encode(text: str, mergeable_ranks: dict, pattern: str) -> list[int]:
    import regex
    ids = []
    for chunk in regex.findall(pattern, text):
        ids.extend(encode_chunk(chunk, mergeable_ranks))
    return ids
```

#### Performance: O(n²) naive, O(n log n) in tiktoken

The implementation above rescans all pairs on every iteration — O(n²) per chunk. tiktoken avoids this by storing parts as **start positions** into the original byte string rather than byte slices, and only updating the two neighbors of a merged pair on each step:

```python
# Instead of: [b'H', b'e', b'l', b'l', b'o']
# Store:       [0, 1, 2, 3, 4, 5]   ← indices, with sentinel at end
#               H  e  l  l  o

# Token i spans: chunk_bytes[parts[i] : parts[i+1]]
# Rank of merging i with i+1: mergeable_ranks.get(chunk_bytes[parts[i]:parts[i+2]], ∞)
```

When you merge position `i` (remove `parts[i+1]`), only two ranks change: the pair `(i-1, i)` and the pair `(i, i+2)`. Everything else is untouched. tiktoken's Rust implementation exploits this to avoid the full rescan.

#### Special tokens bypass the algorithm entirely

Special tokens like `<|bos|>` are matched by exact string lookup before the regex runs, never touched by BPE:

```python
special_tokens = {"<|bos|>": 32760, "<|user_start|>": 32761, ...}
```

tiktoken scans for special token strings first, splits the text around them, then runs the BPE loop only on the segments in between.

---

### Why inference must produce the same tokens as training

This is an essential property, not just a nice-to-have.

During pretraining, every weight in the model was updated based on token ID sequences produced by the training tokenizer. The model learned that token 4821 (`b'Hello'`) appears in certain contexts, that token 256 (`b'll'`) tends to follow certain patterns, and so on.

If at inference you encode the same text differently — say `"Hello"` becomes `[312, 891]` instead of `[4821]` — you are feeding the model an input sequence it has never seen in that form. The embeddings for those IDs will have been shaped by entirely different training contexts. The model produces different, and likely worse, output for semantically identical input.

**A concrete example of what breaks:**

Imagine during training `"Hello"` always merged into one token `4821`. But at inference the encoder stops one step early and produces `[b'He', b'llo']` = `[312, 891]`.

- Token `4821` → model learned: "greeting, next token is probably punctuation or a name"
- Token `312` → model learned: "start of a word beginning with 'He', probably he/her/here/heat..."

Same text, completely different model behavior. The model has no way to know these two representations are equivalent — it only sees integers.

**This is also why you cannot swap tokenizers between models.** GPT-2's tokenizer and Llama 3's tokenizer both encode `"Hello"` — but to different IDs via different merge histories. You cannot use a Llama 3 tokenizer with GPT-2 weights; the integer sequences would be completely foreign to the model's learned associations.

---

## Vocabulary size must be chosen before training

There is no way to "discover" the right vocabulary size from the data. You commit to it upfront — it is the stopping condition for the merge loop. This is a real constraint: the tokenizer is trained once, saved, and the entire pretraining and SFT pipeline uses it. Changing vocab size means restarting from scratch.

| Smaller vocab (e.g. 8K) | Larger vocab (e.g. 100K) |
|---|---|
| Cheaper embedding table | More expensive embedding table |
| More tokens per word → longer sequences | Fewer tokens per word → shorter sequences |
| Better fallback for rare/foreign text | Whole words become single tokens |
| Fine for small models | Better for large models |

GPT-2 (50,257), GPT-4 (~100K), and Llama 3 (128K) are all frozen decisions made before training began. Nanochat uses **32,768 (2^15)** (`tok_train.py:19`).

---

## The digit-grouping choice: `\p{N}{1,2}`

The regex caps how many consecutive digits form a single chunk before BPE runs. This is a special exception for numbers — it does not apply to letters or punctuation.

**Why cap digits specifically?**

There are only 26 letters, and BPE naturally figures out which letter combinations are worth merging (common words, suffixes, prefixes). But 10 digits can combine into an open-ended set of numeric strings: "42", "2024", "99999". Without a cap, BPE might burn vocabulary slots on arbitrary numbers like "1987" or "365" that appear just enough to earn a merge but aren't useful tokens.

**Effect of different values:**

| Setting | `"2024"` pre-tokenizes to | Max numeric token | Vocab slots at risk |
|---|---|---|---|
| `{1,1}` | `["2","0","2","4"]` | 1 digit | 10 |
| `{1,2}` (nanochat) | `["20","24"]` | 2 digits | ~110 |
| `{1,3}` (GPT-4) | `["202","4"]` | 3 digits | ~1,110 |
| `{1,4}` | `["2024"]` | 4 digits | ~11,110 |

With a 32K vocabulary, `{1,3}` might burn ~1,000 slots on number patterns. `{1,4}` could consume a third of the entire vocabulary. Those are slots that would otherwise go to common word endings like `"ing"`, `"tion"`, `"ly"` — far more valuable for language modeling.

From `tokenizer.py:27-29`:
```python
# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
```

GPT-4 uses `{1,3}` because it has a ~100K vocabulary — large enough that 1,000 number slots is affordable.

---

## Language coverage

The tokenizer is **algorithmically language-agnostic** but the **learned vocabulary reflects the corpus language distribution**.

### Why the tokenizer can handle any language

- The regex uses `\p{L}` (Unicode Letter) and `\p{N}` (Unicode Number) — these match any script, not just Latin characters.
- Byte-level BPE means any Unicode character can always be encoded by falling back to its raw UTF-8 bytes (`tokenizer.py:77`).

### Why the vocabulary is English-biased in practice

BPE learns merges from frequency. If 95% of the corpus is English, ~95% of the 32K vocabulary slots go to English word fragments. A Chinese character like `你` takes 3 UTF-8 bytes → up to 3 tokens if no merge was learned for it. The common English word `" the"` gets 1 token.

| Language | Result |
|---|---|
| English | Common words → 1 token. Efficient. |
| French/Spanish/German | Partial coverage. Common words may get tokens. |
| Chinese/Japanese/Arabic | Likely byte-by-byte. 2–4 tokens per character. |

All evaluation tasks in nanochat (MMLU, GSM8K, ARC, HumanEval) are English-only, confirming the English focus of the project.

**To build a multilingual model** you would need either a much larger vocabulary (to give non-English scripts enough slots) or a corpus with deliberate language balancing — the approach taken by mBERT, XLM-R, and Llama 3 (128K vocab).

---

## Nanochat-specific settings (`tok_train.py`)

| Setting | Value | Reason |
|---|---|---|
| `--vocab-size` | 32,768 (2^15) | Power of 2; small enough for cheap embedding tables |
| `--max-chars` | 2B characters | Enough to learn all common English patterns; full 400B not needed |
| `--doc-cap` | 10,000 chars | Prevents a single huge document from dominating pair counts |
| Special tokens | 8, added post-training | Never merged; always exact IDs; not learned by BPE |

### Three separate limits — not the same thing

These are easy to confuse because all three use the word "characters", but they operate at completely different levels:

| Limit | Where | What it caps | Why |
|---|---|---|---|
| `\p{N}{1,2}` | regex split pattern | Digits per pre-tokenization chunk | Saves vocab slots from being spent on long numbers |
| `--doc-cap 10_000` | `tok_train.py` | Characters per individual document | Prevents one huge doc from dominating pair counts |
| `--max-chars 2_000_000_000` | `tok_train.py` | Total characters read from the corpus | Training time budget; stop consuming data after 2B chars |

From `tok_train.py:28-43`:

```python
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:args.doc_cap]   # cap each doc at 10,000 chars
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:     # stop after 2B chars total
                return
```

`--max-chars` has nothing to do with digits or merge boundaries — it is purely a training budget that controls how much of the 400B-token corpus the BPE trainer actually sees.

### Special tokens

```python
SPECIAL_TOKENS = [
    "<|bos|>",            # beginning of every document
    "<|user_start|>",     # ] chat
    "<|user_end|>",       # ] format
    "<|assistant_start|>",# ] markers
    "<|assistant_end|>",  # ] (SFT only)
    "<|python_start|>",   # ] tool
    "<|python_end|>",     # ] use
    "<|output_start|>",   # ]
    "<|output_end|>",     # ]
]
```

Added after BPE finishes, assigned the highest IDs. The regex ensures BPE never naturally produces these strings.

### token_bytes.pt

After training, `tok_train.py` computes and saves a tensor mapping every token ID to its UTF-8 byte length. This is used by `loss_eval.py` to compute **bits per byte (BPB)** — a vocab-size-invariant loss metric that allows fair comparison between models with different vocabulary sizes.

---

## Evaluating tokenizer quality (`tok_eval.py`)

### The metric: compression ratio

```python
ratio = len(text.encode('utf-8')) / len(tokenizer.encode(text))
```

Bytes divided by tokens. **Higher is better** — more bytes per token means each token carries more information, shorter sequences, and less compute for the model downstream.

| Ratio | Meaning |
|---|---|
| ~1.0 | No compression — every byte is its own token |
| ~3.0 | Decent — common short words are single tokens |
| ~4.0+ | Good — most common words are one token |

### What it tests

Five hand-picked text types, each probing a different dimension:

| Sample | What it tests |
|---|---|
| `news_text` | Everyday English prose |
| `korean_text` | Non-English script (expected to be poor) |
| `code_text` | Python source code |
| `math_text` | LaTeX math notation |
| `science_text` | Dense technical English |
| `fwe-train` | Actual training data — tokenizer saw this, expected highest ratio |
| `fwe-val` | Held-out validation data — the honest number |

Then benchmarks nanochat's tokenizer against GPT-2 (50K vocab) and GPT-4 (`cl100k_base`, ~100K vocab).

### What to expect

- **English prose**: nanochat competitive with GPT-2, worse than GPT-4 (smaller vocab = fewer whole-word tokens)
- **Korean**: all three poor; GPT-4 less so because its larger vocab has more non-Latin coverage
- **Code**: all decent — short repetitive tokens (`def`, `self`, `=`) get picked up regardless of vocab size
- **Training data**: nanochat's highest ratio — exactly the documents it was trained on
- **Validation data**: slightly lower — the fair generalisation measure

### The roundtrip assert

```python
decoded = tokenizer.decode(encoded)
assert decoded == text
```

Every test also verifies encode → decode returns the **exact original string**. This catches lossy byte handling, Unicode encoding bugs, or missing character coverage. A tokenizer that cannot roundtrip perfectly is broken regardless of its compression ratio.

---

## How runtimes know which tokenizer to use

Models do not ship executable code. The tokenizer is data bundled with the model artifact — the runtime provides the algorithm implementation, and the model provides the vocabulary and configuration.

### HuggingFace format (multiple files)

A HuggingFace model repository contains both weights and tokenizer as separate files:

```
model/
├── model.safetensors        ← weights
├── tokenizer.json           ← vocabulary + merge rules + regex pattern
├── tokenizer_config.json    ← tokenizer class name, special tokens
└── special_tokens_map.json
```

`tokenizer_config.json` stores a string class name like `"LlamaTokenizer"`. The runtime (`transformers` library) matches that string against a hardcoded registry of classes it already has compiled in:

```python
TOKENIZER_MAPPING = {
    "LlamaTokenizer":            LlamaTokenizer,
    "GPT2Tokenizer":             GPT2Tokenizer,
    "BertTokenizer":             BertTokenizer,
    ...
}
cls = TOKENIZER_MAPPING[config["tokenizer_class"]]
tokenizer = cls.from_pretrained(model_dir)
```

All those classes are Python code that ships with the `transformers` package — not with the model. The model provides the **name** and the **data**; the runtime provides the **algorithm**.

If a model uses an unknown class name (e.g. a new architecture released before the library was updated), the runtime errors or falls back. This is a real compatibility problem. The newer `tokenizer.json` format solves it by making the tokenizer fully reconstructable from data alone — no class name needed.

### GGUF format (single file, used by Ollama)

GGUF embeds everything — weights, vocabulary, merge rules, special tokens, metadata — in one binary file. It stores a type identifier (`"gpt2"`, `"llama"`) rather than a class name, which maps to one of a small number of tokenizer algorithms hardcoded in llama.cpp (C++). No class registry, no extensibility — if the type is unknown, the file cannot be run.

### Summary

| Layer | Who provides it | Language |
|---|---|---|
| Algorithm (how BPE merging works) | Runtime (transformers / llama.cpp / tiktoken) | Python / C++ / Rust |
| Class registry (name → algorithm) | Runtime | Python / C++ |
| Vocabulary + merge rules | Model file | Data (JSON / binary) |
| Class name or type identifier | Model file | String in JSON / GGUF metadata |

Models ship **data and identifiers**, never code.

---

## Reconstructing a tokenizer from data alone

Given only three pieces of data from `tokenizer.json`, you can build a fully working tokenizer with no model-specific code.

### The three pieces

**1. Regex pattern** — stored as a plain string, compiled at load time. Identical to the `SPLIT_PATTERN` used during training.

```json
{ "pre_tokenizer": { "pattern": { "Regex": "'(?i:[sdmt]|ll|ve|re)|..." } } }
```

**2. Vocabulary** — a dict mapping every token (as a string) to its integer ID:

```json
{
  "vocab": {
    "!":       0,
    "Ġ":     220,
    "th":    256,
    "Ġthe":  259,
    "Ġworld": 995
  }
}
```

The `Ġ` is not a real character — it is a stand-in for the space byte `0x20`. HuggingFace's `ByteLevel` pre-tokenizer maps every byte (0–255) to a printable Unicode character so the vocabulary can be stored as readable JSON. This mapping is fixed and identical for all models that use `ByteLevel`:

```python
def bytes_to_unicode():
    # Printable ASCII and Latin chars map to themselves
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = list(bs)
    # Control chars, space, etc. → chars starting at U+0100
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

BYTE_TO_UNICODE = bytes_to_unicode()
# 0x20 (space) → 'Ġ'
# 0x09 (tab)   → 'Ā'
# 0x48 ('H')   → 'H'   (printable ASCII maps to itself)
```

So `"Ġworld"` in the vocab means the token whose actual bytes are `b' world'`. tiktoken/nanochat skip this layer entirely and store raw `bytes` objects directly.

**3. Merge rules** — an ordered list of pairs, one per merge performed during training:

```json
{
  "merges": [
    "t h",
    "th e",
    "Ġ t",
    "Ġt he",
    "Ġ the",
    ...
  ]
}
```

Line 0 was the first merge (most frequent pair in the corpus). Line 32,759 was the last. **The order encodes priority** — this is all you need to reconstruct `mergeable_ranks`:

```python
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# Seed with all 256 base bytes
mergeable_ranks = {bytes([i]): i for i in range(256)}

# Replay merges in order → each gets the next available rank
for rank, merge_str in enumerate(merges_list):
    a, b = merge_str.split(" ")
    a_bytes = bytes([UNICODE_TO_BYTE[c] for c in a])
    b_bytes = bytes([UNICODE_TO_BYTE[c] for c in b])
    mergeable_ranks[a_bytes + b_bytes] = 256 + rank
```

After this loop you have the exact same `mergeable_ranks` dict tiktoken uses — derived purely from the merge list, no model code needed.

### Why both vocab and merge rules are stored

They are redundant — the vocab is derivable from the merges — but serve different purposes:

| Data | Used for |
|---|---|
| `merges` | Encoding — tells you which pair to merge and in what order |
| `vocab` | Decoding — maps final token bytes to integer IDs; enables O(1) token → string lookup |

### Full working implementation

```python
import regex, json

with open("tokenizer.json") as f:
    data = json.load(f)

SPLIT_PATTERN = data["pre_tokenizer"]["pattern"]["Regex"]
vocab         = data["model"]["vocab"]      # {"Ġworld": 995, ...}
merges_list   = data["model"]["merges"]     # ["t h", "th e", ...]

# build byte ↔ unicode mapping
BYTE_TO_UNICODE = bytes_to_unicode()        # function defined above
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# reconstruct mergeable_ranks from merge list
mergeable_ranks = {bytes([i]): i for i in range(256)}
for rank, merge_str in enumerate(merges_list):
    a, b = merge_str.split(" ")
    a_bytes = bytes([UNICODE_TO_BYTE[c] for c in a])
    b_bytes = bytes([UNICODE_TO_BYTE[c] for c in b])
    mergeable_ranks[a_bytes + b_bytes] = 256 + rank

# reconstruct vocab as bytes → id
token_to_id = {
    bytes([UNICODE_TO_BYTE[c] for c in token_str]): token_id
    for token_str, token_id in vocab.items()
}

def encode_chunk(chunk: str) -> list[int]:
    parts = [bytes([b]) for b in chunk.encode("utf-8")]
    while True:
        best_rank, best_idx = float("inf"), -1
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i+1], float("inf"))
            if rank < best_rank:
                best_rank, best_idx = rank, i
        if best_idx == -1:
            break
        parts[best_idx] = parts[best_idx] + parts[best_idx+1]
        parts.pop(best_idx + 1)
    return [token_to_id[token] for token in parts]

def encode(text: str) -> list[int]:
    return [id for chunk in regex.findall(SPLIT_PATTERN, text)
               for id in encode_chunk(chunk)]
```

### Concrete trace: encoding `"the"`

```
merges[0] = "t h"   → rank 256
merges[1] = "th e"  → rank 257
vocab["the"] = 258

encode("the")
  pre-tokenize → ["the"]
  bytes        → [b't', b'h', b'e']

  pairs: b'th'→256 (lowest), b'he'→412
  merge → [b'th', b'e']

  pairs: b'the'→257 (only option)
  merge → [b'the']

  look up → token_to_id[b'the'] = 258
  result: [258]
```
