# Data formats

Training data is **JSONL**: one JSON object per line, UTF-8. Blank lines are
skipped. The expected keys depend on `task`.

Data is validated when you submit, not when the job runs, so a malformed file
returns `400` immediately with the offending line number rather than failing in
the queue minutes later.

## sequence_classification

Single-label text classification, binary or multiclass.

```jsonl
{"text": "great value for the money", "label": "positive"}
{"text": "broke after two days", "label": "negative"}
{"text": "arrived on time", "label": "neutral"}
```

| Key | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | must be non-empty |
| `label` | string or integer | yes | integers are coerced to strings |
| `text_pair` | string | no | second segment for sentence-pair tasks |

Requires **at least 2 distinct labels** across the file.

Sentence-pair example (NLI, paraphrase, QA relevance):

```jsonl
{"text": "A man is playing guitar.", "text_pair": "A person plays an instrument.", "label": "entailment"}
```

Metrics: `accuracy`, `f1_macro`.

## multi_label_classification

Zero or more labels per example, scored independently with a sigmoid.

```jsonl
{"text": "cheap but poorly made", "labels": ["price", "quality"]}
{"text": "shipped quickly", "labels": ["delivery"]}
{"text": "no strong opinion", "labels": []}
```

| Key | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | must be non-empty |
| `labels` | array of strings | yes | may be empty |
| `text_pair` | string | no | |

An empty `labels` array is valid and means "none of the classes apply".

Prediction threshold is `JobConfig.threshold` (default `0.5`).

Metrics: `f1_micro`, `f1_macro`, `subset_accuracy` (fraction of examples where
every label is exactly right).

## token_classification

Per-token labelling: NER, chunking, POS.

```jsonl
{"tokens": ["Ada", "works", "at", "Acme", "in", "Berlin"], "tags": ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"]}
{"tokens": ["No", "entities", "here"], "tags": ["O", "O", "O"]}
```

| Key | Type | Required | Notes |
|---|---|---|---|
| `tokens` | array of strings | yes | pre-tokenised words, non-empty |
| `tags` | array of strings | yes | **must be the same length as `tokens`** |

Text must already be split into words. Length mismatch is the most common error
and is reported as `line N: 5 tokens but 4 tags`.

**Subword alignment.** DeBERTa splits words into subword pieces. Each word's
label is assigned to its *first* subword; continuation pieces and special tokens
get `-100` and are excluded from both loss and metrics. You supply word-level
tags and never think about subwords.

**Truncation.** Sequences longer than `max_length` are cut, and the tags for
dropped words are silently lost. For long documents raise `max_length` (up to
1024) or split records upstream.

Tags are expected in **IOB2** (`B-TYPE`, `I-TYPE`, `O`) because scoring uses
`seqeval`, which is entity-level: a predicted entity counts only if its full
span and type match. If your tags are not IOB2, the F1 will look wrong even
though training is fine.

The `O` tag, when present, is forced to label index 0.

Metrics: `precision`, `recall`, `f1` (all entity-level, via seqeval).

## regression

Continuous target.

```jsonl
{"text": "it was fine, nothing special", "label": 3.0}
{"text": "absolutely superb", "label": 4.8}
```

| Key | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | must be non-empty |
| `label` | number | yes | int or float; booleans rejected |
| `text_pair` | string | no | e.g. STS sentence pairs |

No label vocabulary is built (`labels` is `[]` in the job record). The model has
a single output and is trained with MSE.

Targets are **not** normalised. If your scale is large, either scale it yourself
or expect a correspondingly large MSE.

Metrics: `mse`, `mae`, and `pearson` (omitted when either predictions or targets
are constant, since correlation is undefined).

## Labels

For classification tasks the label set is inferred from the data — you never
declare it. Labels are sorted alphabetically to assign indices, except that `O`
is pinned to index 0 for token classification.

If you upload a separate `eval_file`, the label sets of both files are unioned,
so a label appearing only in eval will not crash scoring.

The mapping is written into the saved model's `id2label` / `label2id`, so the
artifact is self-describing:

```python
model.config.id2label   # {0: 'negative', 1: 'positive'}
```

Because indices come from sorted order, **adding a new label changes the
indices** of existing ones in a retrain. Always read `id2label` from the model
config rather than hardcoding integers.

## Train/eval split

- Upload an `eval_file` and it is used as-is.
- Otherwise `eval_split` (default `0.1`) of the training data is held out, using
  a seeded shuffle so the split is reproducible for a given `seed`.
- Set `eval_split: 0` to train on everything. No metrics beyond `train_steps`
  are then produced.
- Datasets with fewer than 4 rows are never split.

The split is random, **not stratified**. With few examples per class a rare class
may land entirely in one side.

## Validation errors

Every error names the line:

| Message | Cause |
|---|---|
| `line 3: invalid JSON (Expecting value)` | malformed JSON |
| `line 3: expected a JSON object` | line is an array or scalar |
| `line 3: 'text' must be a non-empty string` | missing or blank `text` |
| `line 3: 'label' must be a string or integer` | wrong type, or `null` |
| `line 3: 'label' must be a number for regression` | non-numeric target |
| `line 3: 'labels' must be a list of strings` | multi-label shape wrong |
| `line 3: 5 tokens but 4 tags` | length mismatch |
| `need at least 2 distinct labels, found ['x']` | single-class file |
| `file contains no records` | empty or all-blank file |

## Size limits

Uploads are capped by `TUNER_MAX_UPLOAD_MB` (default 512 MB) and enforced while
streaming, so an oversized file is rejected with `413` without being written to
disk in full. Empty files are rejected with `400`.
