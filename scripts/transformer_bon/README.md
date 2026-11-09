# Transformer Best-of-N

This package separates benchmark prompts, model chat templates, and confidence-
trajectory scoring.

## Request flow

```text
raw dataset row
  -> DatasetAdapter.build_request(...)
  -> BestOfNRequest(messages or raw_prompt)
  -> qwen3 / deepseek_r1_distill_qwen / raw renderer
  -> BestOfNService.generate_and_select(...)
```

The service never reads fields such as `problem`, `question`, or `title`.
Adapters in `adapters/` own those mappings and the benchmark-specific prompt.

```python
from scripts.transformer_bon import BestOfNConfig, BestOfNService
from scripts.transformer_bon.adapters import AIMEAdapter

adapter = AIMEAdapter()
request = adapter.build_request(sample, sample_idx=0)
service = BestOfNService(BestOfNConfig(
    model_path="/models/Qwen3-8B",
    checkpoint_path="/checkpoints/best.pt",
    n=8,
    chat_template="qwen3",
    enable_thinking=True,
))
result = service.generate_and_select(request)
print(result.selected_text, result.selected_score)
service.close()
```

Built-in adapters cover AIME, GPQA, and configurable field-mapped JSON/JSONL.
Additional adapters can be selected by import path through
`create_adapter("my_package.my_adapter:MyAdapter")`. ICPC-Eval's dynamic adapter
lives in `icpc_eval/bon_adapter.py` because it also consumes refinement history.

## DP/TP and compilation caches

`--bon_gpus "1;2;3;4;5;6;7"` with TP size 1 creates seven independent vLLM
data-parallel engines. With `--bon_vllm_cache_mode auto` (the default), runs
with more than one DP engine give every engine a stable, isolated
`VLLM_CACHE_ROOT`. This prevents concurrent cold-start compilation from writing
the same `rank_0_0` TorchInductor artifacts. The per-worker cache is reused on
later runs.

Use `--bon_vllm_cache_root /fast-local-disk/vllm-cache` to put those caches on
fast persistent storage. `shared` should only be used with one DP engine or a
cache that has already been completely prewarmed. `--bon_vllm_enforce_eager`
is an emergency compatibility fallback that avoids the compiled execution path
at the cost of lower throughput.

The confidence-trajectory model remains independent. For example,
`--bon_value_device cuda:0 --bon_value_residency gpu` keeps it on GPU 0 while
the generation and entropy stages use GPU groups 1-7.

## Selectable PRM backends

PRM phase scoring defaults to `qwen2.5-math-prm-7b`. Use
`--prm_name versa-prm` (or `--bon_prm_name versa-prm` in dataset runners) to
load `UW-Madison-Lee-Lab/VersaPRM`. VersaPRM is a PEFT adapter over
`UW-Madison-Lee-Lab/Llama-PRM800K`; the backend loads the base model, attaches
the adapter, preserves the benchmark prompt context, and applies the model
card's candidate-token/step-marker scoring protocol. All registered backends emit the
same phase probability range and product/min/mean/last rollout aggregates.
Install `peft` in the PRM runtime before selecting VersaPRM.

Use `--prm_name math-shepherd` to load
`peiyi9979/math-shepherd-mistral-7b-prm`. This backend follows the model card:
each solution step is followed by the fixed `ки` marker, and the score is the
softmax probability of `+` versus `-` at that marker. It uses the same `[0, 1]`
phase-score interface and product/min/mean/last aggregation as the other
backends. The checkpoint may be supplied as a Hugging Face ID or as a local
directory containing `config.json`, tokenizer files, and the PyTorch `.bin`
shards plus `pytorch_model.bin.index.json`.

Use `--prm_name rlhflow-mistral` to load
`RLHFlow/Llama3.1-8B-PRM-Mistral-Data`. This backend follows RLHFlow's causal
next-token protocol: every solution phase is represented as a user turn
followed by an assistant `+` label, and the phase score is
`softmax(logits[+, -])[+]` at the official `-3` prediction position. It emits
the same `[0, 1]` phase scores and product/min/mean/last rollout aggregates as
the other registered backends, so changing PRM backends does not change the
selection or output schema.

Use `--prm_name rlhflow-deepseek` to load
`RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`. This generative process-reward model
uses the official Llama chat-template protocol: each solution step is a user
message followed by an assistant `+` label. The backend scores the `+` versus
`-` logits at every requested phase end and returns the `P(+)` value in the
same `[0, 1]` range as the other PRMs, so all rollout aggregations remain
directly comparable.

The generic JSON/JSONL runner exposes the adapter registry directly:

```bash
python -m scripts.transformer_bon.cli \
  --input aime.json \
  --dataset_adapter aime \
  --model_path /models/Qwen3-8B \
  --checkpoint_path /checkpoints/best.pt \
  --output_dir outputs/aime_bon \
  --chat_template qwen3 \
  --enable_thinking true \
  --N 8
```

For `field_mapped`, pass constructor settings with, for example,
`--adapter_kwargs_json '{"question_field":"question","answer_field":"answer"}'`.

## PRM scoring throughput and exact output compatibility

The scoring stage keeps up to four requests in flight, with an estimated 16 MiB
text budget (one existing request remains atomic and may exceed that budget).
Scheduling runs in a background controller, with at most one scored request
waiting for the consumer in addition to that input window. Benchmark evaluation
and final-result writes therefore do not stop queue replenishment. Aggregation
and checkpoint submission remain consumer-owned: cancelling between yields does
not commit candidates for requests the consumer has not requested.
Each DP worker keeps the previous per-request LPT shard, stable length ordering
and microbatch boundaries. Faster workers can start the next queued request
while another worker finishes the previous one. Results, checkpoints and progress
callbacks are consumed in the original request/candidate order. A single-process
scorer uses one inference thread and the same device map, without another replica.
Closing the iterator cancels outstanding work; worker errors abort the pool.

CPU rendering, tokenization and marker discovery run ahead of GPU inference in a
bounded prefetch queue: one ready CPU batch and at most one being prepared. RLHFlow
encodes the exact original cumulative conversation at requested phase ends and
the final complete solution, skipping unused intermediate prefix encodings.
No prepared-token copies or additional score caches are written to disk.
Forward and softmax operations keep their original shapes and order. Only the
small probability vectors are gathered for one CPU transfer per microbatch;
the four single-row backends still execute separate forwards.

The existing `.bon_prm.jsonl` is scanned once per run into an in-memory byte-offset
index. Only candidates needed for current requests are decoded again; complete
trajectories are not retained in a second checkpoint dictionary. The writer stays
open and retains the existing serialization, row order and **per-candidate fsync**.
An ordered append thread overlaps each fsync with subsequent CPU aggregation.
Its queue holds at most two waiting encodings with a 4 MiB byte budget, allowing
one oversized record when otherwise empty. Encoding occurs before enqueueing,
so later benchmark mutations cannot change a pending row. Progress callbacks
stay on the consumer thread and run only after the corresponding fsync succeeds.
Returning a request waits for all its checkpoint writes. Write errors stop later
appends and propagate to the caller; closing the iterator joins background work.
An interrupted final append is repaired before appending new rows. Existing
complete checkpoints and final summary/steps files keep their format and resume
behavior. Normal inference no longer calls `empty_cache()` after every batch;
OOM splitting and model-unload cleanup still release unused cached allocations.

Standard prepared records and final summary/steps stores also keep run-local
offset indexes instead of all historical trajectories. Prepared layouts, input
identity validation, last-row-wins handling and imported-file precedence are
unchanged; validation still scans every existing row before scoring. Current
requests are read with one handle per source file, and old generation payloads
are released after phase preparation. Result lookups use request-key indexes;
relabel/replacement operations stream the existing atomic rewrite in the original
key order. Final serialization and native newlines are unchanged, with no extra
on-disk indexes. Queues are bounded, but their budgets are not a total process
RAM limit: models, CPU preparation buffers and one complete N-candidate result
still require memory. Lightweight index metadata grows with candidate count.

To preserve numerical results, these optimizations deliberately retain the old
dtype, attention implementation, device assignment and forward batch shapes.
FoVer, Versa, Math-Shepherd and RLHFlow-Deepseek still forward one unpadded row at
a time, overlapping CPU preparation and subsequent requests instead of replacing
those forwards with padded batches. Changing precision or batching can change
floating point scores and requires a separate numerical compatibility decision.
All existing CLI arguments and their defaults are unchanged.

### Exceptional CUDA OOM recovery

Errors in generation, entropy computation, phase preparation and scoring abort
the run instead of producing failed candidate rows. Worker exceptions are
forwarded to the parent for termination and cleanup. Writers reject errored
records; resume readers reject historical failed records rather than accepting
them as completed candidates. Previously written successful rows remain usable;
historical files are not automatically cleaned. The bounded CUDA OOM recovery
below remains active, and an unsuccessful recovery aborts as well.

After ordinary OOM batch splitting, a candidate that still fails alone is
deferred. Persistent workers finish the already-submitted window (at most four
requests); the parent retains successful scores and then releases those workers.
One temporary replica retries only failed candidates, one at a time, over the
union of the configured PRM GPU groups. It reserves activation space by limiting
resident weights to at most 3 GiB and one quarter of free memory per GPU. CPU
weight storage is limited to 32 GiB and available RAM minus the larger of 2 GiB
or half the available RAM. These are weight-placement budgets, not hard limits
on process RSS or activation allocations. Loading buffers still need headroom.
Disk weight offload and temporary tensor files are not enabled.

The first recovery attempt retains the original attention selection. If it also
raises CUDA OOM, that replica is discarded and a fresh replica tries Flash SDPA,
then Efficient SDPA separately (cuDNN workspace is excluded). If neither works,
it slices attention queries and heads, preserving all visible keys/values and
global causal positions. It uses math SDPA only on these bounded tiles, avoiding
unbounded full-sequence score matrices and copies of all GQA heads.
Estimated scratch space is limited to 512 MiB or one quarter of currently free
GPU memory, whichever is smaller. A tile OOM reduces this budget, with at most
eight attempts; this is not a hard bound on native kernel allocations or total
model memory. Remaining activations and output tensors still need headroom.
Backend choices are cached per input shape within a forward; no tensors are
cached. Remaining failed candidates can reuse the recovery replica. Unsupported
operations and insufficient memory still abort; recovery does not truncate,
change model precision, write temporary tensors, or retry indefinitely. Ordinary
workers are restarted after recovery; other errors still abort the run.
The local single-scorer path also releases its model before temporary recovery.

Input, aggregation, candidate order, checkpoint durability and output schemas
remain unchanged. Normal inference retains its previous operations and batches.
Only OOM recovery may be slower and produce small floating point differences,
potentially changing close rankings. Recovery uses layer placement rather than
true tensor parallelism; unavoidable Q/K/V/output tensors and operations outside
SDPA may still exceed one GPU's capacity. Recovery progress is logged to stdout,
not added to JSONL rows.

## LiveCodeBench ranked Top-K

The local LiveCodeBench integration generates one pool of `--bon_N` candidates,
ranks all candidates with this package, and sends only `--pass_k` candidates to
the unchanged official CodeGeneration evaluator. All `N` candidates remain in
the paired `summary.jsonl` and `steps.jsonl` files.

```bash
cd LiveCodeBench
python -m lcb_runner.runner.main \
  --dataset_source local \
  --local_dataset_path ../deepscaler/data/test/livecodebench_v6.json \
  --generation_backend transformer_bon \
  --scenario codegeneration \
  --release_version release_v6 \
  --bon_N 8 \
  --pass_k 3 \
  --bon_model_path /models/Qwen3-8B \
  --bon_checkpoint_path /checkpoints/best.pt \
  --bon_chat_template qwen3 \
  --bon_enable_thinking true \
  --bon_output_dir ../results/livecodebench_bon \
  --evaluate
```

`--pass_k` must satisfy `1 <= K <= N`. Candidate ties are resolved by the
original sequence index. Failed candidates rank last and are retained so the
official output always contains exactly `K` entries.

Use `--dataset_source online` (the default) to retain LiveCodeBench's original
download path. Local and online rows are converted to the same official
`CodeGenerationProblem` representation before prompt formatting, so changing
the source does not change the benchmark prompt or evaluator.

## GPQA Diamond Top-1

GPQA's benchmark-owned adapter and evaluator live under `GPQA/baselines/`.
They reuse this package's staged batch generation and K=1 record writer while
preserving GPQA's official closed-book prompts, answer parser, accuracy, and
refusal fraction. See `GPQA/README.md` for local-array and gated Hugging Face
examples.

## Adaptive phases and reusable token entropy

The staged Transformer and PRM benchmark runners support two phase algorithms.
The default `original` preserves the existing fixed `num_segments` and
`entropy_threshold` algorithm. Select `adaptive` with both `high_entropy_ratio`
and `target_phase_tokens`; the original two controls are then ignored.

| Entry point | Original algorithm | Adaptive algorithm |
| --- | --- | --- |
| GPQA, LiveCodeBench, ICPC-Eval, BBH, both Transformer and PRM | `--bon_phase_partition_algorithm original --bon_num_segments 8 --bon_entropy_threshold 1.2` | `--bon_phase_partition_algorithm adaptive --bon_high_entropy_ratio 0.05 --bon_target_phase_tokens 200` |
| NaturalPlan calendar/meeting/trip, shared CLI, and the three `3_best_of_n_via_phases_scores*.py` scripts | `--phase_partition_algorithm original --num_segments 8 --entropy_threshold 1.2` | `--phase_partition_algorithm adaptive --high_entropy_ratio 0.05 --target_phase_tokens 200` |

`high_entropy_ratio` is a fraction in `(0, 1]`: `0.05` means the top 5% of
finite internal reasoning-token entropies. `target_phase_tokens` must be positive.
Only reasoning through the first step containing `</think>` participates; when
there is no `</think>`, the existing full-output rule is retained. The algorithm
compares the floor and ceiling of `T / target_phase_tokens`, constrained by the
reasoning step count, and chooses the integer K minimizing the average-length
error. Ties favor fewer phases. This controls the number of phases, without
constraining individual phase lengths.

The top `ceil(p * N)` internal tokens are ranked by descending entropy, then
increasing token index to resolve equal entropies. Their last entropy is the
rollout's implicit cutoff; tied membership is determined by rank rather than a
strict `H > cutoff` comparison. Tokens are mapped with the original overlap/
separator rule to unique internal step boundaries. Each step boundary retains
its maximum token entropy and its representative token position. Sufficient
high-entropy boundaries are sampled with the original candidate-rank formula;
otherwise all high-entropy boundaries are kept and the remaining boundaries are
filled by descending maximum entropy. Boundaries are sorted in trajectory order.
The score aggregations, model features, final outputs and prepared-record schema
are unchanged.

Stage two now consists of two sequential passes. The first computes and saves
**full-response** token entropies, including tokens after `</think>`. Only after
all required rollout records are saved does the phase pass reopen the cache
and create the existing phase intermediate file. Each pass has a rollout progress
bar. Entropy rows are flushed and fsynced individually and reused on resume;
an incomplete final JSONL row is repaired before appending. Completed successful
rollouts never require another entropy forward just because phase parameters
changed. Failed entropy rows are retried on resume.

**Prepared records take precedence.** Before generation or entropy work, the
standard Transformer/PRM service checks for complete saved phase records for
each requested rollout. Successful rows must contain the existing steps,
confidence and phase-range structure; terminal prepared error rows are retained
for the scorer's existing error handling. Complete records go directly to
scoring without generation, entropy forward, tokenizer loading or repartition,
even when generation/entropy/structure cache files are absent. Partial files
reuse completed rollouts and fill only missing or structurally incomplete ones.

GPQA, LiveCodeBench, ICPC-Eval and BBH read
`<output_dir>/.bon_prepared.jsonl`. NaturalPlan also recognizes this file in its
output root, in addition to its internal `.staging/.bon_prepared.jsonl`. The
three standalone scripts import `<output_dir>/.bon_prepared.jsonl` while
retaining `.prepared_tmp.jsonl` for their own phase output. Candidate envelopes
(`request_id`/`record`), request bundles (`request_id`/`records`) and flat
prepared rows are accepted. Matching uses request/sample identity and rollout
indices, never line counts alone; the supplied complete file is not rewritten.
Standalone sampled runs require only their selected candidate slots.

Existing saved phases are reused even if phase parameters are changed. To
explicitly repartition, remove the prepared files (including an imported root
copy and any internal phase copy), keeping the entropy/structure caches. The
existing explicit `--overwrite` / `--bon_overwrite` behavior still rebuilds
preparation; it also removes the imported root phase file so it cannot shadow
the newly generated internal phase file on a later resume. Final summary/steps
resume and overwrite rules, scoring and output schemas remain unchanged.

On the shared Transformer/PRM service, `max_tokens` also limits responses read
from generation checkpoints. The generation file is left intact. Exact entropy
is computed only for the retained tokens when absent; a matching longer entropy
checkpoint supplies its existing prefix without another forward. Invalid or
insufficient cached entropy for a truncated rollout raises instead of silently
recomputing it. Prompt tokens are unchanged; `max_model_len` retains its existing
generation-engine meaning.

Full retained steps keep their saved text, confidence and first-token position.
If prepared steps are absent, the original saved text supplies the step layout.
A private reusable structure cache records the original token/character mapping;
only the affected final step is locally decoded and its confidence recomputed.
An incompatible decoder/Unicode cutoff raises rather than changing saved text.
No text-to-token encoding or model forward is added by truncation. Phase selection
and normal Transformer/PRM scoring operate on the retained trajectory, with the
existing schemas, precision and tie rules. Generation payloads are read by offset
with bounded caching; entropy DP workers receive small file references.

Prepared rows exceeding the current response budget and stale PRM scores are
not reused. Benchmark resume checks similarly reject over-budget final results,
so their existing evaluators judge the shortened text and write fresh correctness
labels. Complete in-budget prepared records retain their fast scoring-only path.
After successfully replacing internal over-budget prepared rows, their obsolete
rows are removed atomically, retaining unrelated records and existing envelopes.

When GPU groups are configured, the phase pass flattens those groups and starts
one independent worker per distinct GPU. For example, `0,1;2,3` uses four
single-GPU DP workers for phase partition, while generation and exact entropy
forward retain their original two DP groups with TP=2. Lightweight subprocesses
start directly, without re-importing the benchmark or the legacy research
script. They keep a CUDA context for the phase pass and receive bounded batches
dynamically. Cold rollouts compile their invariant structure once; warm rollouts
send only mmap descriptors. A tokenizer is loaded lazily only on a structure
cache miss. CUDA operates on cached token arrays (original mode) or much smaller
boundary summaries (adaptive mode), with one result synchronization per batch.
There is no dense token-by-step overlap matrix. No reasoning/entropy model or TP
process group is loaded. Float64 entropy operations and stable tie rules
preserve both algorithms' existing results and prepared-record schema.
The same scheduling path serves all five benchmarks, Transformer/PRM backends
and the three standalone scripts, without adding command-line parameters. An
empty effective GPU-group list retains the existing CPU phase path; standalone
scripts use the same resolved group list as their entropy stage.

The private `.phase_structure_cache/` directory lives beside the entropy JSONL
(including outside NaturalPlan's disposable `.staging/`). It contains:

* `manifest.json`: version and explicitly typed little-endian array layouts;
* `rollouts-<shard>.jsonl`: committed rollout keys, byte offsets and counts;
* `tokens-<shard>.bin`: float64 entropies, character spans, reasoning-boundary
  mapping, valid positions and the original algorithm's fallback order;
* `steps-<shard>.bin`: character spans, exact step confidences, first-token
  indices, adaptive representatives/ranks and fallback order;
* `<entropy filename>.index.json`: validated offsets into the unchanged entropy
  JSONL. An unchanged file does not need to be reparsed on each run.

The token arrays require 28 bytes per response token; step arrays require 37
bytes per step, plus JSON indexes. Text is normally referenced from generation
rather than duplicated. Binary files are memory-mapped read-only. Each worker
appends its own shard and flushes/fsyncs the arrays before committing each
rollout index. Incomplete index tails and truncated binary records are not
reused. This cache is shared by both algorithms and all parameter choices;
token IDs, token logprobs, entropy values, tokenizer artifacts/library versions
and preprocessing version participate in its identity. It may be deleted to
rebuild structures without repeating entropy forward.

For standard ByteLevel and supported SentencePiece ByteFallback decoder chains,
prefix character spans are computed in a single pass, preserving their different
replacement-character rules at incomplete UTF-8 prefixes.
Unsupported decoders use the original repeated-prefix decoding semantics on
the first build. Interval searches visit actual step overlaps rather than all
token/step pairs. Confidence summation keeps the original token order. Full-step
confidence mapping and reasoning-prefix boundary mapping remain distinct.

Adaptive queries use a boundary representative's global token entropy rank:
that boundary is high iff its representative belongs to the top `ceil(p*N)`
tokens. Ranking still includes tokens mapped to the first step or no boundary,
so the top-percentage quota and all ties match the original adaptive selector.
Original queries still select tokens before mapping/deduplicating boundaries.

Completion updates the progress bar as soon as results arrive, independently
of ordered JSONL persistence. Workers return only selected positions and cache
descriptors; the parent writes the existing prepared schema in rollout order
and does not retain a second copy of all completed prepared records. Startup
and the first cache migration remain real costs; steady-state partition time
also includes writing the unchanged full output, including token entropies
when requested. No fixed dataset runtime is implied by the optimization.

| Runner | New entropy file | Existing phase file |
| --- | --- | --- |
| GPQA / LiveCodeBench / ICPC-Eval / BBH | `<staging_dir>/.bon_token_entropies.jsonl` | `<staging_dir>/.bon_prepared.jsonl` |
| NaturalPlan | `<output_dir>/.bon_token_entropies.jsonl` | `<output_dir>/.staging/.bon_prepared.jsonl` |
| Three standalone scripts | `<output_dir>/.token_entropies.jsonl` | `<output_dir>/.prepared_tmp.jsonl` |

One entropy JSONL row describes one rollout. The essential fields are:

```json
{
  "schema_version": 1,
  "status": "ok",
  "request_id": "benchmark:sample",
  "sample_idx": 0,
  "sequence_idx": 0,
  "entropy_definition": "full_vocab_next_token_before_response_token_nats",
  "entropy_model": {"model_path": "/models/Qwen3-8B", "dtype": "auto"},
  "token_fingerprint": "sha256 of the exact prompt and response token arrays",
  "prompt_token_ids": [101, 102],
  "response_token_ids": [201, 202, 203],
  "token_entropies": [0.31, 1.72, 0.48]
}
```

At every zero-based position `t`, `token_entropies[t]` describes the full-vocabulary
distribution **before** `response_token_ids[t]`, conditioned on the exact prompt
and preceding response tokens, in natural-log units. Token IDs, lengths, model
identity and fingerprint are checked before reuse. Phase parameters and entropy
chunk size do not enter the cache key. `--no_save_token_entropies` retains its old
effect on phase/final records; it does not disable this new independent checkpoint.

`--resume` still skips completed final evaluations. To compare phase settings
on exactly the same rollouts, use a separate output directory and copy only the
generation and token-entropy intermediates there, then run with `--resume` and
the new phase options. Shared runners need `.bon_generation.jsonl` plus
`.bon_token_entropies.jsonl`; standalone scripts need `.generation_tmp.jsonl`
plus `.token_entropies.jsonl`. For NaturalPlan the generation file remains under
`.staging/`, while its entropy file belongs in the output root. Do not carry over
old phase/final/PRM score files to the new experiment directory. The new cache
continues to validate prompt and response IDs if an overwrite regenerates outputs.
