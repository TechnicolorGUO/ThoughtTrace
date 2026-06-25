# ThoughtTrace Reproduction — Implementation Spec

**Audience:** Claude Code (implementer). **Backbone model:** `Qwen3-8B` (local).
**Source paper:** Jin et al., *ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions*, arXiv:2605.20087. Method details live in the paper's **Appendix C (data collection)** and **Appendix D (analyses & experiments)** — this spec references those by number; pull exact prompt wording from the paper when a step says "use the Appendix D.x prompt."
**Upstream repo (already forked):** `github.com/thoughttrace-project/thoughttrace` — ships `data/ThoughtTrace.jsonl`, `filter_conversations.ipynb`, `check_dataset_stats.ipynb`, `requirements.txt`. **No analysis/experiment code is released** — everything in Phases 1–4 below must be implemented from scratch.

---

## 0. Scope & strategy

We reproduce the **analyses + the two utility experiments on the released data**. We do **not** reproduce the human data-collection pipeline (IRB/Prolific/Firebase chat UI from Appendix C) — that only matters if collecting fresh data, and is out of scope here.

Two strategic shortcuts that save large amounts of compute and avoid noisy re-labeling — **use them**:

1. **The released dataset already contains gold thought-type labels.** Each `reasons[]` entry has a `label` (one of 7 reason types) and each `reactions[]` entry has a `label` (one of 5 reaction types). So all *distribution* and *stage-dynamics* analyses (Thought Property 3 & 4) are computed **directly from gold labels** — no LLM labeling needed. Re-running a labeler is only for reproducing/validating the *labeling pipeline* (optional, Phase 2C).
2. **Dissatisfaction reactions for the alignment experiment** (Phase 4) are selected directly via gold reaction labels `content_relevance` / `presentation_style` / `scope_fit` — no re-classification.

### Model substitution table (paper → this reproduction)

| Role in paper | Paper used | This reproduction (Qwen3-8B / local) | Impact |
|---|---|---|---|
| Topic / relationship / thought-type labeler | GPT-5.4 | Qwen3-8B (temp 0) | Lower label accuracy → validate against gold labels (Phase 2C) |
| Semantic-coverage scorer (Prop 1) | GPT-5.4 | Qwen3-8B | Absolute scores shift; relative reason-vs-reaction gap should hold |
| Thought-inference predictor (Prop 2) | GPT-5.4 / Gemini 3.1 Pro / Opus 4.6 | Qwen3-8B | Single weaker model → lower scores; conclusion ("hard to infer") still expected |
| Next-message predictor (Utility 1) | same 3 frontier models | Qwen3-8B | **Track the relative delta** (history-only vs +thoughts), not absolute |
| LLM judge (Prop 2, Utility 1) | cross-judged across the 3 frontier models | see §Judge note below | Self-preference risk |
| Sentence embeddings (Prop 1) | `text-embedding-3-large` | `Qwen3-Embedding-0.6B` (or 4B/8B) | Geometry differs; rank order of the 3 pair-types should hold |
| Tokenizer for length stats | tiktoken `gpt-4o` | keep tiktoken `gpt-4o` | unchanged (it's just counting) |
| DPO base model (Utility 2) | Qwen3.5-4B | **Qwen3-8B** | larger base; expect same direction |
| Rewrite generator (Utility 2) | GPT-5.4 | Qwen3-8B (default) — see tradeoff note | weaker "chosen" responses → smaller margin |
| Alignment benchmark judge (Utility 2) | GPT-4o on Arena-Hard | see §Judge note | self-eval bias if Qwen judges Qwen |
| DPO training infra | Tinker API | **TRL `DPOTrainer`** (local) | unchanged objective |

**§Judge note (important).** The paper deliberately avoids self-evaluation by having a *different* model judge each predictor. With a single backbone you lose that. Pick one and **document it in the README**:
- **Preferred:** use a second, architecturally different open judge (e.g. a Llama-3.x-8B-Instruct or Mistral-Small) purely for judging, so predictor ≠ judge.
- **Acceptable fallback:** judge with Qwen3-8B too, but then the absolute numbers are only internally comparable, never vs the paper. Either way, the headline finding is a **within-setup delta** (does adding thoughts help?), which is robust to judge choice.

### What "success" means here

Do **not** target the paper's absolute numbers (Table 1: 21.6→30.6; Table 2 win rates). With an 8B backbone they will not match. Target **directional reproduction**:
- Prop 1: reaction→next-message pairs show larger embedding shift than message→reason, which show larger shift than consecutive messages.
- Prop 2: inferred thoughts score low (≈ "minimal–partial overlap").
- Utility 1: thought-augmented next-message prediction **> history-only**.
- Utility 2: `thought-guided > message-guided > base` on the alignment benchmark.

---

## 1. Repo layout to build

Add this on top of the fork (keep the upstream `data/` and notebooks):

```
thoughttrace/
├── data/
│   ├── ThoughtTrace.jsonl              # upstream (already present)
│   └── baselines/                      # you create: WildChat / LMSYS subsets
├── src/
│   ├── io_utils.py                     # load jsonl → dict, schema helpers, turn indexing
│   ├── llm_client.py                   # OpenAI-compatible client → local vLLM (Qwen3-8B)
│   ├── embed.py                        # Qwen3-Embedding wrapper
│   ├── phase0_stats.py
│   ├── phase1_conversation_props.py
│   ├── phase2_thought_props.py
│   ├── phase3_utility_prediction.py
│   └── phase4_utility_alignment/
│       ├── build_dpo_data.py
│       ├── train_dpo.py
│       └── eval_arenahard.py
├── prompts/                            # one file per prompt template (see §6)
├── configs/
│   └── default.yaml                    # model names, paths, hyperparams, seeds
├── outputs/                            # figures, tables, jsonl artifacts, checkpoints
├── tests/
└── REPRODUCTION.md                     # this file
```

---

## 2. Environment & serving

- Python 3.10+. Core deps: `datasets`, `transformers`, `vllm`, `trl`, `peft`, `accelerate`, `sentence-transformers`, `umap-learn`, `scikit-learn`, `tiktoken`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`.
- **Serve Qwen3-8B once via vLLM** with an OpenAI-compatible endpoint so every LLM-as-X call in Phases 1–4 reuses `llm_client.py`:
  ```bash
  vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768
  ```
- **Qwen3 thinking mode:** for all labeling / scoring / judging / prediction calls, **disable thinking** for deterministic short outputs. Pass `chat_template_kwargs={"enable_thinking": false}` (or append `/no_think`). Use `temperature=0` for any task the paper ran at temp 0 (labeling, judging).
- **Embeddings:** load `Qwen/Qwen3-Embedding-0.6B` via `sentence-transformers`. Make the model name a config knob so 4B/8B can be swapped in.
- `llm_client.py` must implement retry + on-disk response caching keyed by `(prompt, params)` — Phases 2–4 make tens of thousands of calls; caching makes reruns free.

### Baseline datasets (needed for some figures + the WildChat training arm)
- `allenai/WildChat-1M` and the LMSYS-Chat-1M release. Download via `datasets`. For length-distribution figures you only need turn/token counts; sample if full download is too large, but the **WildChat message-guided training arm (Phase 4)** needs ~5k usable conversations to filter down to 1k instances.

---

## 3. Phase 0 — data load & sanity stats

1. `io_utils.load(path) -> dict[id, conversation]` mirroring the upstream loader.
2. Reproduce `check_dataset_stats.ipynb` programmatically and assert the global totals: **1,058 users, 2,155 conversations, 17,058 messages/turns, 10,174 thoughts**, 20 models. Per-model counts should match Appendix Table A1.
3. Helpers: `iter_user_turns(conv)`, `turn_index(msg)` (1-indexed), `get_reason(msg)`, `get_reaction(msg)`, `dissatisfaction_reactions(conv)` (filter labels in {content_relevance, presentation_style, scope_fit}).

**Acceptance:** totals match exactly; a `--quick` mode runs on `ThoughtTrace_examples.jsonl`.

---

## 4. Phase 1 — Conversation properties (Appendix D.1–D.3)

**1A. Demographics (D.1).** Aggregate the per-conversation survey fields (age→brackets 18–24/25–34/.../65+, gender, education, occupation top-8, frequency 1–5, purposes via keyword grouping into Learning/Working/Brainstorming/Research/Coding/Planning/Writing/Translation). Render the 6-panel horizontal bar chart (Fig 2). All from gold metadata — no LLM.

**1B. Length distributions (D.2).** Count turns (max turn index per conv) and tokens (tiktoken `gpt-4o`) per conversation for ThoughtTrace + WildChat + LMSYS. Bucket and overlay (Figs 3a, A1). Verify ThoughtTrace **median 8 turns** vs ~2 for the baselines. Also reproduce the per-role message-length stats (Fig A2: user-prompt median ≈13 tokens, assistant median ≈561).

**1C. Topic labeling (D.2).** Multi-label each conversation against the 36-subtopic taxonomy (grouped into 7 parents). Use Qwen3-8B, temp 0, JSON output, validate labels against the allowed set, drop hallucinated labels. Aggregate into the treemap / 7-way distribution (Fig 3b). Use the **exact taxonomy + labeling instruction from Appendix D.2**; the prompt asks for *all* applicable topics, not a single primary one.

**1D. Multi-turn relationship (D.3).** First user turn = `First request`; for every later user turn, classify the relation to the previous user turn into {extend/deepen, re-attempt/revise, new variation, completely new}. Qwen3-8B, temp 0, single-label, keyword-fallback normalization. Reproduce the distribution (Fig A7: First 25.2%, New 12.5%, Re-attempt 2.9%, Variation 2.3%, Extend 57.0%) and the turn-transition flow (Fig 4). Use the **Appendix D.3 relationship taxonomy/prompt**.

**Acceptance:** the four figures regenerate; Extend is the dominant category (~57%) and strengthens with turn depth.

---

## 5. Phase 2 — Thought properties (Appendix D.4–D.7)

**2A. Thoughts ≠ messages (Prop 1 / D.4 + D.6-embeddings / B.6).**
- Embedding shift: embed three paired sets — (i) user message → next user message, (ii) user message → its reason, (iii) reaction → next user message. Project paired *difference* vectors with UMAP (Fig 5) and compute the three distributional metrics from B.6: **Centroid distance (L2), MMD (RBF kernel), Linear-probe AUC (logistic reg, 5-fold)**. Expected ordering (paper Table A2): consecutive-messages < message→reason < reaction→next-message.
- Semantic coverage: score (1–5 rubric) how much a user message covers (i) its reason and (ii) the prior reaction. Use the **D.4 coverage prompt**. Paper got ≈3.22 (reasons) / ≈2.00 (reactions); expect the **reason > reaction** ordering to persist even if absolute values shift under Qwen3-8B.

**2B. Thoughts hard to infer (Prop 2 / D.5).**
- Reason inference: give context up to & including the target user turn, predict the reason (one sentence, user's voice). Reaction inference: give context up to the assistant turn + the following user message, predict the reaction. Use the **D.5 inference prompts**.
- Score each prediction vs the gold thought text with the **D.5 semantic-similarity judge prompt** (1–5). Apply the §Judge note (predictor ≠ judge if possible). Report mean for reasons and reactions. Expected: both land between "minimal" and "partial" (~2–3).

**2C. Thought-type distributions (Prop 3 / D.6).**
- **Distributions: compute directly from the gold `label` fields** — 7 reason types (Fig 6) and 5 reaction types (Fig 7). No LLM needed.
- *Optional pipeline validation:* re-label a sample with Qwen3-8B using the **D.6 reason/reaction classification prompts**, then report **agreement (accuracy / macro-F1) against the gold labels**. This is the single best signal of how trustworthy your Qwen3-8B labeler is, and it gates how much to trust Phase 1C/1D.

**2D. Stage dynamics (Prop 4 / D.7).** Bin each thought into 4 normalized stages (Early/Mid-Early/Mid-Late/Late), build the stacked-bar + Sankey transition flows for reason types and reaction types (Fig 8). Also the cross-tabs: thought-type × multi-turn-relationship (Fig A8), × topic (Figs A9–A10, expect near-independence), × conversation length (Figs A11–A12). All from gold labels + Phase 1 outputs.

**Acceptance:** Table A2 ordering reproduced; Prop 2 scores low; gold-label distributions match Figs 6–7; (if run) labeler agreement reported.

---

## 6. Phase 3 — Utility 1: thoughts predict user behavior (Appendix D.8)

1. **Candidate selection:** every assistant message followed by a user turn.
2. **Quality filter:** score each thought annotation 1–5 with an LLM judge; **keep only examples whose thoughts score ≥ 4** (the paper restricts to genuinely informative thoughts).
3. **Two contexts per candidate:** `history-only` (raw dialogue) and `thought-augmented` (interleave the user's reasons/reactions at their turns).
4. **Predict** the next user message under each context with Qwen3-8B (use **D.8 prediction prompts**, output a single message).
5. **Score** each prediction's semantic similarity to the actual next message on **0–100** (use the **D.8 similarity judge prompt**). Apply §Judge note.
6. **Report** mean similarity for both conditions. **Headline metric = the delta.** Expected: thought-augmented > history-only (paper: +41.7% relative).

**Acceptance:** thought-augmented mean strictly exceeds history-only on the same filtered set; bootstrap CI on the delta excludes 0.

---

## 7. Phase 4 — Utility 2: thoughts improve alignment (Appendix D.9)

Three training runs, all DPO from **Qwen3-8B**, then evaluate on Arena-Hard.

**7A. Build DPO data (`build_dpo_data.py`).** Keep conversations with **2–20 turns**.
- *Thought-guided (ThoughtTrace):* take user reactions whose gold label ∈ {content_relevance, presentation_style, scope_fit}; drop thoughts that are empty / < 6 words / no alphabetic chars; slice context up to (not including) the dissatisfying assistant turn; generate a revised assistant response with the **D.9 thought-guided rewrite prompt** (rewriter = Qwen3-8B by default — see tradeoff). DPO pair: `prompt`=context, `chosen`=rewrite, `rejected`=original response. **Target 1,000 instances.**
- *Message-guided (ThoughtTrace):* same conversations, but classify (assistant, next-user-msg) as satisfied/dissatisfied with the **D.9 message-guided setup**, rewrite using the user's follow-up message. **Target 450 instances** (intentionally smaller — same convs — to show thoughts surface ~2.2× more dissatisfaction than messages).
- *Message-guided (WildChat):* process WildChat in random order until **1,000** filtered instances (~4.6k convs).

> **Rewriter tradeoff:** the paper used GPT-5.4 to write the `chosen` responses, which makes them strong. With Qwen3-8B as rewriter the `chosen` responses are weaker, shrinking the win-rate margin. If a stronger model is available *only* for the rewrite-generation step, using it improves fidelity without changing the thesis. Document whichever you choose.

**7B. Train (`train_dpo.py`).** TRL `DPOTrainer`, init from Qwen3-8B (LoRA recommended at 8B to fit memory). Hyperparams from the paper: **batch size 64, lr 1e-6, ≤20 epochs with early stopping on a 10% validation split.** Set and log seeds.

**7C. Evaluate (`eval_arenahard.py`).** Run the official Arena-Hard-Auto harness on each checkpoint + the base Qwen3-8B. Report **raw win rate and style-controlled (SC) win rate.** Judge = per §Judge note (do **not** let the eval judge be the same model being graded). Expected ordering: `thought-guided > message-guided(TT) > base`, and `message-guided(TT) > WildChat` at equal/ smaller data.

**Acceptance:** the four-row table reproduces the **ranking**; thought-guided is top; message-guided(TT) beats WildChat.

---

## 8. Prompts (`prompts/`)

Create one template file per task. For faithful reproduction, **copy the exact prompt text from the paper** for each — they're all printed in full in the appendix:
- `topic_labeling.txt`, `topic_taxonomy.txt` — Appendix D.2
- `multiturn_relationship.txt` — Appendix D.3
- `coverage_scoring.txt` — Appendix D.4
- `reason_inference.txt`, `reaction_inference.txt`, `reason_judge.txt`, `reaction_judge.txt` — Appendix D.5
- `reason_classify.txt`, `reaction_classify.txt` — Appendix D.6
- `nextmsg_predict_context.txt`, `nextmsg_predict_thoughts.txt`, `nextmsg_judge.txt` — Appendix D.8
- `rewrite_thought_guided.txt`, `rewrite_message_guided.txt` — Appendix D.9

Each is short, deterministic, and expects a constrained output (a single label, a single integer, or a single message). Wrap with Qwen3 chat template, thinking disabled.

---

## 9. Caveats to record in the README (so results are interpretable)

1. **Single weak backbone replaces 3 frontier models** for every LLM-as-X role → absolute numbers are not comparable to the paper; only within-setup deltas and rankings are.
2. **Self-judging bias** unless a separate judge model is used (see §Judge note).
3. **Local embedding model** changes the embedding geometry; the *ordering* of the three pair-types is the reproducible claim, not the exact distances.
4. **Qwen3-8B rewriter** yields weaker `chosen` responses than the paper's GPT-5.4, shrinking Phase 4 margins.
5. **Stochasticity** from sampling/judging — fix seeds, set temp 0 where the paper did, cache all LLM calls, and report bootstrap CIs on the headline deltas.

## 10. Suggested build order

`Phase 0 → 2C (gold-label distributions + labeler validation) → 1 → 2A/2B/2D → 3 → 4`.
Rationale: Phase 0 and the gold-label distributions are cheap and confirm the data is loaded correctly; the labeler-vs-gold agreement in 2C tells you up front how much to trust the LLM-labeled figures in Phase 1; the two utility experiments (3, 4) are the most compute-heavy, so do them last once the cheaper pieces validate the setup.
