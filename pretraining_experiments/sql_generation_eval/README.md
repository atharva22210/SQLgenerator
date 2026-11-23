# Held-out SQL generation

Use this after training to inspect whether a final checkpoint can generate a query from the schema, optional additional context, and question. It uses the same Transformer weights as the loss evaluation. During generation, each predicted token becomes the next input. A mistake can therefore affect the rest of the query.

From the repository root, point the launcher at a completed shared-trainer checkpoint on the Modal volume. Replace SUITE and CASE with the names from the sweep. Choose a fresh local output filename for each checkpoint.

```sh
.venv/bin/modal run -m pretraining_experiments.sql_generation_eval.launch \
  --checkpoint /vol/checkpoints/pretraining_experiments/optimizer_batch_lr/SUITE/CASE/latest.pt \
  --output artifacts/CASE-generations.jsonl
```

The default is 60 distinct validation documents, seed 12345, and a maximum of 256 generated tokens per prompt. Set `--count 50` for a smaller panel. One A10 runs the cases sequentially. At the default cap this is at most 15,360 greedy token predictions. The decoder recomputes context at each step. For a quick GPU check, use `--count 1` with the same seed and token cap.

## Which examples are included?

The loader verifies the checkpoint reached its configured final update and checks the published corpus and tokenizer identity. It shuffles EOS-delimited documents from the pinned sql_val.bin and retains examples with the structured schema, question, and SQL markers. Standalone repository SQL, ambiguous field boundaries, and prompts too long to leave the full generation allowance are excluded. Selection fails if too few eligible examples exist. The reference answer's length does not affect eligibility.

Use the same corpus, context window, seed, and token cap across checkpoints. Match outputs by corpus ID and validation document index. This panel measures performance on eligible question-to-SQL examples. The token-loss metric also covers standalone SQL.

## Judge input

Each UTF-8 JSONL line includes the exact prompt, reference_sql, generated_sql, raw_completion, stop_reason, token counts, and checkpoint and sample identifiers. The prompt ends with the opening SQL tag and a newline. Decoding stops at a closing SQL tag, EOS, or the cap. The candidate preserves empty and truncated answers. The raw completion retains any emitted terminator.

For judging, present the prompt, reference_sql, and generated_sql. Keep checkpoint identity hidden. Ask whether the candidate answers the question for the supplied schema, allowing semantically equivalent SQL. Record a verdict of correct, incorrect, or uncertain and a short rationale for each row. Report all three counts and the denominator. This yields a judge assessment of semantic correctness. Database execution would provide a separate measurement.
