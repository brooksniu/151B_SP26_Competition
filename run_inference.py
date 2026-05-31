"""
CSE 151B SP26 Math Reasoning Competition — End-to-end inference pipeline.

Loads Qwen3-4B-Thinking-2507 via vLLM, generates k=8 samples per problem,
identifies problems whose samples all failed to produce a final boxed answer,
re-runs those at a higher token budget, and emits the submission CSV via
majority voting across samples.

Usage:
    python run_inference.py

Inputs:
    data/private.jsonl  — one JSON record per line with fields:
        id (int), question (str), options (list[str] or absent)

Outputs:
    submission.csv  — CSV with header [id, response]
"""

import csv
import gc
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
PRIVATE_DATA_PATH = "data/private.jsonl"
OUTPUT_CSV = "submission.csv"

# First-pass generation: standard sampling, k=8
PASS1_MAX_MODEL_LEN = 12288
PASS1_MAX_NUM_SEQS = 32
PASS1_MAX_TOKENS = 8192
PASS1_N_SAMPLES = 8
PASS1_CHUNK_SIZE = 50

# Truncation rerun: large token budget, k=4
RERUN_MAX_MODEL_LEN = 24576
RERUN_MAX_NUM_SEQS = 16
RERUN_MAX_TOKENS = 20480
RERUN_N_SAMPLES = 4
RERUN_CHUNK_SIZE = 10

# Shared sampling parameters
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0

# ── Prompts ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "For each [ANS] placeholder in the question, output exactly one separate "
    "\\boxed{} containing only that answer, in order. Do NOT combine multiple "
    "answers into one \\boxed{}. "
    "Provide numeric answers with at least 6 significant figures of precision "
    "(e.g. write 7.79743547584717, not 7.797; write 7091.666667, not 7091.67). "
    "Only put your FINAL answers in \\boxed{} — never use \\boxed{} for "
    "intermediate steps."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and choices, then choose "
    "the single best option. After reasoning, output ONLY the letter inside "
    "\\boxed{}, e.g. \\boxed{C}. Use \\boxed{} only once, at the very end."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ── Majority voting helpers ────────────────────────────────────────────────
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def final_answer_key(sample: str, is_mcq: bool):
    """Hashable representation of a sample's final answer(s) for voting.

    For MCQ: the letter in the last boxed answer.
    For free-form: a tuple of the last up-to-3 boxed answers.
    """
    boxed = BOXED_RE.findall(sample)
    if not boxed:
        return None
    if is_mcq:
        m = re.search(r"([A-Z])", boxed[-1].upper())
        return m.group(1) if m else None
    return tuple(boxed[-3:])


def pick_voted_sample(samples: list[str], is_mcq: bool) -> str:
    """Pick the sample whose final answer matches the majority across samples."""
    keyed = [(final_answer_key(s, is_mcq), s) for s in samples]
    valid = [(k, s) for k, s in keyed if k is not None]
    if not valid:
        return samples[0]
    winner_key = Counter([k for k, _ in valid]).most_common(1)[0][0]
    for k, s in valid:
        if k == winner_key:
            return s
    return samples[0]


# ── Generation passes ──────────────────────────────────────────────────────
def make_to_prompt(tokenizer):
    def to_prompt(item):
        system, user = build_prompt(item["question"], item.get("options"))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return to_prompt


def generate_chunked(llm, sampling_params, items, to_prompt, chunk_size, label):
    """Generate samples for items in chunks; return {id: {samples, is_mcq}}."""
    records = {}
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_prompts = [to_prompt(item) for item in chunk]
        outputs = llm.generate(chunk_prompts, sampling_params=sampling_params)
        for item, out in zip(chunk, outputs):
            records[item["id"]] = {
                "id": item["id"],
                "is_mcq": bool(item.get("options")),
                "samples": [s.text.strip() for s in out.outputs],
            }
        print(f"  [{label}] {min(i + chunk_size, len(items))}/{len(items)}")
    return records


def free_gpu(llm):
    """Release the vLLM engine and its GPU memory."""
    try:
        del llm
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()


# ── Main pipeline ──────────────────────────────────────────────────────────
def run_inference(
    private_data_path: str = PRIVATE_DATA_PATH,
    output_csv_path: str = OUTPUT_CSV,
) -> None:
    """Run the full inference pipeline end-to-end and write the submission CSV."""
    # Load private set
    private_data = [json.loads(l) for l in open(private_data_path)]
    print(f"Loaded {len(private_data)} private problems")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    to_prompt = make_to_prompt(tokenizer)

    # ── Pass 1: k=8 at standard max_tokens ─────────────────────────────────
    print("\n=== Pass 1: k=8, max_tokens=8192 ===")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=PASS1_MAX_MODEL_LEN,
        trust_remote_code=True,
        max_num_seqs=PASS1_MAX_NUM_SEQS,
    )
    pass1_params = SamplingParams(
        n=PASS1_N_SAMPLES,
        max_tokens=PASS1_MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
    )
    records_by_id = generate_chunked(
        llm, pass1_params, private_data, to_prompt, PASS1_CHUNK_SIZE, "pass1")

    # Identify problems where no sample produced a boxed answer
    truncated_ids = [
        rid for rid, rec in records_by_id.items()
        if not any(BOXED_RE.findall(s) for s in rec["samples"])
    ]
    print(f"\nTruncation rerun needed for {len(truncated_ids)} problems")

    # Release pass-1 engine before loading the rerun engine
    free_gpu(llm)

    # ── Pass 2: targeted rerun at high max_tokens ──────────────────────────
    if truncated_ids:
        print(f"\n=== Pass 2: rerun {len(truncated_ids)} truncated, k=4, max_tokens=20480 ===")
        llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=RERUN_MAX_MODEL_LEN,
            trust_remote_code=True,
            max_num_seqs=RERUN_MAX_NUM_SEQS,
        )
        rerun_params = SamplingParams(
            n=RERUN_N_SAMPLES,
            max_tokens=RERUN_MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )
        private_by_id = {d["id"]: d for d in private_data}
        rerun_items = [private_by_id[rid] for rid in truncated_ids]
        rerun_records = generate_chunked(
            llm, rerun_params, rerun_items, to_prompt, RERUN_CHUNK_SIZE, "rerun")
        records_by_id.update(rerun_records)
        free_gpu(llm)

    # ── Build submission CSV ───────────────────────────────────────────────
    print(f"\n=== Building submission CSV ===")
    private_ids_in_order = [d["id"] for d in private_data]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for rid in private_ids_in_order:
            rec = records_by_id[rid]
            chosen = pick_voted_sample(rec["samples"], rec["is_mcq"])
            writer.writerow([rid, chosen])
    print(f"Submission written to {output_csv_path}")


if __name__ == "__main__":
    run_inference()