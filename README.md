# CSE 151B SP26 Competition

## Environment

- **GPU**: NVIDIA H100 PCIe 80GB (initial) and NVIDIA RTX PRO 6000 96GB (final)
- **Model**: Qwen/Qwen3-4B-Thinking-2507 (base, no fine-tuning)
- **Total runtime**:
  - First pass (H100 PCIe 80GB) (k=8, max_tokens=8192): ~6-7 hours
  - Truncation rerun (NVIDIA RTX PRO 6000 96GB) (k=4, max_tokens=20480): ~5 hours

## Setup

The model weights are loaded directly from HuggingFace Hub at runtime;
no manual download required.

Dependencies:
- vllm
- transformers
- torch (with CUDA support)

## Running

```bash
pip install vllm transformers torch
python run_inference.py
```

This will:
1. Load `data/private.jsonl` (must exist in the working directory)
2. Load Qwen3-4B-Thinking-2507 via vLLM
3. Generate k=8 samples per problem
4. Identify problems where no sample produced a final boxed answer
5. Re-run those problems with higher max_tokens budget
6. Apply majority voting across samples to pick the final response
7. Write the submission CSV to `submission.csv`
