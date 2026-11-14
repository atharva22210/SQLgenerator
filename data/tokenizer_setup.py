"""Create the tokenizer used by the first text-to-SQL experiments.

This script is intentionally local-only for now.  The next milestone will copy
the saved tokenizer directory to the shared Modal Volume; keeping that transfer
separate makes it easier to understand what is happening at each stage.

Run from the repository root with::

    uv run --locked python data/tokenizer_setup.py

The first run downloads the base tokenizer from Hugging Face and writes the
result to ``data/tokenizer``.  That generated directory should not be committed
to Git; it is an artifact that we will later place on the Modal Volume.
"""

from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer


# Qwen2.5-0.5B is small enough for experimentation, while its tokenizer is
# already useful for English and code-like text.  We are choosing the
# tokenizer, not loading the model weights here.
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"

# These delimit the model's internal reasoning and its SQL answer.  They are
# special tokens so the tokenizer will not split them into ordinary characters
# and the model can learn their boundaries explicitly.
ADDITIONAL_SPECIAL_TOKENS = [
    "<think>",
    "</think>",
    "<sql>",
    "</sql>",
]

# Padding is not part of the semantic target.  It is needed later when examples
# of different lengths are assembled into a batch.  Qwen has an EOS token, but
# no padding token, so we add a distinct one rather than reusing EOS.
PAD_TOKEN = "<|pad|>"


def build_tokenizer(
    model_name: str = DEFAULT_TOKENIZER,
    output_dir: Path = Path("data/tokenizer"),
) -> dict[str, object]:
    """Download, configure, and save the tokenizer.

    Returning metadata makes this function easy to test without having to
    inspect files manually.  ``added_tokens`` is especially useful: it tells us
    whether a token was genuinely new or was already present in the base
    tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # The Qwen tokenizer does not need executable custom Python code.
        # Keeping this false is a safer default when downloading artifacts.
        trust_remote_code=False,
    )

    vocabulary_before = len(tokenizer)

    # ``additional_special_tokens`` preserves these strings as atomic tokens
    # during encode/decode.  ``add_special_tokens`` is idempotent for repeated
    # runs: it does not keep inventing another token for the same string.
    added_tokens = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
            "pad_token": PAD_TOKEN,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    special_token_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in [*ADDITIONAL_SPECIAL_TOKENS, PAD_TOKEN]
    }
    return {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "vocabulary_before": vocabulary_before,
        "vocabulary_after": len(tokenizer),
        "added_tokens": added_tokens,
        "special_token_ids": special_token_ids,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=Path("data/tokenizer"))
    args = parser.parse_args()

    metadata = build_tokenizer(args.model, args.output)
    print("Tokenizer created successfully:")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
