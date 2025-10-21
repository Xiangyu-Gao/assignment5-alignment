import time
import regex as re


from pathlib import Path


def print_color(text: str, color: str = "red"):
    """Print text in a specified color in the terminal."""
    print(f"[{color}]{text}[/{color}]")


def safe_slug(s: str) -> str:
    # Replace path separators and any weird chars with '-'
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.replace("/", "-").replace("\\", "-"))


def get_run_name(prefix: str, config):
    date = time.strftime("%m%d-%H%M%S")
    return f"{prefix}-{safe_slug(config.model_name)}-{config.num_example}-{config.data_path.split('/')[2]}-{date}"


def print_rich_dict(data: dict) -> None:
    """Pretty print dictionary with colors using rich."""
    from rich.pretty import pprint

    pprint(data, expand_all=True)


def save_model_and_tokenizer(model, tokenizer, config):
    out_dir = Path(f"./{config.experiment_name_base}/{config.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Model and tokenizer saved to {out_dir}")