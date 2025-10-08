import regex as re


def print_color(text: str, color: str = "red"):
    """Print text in a specified color in the terminal."""
    print(f"[{color}]{text}[/{color}]")


def safe_slug(s: str) -> str:
    # Replace path separators and any weird chars with '-'
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.replace("/", "-").replace("\\", "-"))
