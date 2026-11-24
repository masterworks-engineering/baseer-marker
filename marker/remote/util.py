def detect_repeat_loop(
    text: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    scaling_factor: float = 3.0,
) -> bool:
    """True iff the tail of ``text`` ends in a repeating sequence (decoder loop)."""
    if not text:
        return False
    for seq_len in range(1, window_size // 2 + 1):
        candidate = text[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeats = 0
        pos = len(text) - seq_len
        while pos >= 0 and text[pos : pos + seq_len] == candidate:
            repeats += 1
            pos -= seq_len
        if repeats > max_repeats:
            return True
    return False
