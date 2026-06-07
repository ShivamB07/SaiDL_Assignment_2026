import torch


def SlidingwindowAttention(T, window_size, device=None):
    if not isinstance(T, int) or T <= 0:
        raise ValueError("T must be a positive integer")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size > T:
        window_size = T

    device = torch.device(device) if device is not None else torch.device("cpu")
    row = torch.arange(T, device=device).unsqueeze(1)
    col = torch.arange(T, device=device).unsqueeze(0)
    dist = row - col
    valid_mask = (dist >= 0) & (dist < window_size)
    attention_mask = torch.full((T, T), float("-inf"), device=device)
    attention_mask.masked_fill_(valid_mask, 0.0)
    return attention_mask
