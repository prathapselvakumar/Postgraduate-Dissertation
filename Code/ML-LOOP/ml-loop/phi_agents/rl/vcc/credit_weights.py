import torch

def compute_credit_weights(
    turn_bookmarks: list[bool],
    turn_token_spans: list[tuple[int, int]],
    alpha: float = 0.1,
    device: torch.device | None = None
) -> torch.Tensor:
    """
    Map per-turn bookmark flags to per-token credit weights.
    
    Args:
        turn_bookmarks: list of bool, True if turn t is a visual bookmark
        turn_token_spans: list of (start, end) token indices for each turn
        alpha: background credit floor for non-bookmarked turns
        device: target device for the output tensor
    
    Returns:
        weights: tensor of shape [n_output_tokens]
    """
    if not turn_token_spans:
        return torch.ones(0, device=device)
        
    total_tokens = max(end for _, end in turn_token_spans)
    weights = torch.full((total_tokens,), alpha, device=device)
    
    for turn_idx, is_bookmark in enumerate(turn_bookmarks):
        if turn_idx < len(turn_token_spans):
            start, end = turn_token_spans[turn_idx]
            if is_bookmark:
                weights[start:end] = 1.0
                
    # Normalise so weights sum to total_tokens (preserve gradient scale)
    total_sum = weights.sum()
    if total_sum > 0:
        weights = weights * (total_tokens / total_sum)
        
    return weights
