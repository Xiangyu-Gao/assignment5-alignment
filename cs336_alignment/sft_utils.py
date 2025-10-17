import torch

from transformers import PreTrainedModel


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. 
        Then the returned dictionary should have the following keys:
            input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
            labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token.
            response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask on the response tokens in the labels.
    """
    # 151643 for Qwen/Qwen2.5-Math-1.5B
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_input_ids = []
    output_input_ids = []

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)
        prompt_input_ids.append(torch.tensor(prompt_ids))
        output_input_ids.append(torch.tensor(output_ids))
    
    # find the max length of prompt + output
    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_input_ids, output_input_ids)]
    max_len = max(prompt_and_output_lens)

    # Concatenate prompt and output input ids, and pad to max length
    concatenated_input_ids = []
    concatenated_labels = []
    response_masks = []

    for p_ids, o_ids in zip(prompt_input_ids, output_input_ids):
        concatenated = torch.cat([p_ids, o_ids])
        pad_length = max_len - len(concatenated)
        padded_input_ids = torch.cat([concatenated, torch.full((pad_length,), pad_token_id)])
        concatenated_input_ids.append(padded_input_ids[:-1])

        # Create labels by shifting input ids to the left
        concatenated_labels.append(padded_input_ids[1:])

        # Create response mask: 0 for prompt tokens, 1 for output tokens, 0 for padding
        response_mask = torch.cat([torch.zeros(len(p_ids)), torch.ones(len(o_ids)), torch.zeros(pad_length)])
        response_masks.append(response_mask[1:])  # Shift mask to align with labels
    
    batch_input_ids = torch.stack(concatenated_input_ids)
    batch_labels = torch.stack(concatenated_labels)
    batch_response_masks = torch.stack(response_masks)

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "response_mask": batch_response_masks
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
        Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    # use a numerically stable method (e.g., using logsumexp) to avoid overflow
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)  # add a small constant to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
        and in inference mode if gradients should not be computed).
        
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
        response tokens as produced by your tokenization method.
        
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
        tokenization method.
        
        return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.
    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ (yt | x<t ).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """
    logits = model(input_ids).logits  # (batch_size, sequence_length, vocab_size)

    log_prob = torch.log_softmax(logits, dim=-1)  # (batch_size, sequence_length, vocab_size)
    label_token_log_softmax = torch.gather(log_prob, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (batch_size, sequence_length)

    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # (batch_size, sequence_length)
        return {
            "log_probs": label_token_log_softmax,
            "token_entropy": token_entropy
        }
    else:
        return {
            "log_probs": label_token_log_softmax
        }


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float, dim: int | None = None) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to the sum.
    """
    masked_tensor = tensor * mask
    summed = torch.sum(masked_tensor, dim=dim)
    normalized = summed / normalize_constant
    return normalized
