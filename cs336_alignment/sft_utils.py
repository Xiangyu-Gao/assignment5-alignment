import torch


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
