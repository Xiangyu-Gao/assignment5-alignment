import torch

from typing import Callable, Literal


def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
    ):
    """ 
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        rewards.append(reward_dict["reward"])
    rewards_tensor = torch.tensor(rewards)  # (rollout_batch_size,)
    
    num_groups = len(rollout_responses) // group_size
    
    rewards_reshaped = rewards_tensor.view(num_groups, group_size)  # (num_groups, group_size)
    
    group_means = torch.mean(rewards_reshaped, dim=1, keepdim=True)  # (num_groups, 1)
    
    if normalize_by_std:
        group_stds = torch.std(rewards_reshaped, dim=1, keepdim=True)  # (num_groups, 1)
        group_stds = group_stds + advantage_eps  # Avoid division by zero
        normalized_rewards = (rewards_reshaped - group_means) / group_stds  # (num_groups, group_size)
    else:
        normalized_rewards = rewards_reshaped - group_means  # (num_groups, group_size)
    advantages = normalized_rewards.view(-1)  # (rollout_batch_size,)
    metadata = {
        "mean_reward": torch.mean(rewards_tensor).item(),
        "std_reward": torch.std(rewards_tensor).item(),
        "max_reward": torch.max(rewards_tensor).item(),
        "min_reward": torch.min(rewards_tensor).item(),
    }
    return advantages, rewards_tensor, metadata


def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)  # (batch_size, sequence_length)
    unclipped_loss = -advantages * ratios  # (batch_size, sequence_length)
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)  # (batch_size, sequence_length)
    clipped_loss = -advantages * clipped_ratios  # (batch_size, sequence_length)
    grpo_clip_loss = torch.max(unclipped_loss, clipped_loss)  # (batch_size, sequence_length)
    clip_fraction = ((ratios > 1.0 + cliprange) | (ratios < 1.0 - cliprange)).float().mean()
    metadata = {
        "clip_fraction": clip_fraction,
    }
    return grpo_clip_loss, metadata


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: str,
        raw_rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ϵ used for clipping.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        )
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        )
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss, metadata


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_tensor = tensor * mask
    if dim is not None:
        sum_masked = torch.sum(masked_tensor, dim=dim)
        count_nonmasked = torch.sum(mask, dim=dim)
    else:
        sum_masked = torch.sum(masked_tensor)
        count_nonmasked = torch.sum(mask)

    mean_masked = sum_masked / count_nonmasked
    
    return mean_masked


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )   # (batch_size, sequence_length)
    masked_loss = masked_mean(loss, response_mask, dim=1)  # (batch_size,)
    mean_loss = torch.mean(masked_loss) / gradient_accumulation_steps  # scalar
    mean_loss.backward()
    return mean_loss, metadata
