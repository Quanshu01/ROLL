import inspect

import enum
import traceback
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.platforms import current_platform
from roll.utils.kl_controller import AdaptiveKLController
from roll.utils.logging import get_logger


logger = get_logger()


def tensor_to_cpu_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cpu:
            obj.data = obj.data.detach().cpu()
        return True
    return False


def tensor_to_cuda_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            obj.data = obj.data.detach().to(device=torch.device(current_platform.device_type))
        return True
    return False


def delete_tensor_grad_visitor(obj, path):
    if torch.is_tensor(obj):
        obj.grad = None
        return True
    return False


def traverse_obj(value, visitor, path=()):
    """
    遍历对象的所有属性，包括属性的属性，找到所有的 Tensor。
    :param value: 任意 Python 对象
    :visitor
    :path
    """
    if visitor(value, path):
        return
    elif isinstance(value, dict):
        for key, value in value.items():
            traverse_obj(value, visitor, path + (str(key),))
    elif isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            traverse_obj(item, visitor, path + (index,))
    elif hasattr(value, "__dict__"):
        for attr_name in dir(value):
            if not attr_name.startswith("__"):
                try:
                    attr_value = getattr(value, attr_name)
                    traverse_obj(attr_value, visitor, path + (f"attr:{attr_name}",))
                except Exception as e:
                    logger.error(e)
                    continue


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def divide_by_chunk_size(
    data: Union[np.ndarray, TensorDict], chunk_sizes: List[int]
) -> List[Union[np.ndarray, TensorDict]]:
    """
    将numpy数组按照chunks的大小切分
    """
    if not isinstance(data, (np.ndarray, TensorDict)):
        raise TypeError("Input 'array' must be a numpy ndarray or a TensorDict.")

    if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
        raise ValueError("All chunk sizes must be positive integers.")

    total_size = sum(chunk_sizes)
    if total_size != len(data):
        raise ValueError(f"The sum of chunk_sizes ({total_size}) does not match the size of the array ({len(data)}).")

    split_data = []
    start_index = 0
    for size in chunk_sizes:
        end_index = start_index + size
        split_data.append(data[start_index:end_index])
        start_index = end_index
    return split_data


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


def compute_clip_fraction(values: torch.Tensor, clip_max: float, clip_min: float):
    numel = values.numel()
    num_clipped = (values > clip_max).sum().item() + (values < clip_min).sum().item()
    clipfrac = num_clipped / numel if numel > 0 else 0.0
    return clipfrac


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_penalty: str = "kl",
) -> torch.Tensor:
    """
    ref: https://github.com/OpenRLHF/OpenRLHF/blob/494850f50342ed38d5ae76ef45a3207f3523b582/openrlhf/models/utils.py#L7
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty == "kl":
        log_ratio = log_probs - log_probs_base
    elif kl_penalty == "abs":
        log_ratio = (log_probs - log_probs_base).abs()
    elif kl_penalty == "mse":
        log_ratio = 0.5 * (log_probs - log_probs_base).square()
    elif kl_penalty == "k3":
        kl = log_probs_base - log_probs
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        log_ratio = torch.clamp(kld, min=-10, max=10)
    elif kl_penalty == "full":
        log_ratio = F.kl_div(log_probs_base, log_probs, log_target=True, reduction="none").sum(-1)
    else:
        raise NotImplementedError

    if action_mask is not None:
        return log_ratio * action_mask

    return log_ratio


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute log-probs of given labels from logits.

    Note: some upstream code paths may accidentally cast `labels` to float;
    `gather` requires integer indices, so we defensively coerce to long here.
    """
    logits = logits.float()
    if labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        labels = labels.long()
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    logits = logits.float()
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str,
             weights: Optional[torch.Tensor] = None, loss_scale: Optional[float] = None):
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/trainer/ppo/core_algos.py#L370
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "seq-mean-token-sum" is the default behavior
        weights: `torch.Tensor`
        loss_scale: `(float)`
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = masked_mean(loss_mat * weights.unsqueeze(-1), loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = masked_sum(loss_mat, loss_mask, dim=-1) # token-sum
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (valid_samples.sum() + 1e-8) # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1)
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (valid_samples.sum() + 1e-8)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = masked_sum(loss_mat, loss_mask, dim=-1)
        valid_samples = torch.any(loss_mask > 0, dim=-1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss * loss_scale if loss_scale else loss


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        mask_sum = mask.sum(axis=dim)
        return torch.where(mask_sum > 0, (tensor * mask).sum(axis=dim) / (mask_sum + 1e-8), torch.zeros_like(mask_sum))
    else:
        return (tensor * mask).sum() / (mask.sum() + 1e-8)

def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        mask_sum = mask.sum(axis=dim)
        return torch.where(mask_sum > 0, (tensor * mask).sum(axis=dim), torch.zeros_like(mask_sum))
    else:
        return (tensor * mask).sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    """
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def get_pad_mask(response_id: torch.Tensor, pad_token: int = 0, dtype=torch.int64):
    """
    e.g. pad token=0
    response_id: [1, 2, 2, 42, 3, 5, 1, 0, 0]
    pad_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    pad_mask = response_id.not_equal(pad_token).to(dtype)
    assert (
        not (pad_mask[:, 0] == 0).logical_and(pad_mask.sum(-1) != 0).any()
    ), f"response_id is not valid: {response_id}, pad_token is {pad_token}"
    return pad_mask


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim).unsqueeze(-1)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim).unsqueeze(-1)
    return mean_centered * var.clamp(min=eps).rsqrt()


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def response_level_masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    # 考虑response的影响？
    mean = masked_mean(values, mask, dim=-1)
    var = masked_var(mean, mask)
    mean = mean.mean()
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def reduce_metrics(metrics: dict, reduce_func=np.mean) -> dict:
    for key, val in metrics.items():
        metrics[key] = reduce_func(val)
    return metrics


def pad_to_length(tensor: torch.Tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(0, length)
        return tensor[indices]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
        )


def concatenate_input_and_output(input_ids, output_ids, num_return_sequences):
    batch_size, input_seq_len = input_ids.size()
    _, output_seq_len = output_ids.size()
    repeated_input_ids = (
        input_ids.unsqueeze(1)
        .repeat(1, num_return_sequences, 1)
        .view(batch_size * num_return_sequences, input_seq_len)
    )
    sequences = torch.cat((repeated_input_ids, output_ids), dim=1)
    return sequences


def compute_reinforce_return(token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor):
    with torch.no_grad():
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        cumulative_reward = 0
        for t in reversed(range(gen_len)):
            local_reward = token_level_rewards[:, t] if t < gen_len else 0.0
            cumulative_reward = local_reward + gamma * cumulative_reward
            advantages_reversed.append(cumulative_reward)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages
    return advantages, returns


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor, values: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        gamma: `(float)`
            discounted factor used in RL
        lambd: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values

    return advantages, returns


def compute_step_level_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: torch.Tensor,
    lambd: torch.Tensor,
    step_response_lengths: List[int],
    logger=None,
):
    """
    计算 step-level 的 GAE advantage 和 returns。
    
    对于每个 step：
    1. 先计算后续 step 的 cost 通过 TD-error 传播到当前 step 的值
    2. 将这个值加到当前 step 的 cost 上，放在该 step 的最后一个 token 上
    3. 在该 step 内部，用 TD-error 传播回该 step 的每个 token 计算 GAE
    
    Args:
        token_level_rewards: `(torch.Tensor)` shape: (bs, response_length)
            每个 step 的 cost 已经放在对应 step 的最后一个 token 上
        values: `(torch.Tensor)` shape: (bs, response_length)
        gamma: `(float)` discounted factor
        lambd: `(float)` lambda value for GAE
        step_response_lengths: `(List[int])` 每个 step 的 response token 长度
        logger: logger instance for debugging
    
    Returns:
        advantages: `(torch.Tensor)` shape: (bs, response_length)
        returns: `(torch.Tensor)` shape: (bs, response_length)
    """
    from roll.utils.logging import get_logger
    if logger is None:
        logger = get_logger()
    
    with torch.no_grad():
        # 复制 token_level_rewards，避免修改原始数据
        token_level_rewards = token_level_rewards.clone()
        
        batch_size = token_level_rewards.shape[0]
        seq_len = token_level_rewards.shape[1]
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        # 第一步：计算每个 step 的"未来 cost 的 TD-error 传播值"
        # 从后往前遍历每个 step，计算后续 step 的 cost 通过 TD-error 传播到当前 step 的值
        num_steps = len(step_response_lengths)
        step_boundaries = []  # 每个 step 的 [start_idx, end_idx]
        current_idx = 0
        for step_len in step_response_lengths:
            if step_len > 0:
                start_idx = current_idx
                end_idx = current_idx + step_len - 1
                # 确保索引在有效范围内
                if end_idx < seq_len:
                    step_boundaries.append((start_idx, end_idx))
                else:
                    # 超出范围，截断到有效范围
                    step_boundaries.append((start_idx, seq_len - 1))
                current_idx += step_len
            else:
                step_boundaries.append((current_idx, current_idx - 1))  # 空 step
        
        # 计算每个 step 的最后一个 token 位置上的"未来 cost 传播值"
        # 从最后一个 step 开始，向前传播
        future_cost_values = torch.zeros(batch_size, num_steps, device=token_level_rewards.device)
        
        for step_idx in reversed(range(num_steps)):
            start_idx, end_idx = step_boundaries[step_idx]
            if end_idx < start_idx:  # 空 step
                continue
            
            # 当前 step 的最后一个 token 位置
            step_last_token_idx = end_idx
            
            # 当前 step 的 cost（已经在 token_level_rewards 中）
            current_step_cost = token_level_rewards[:, step_last_token_idx].clone()
            
            # 计算后续 step 的 cost 通过 TD-error 传播到当前 step 的值
            if step_idx < num_steps - 1:
                # 下一个 step 的最后一个 token 位置
                next_start_idx, next_end_idx = step_boundaries[step_idx + 1]
                if next_end_idx >= next_start_idx:
                    next_step_last_token_idx = next_end_idx
                    # 下一个 step 的 cost + 下一个 step 的未来传播值
                    next_step_cost = token_level_rewards[:, next_step_last_token_idx]
                    next_step_future_value = future_cost_values[:, step_idx + 1]
                    # TD-error 传播：gamma * (next_cost + next_future_value)
                    future_cost_values[:, step_idx] = gamma * (next_step_cost + next_step_future_value)
            
            # 将未来 cost 传播值加到当前 step 的 cost 上
            # 更新 token_level_rewards 中该 step 最后一个 token 的值
            token_level_rewards[:, step_last_token_idx] = current_step_cost + future_cost_values[:, step_idx]
            
            if logger:
                logger.debug(
                    f"[compute_step_level_gae] Step {step_idx}: "
                    f"current_cost={current_step_cost.mean().item():.4f}, "
                    f"future_value={future_cost_values[:, step_idx].mean().item():.4f}, "
                    f"total_cost={token_level_rewards[:, step_last_token_idx].mean().item():.4f}"
                )
        
        # 第二步：在每个 step 内部计算 GAE
        for step_idx in range(num_steps):
            start_idx, end_idx = step_boundaries[step_idx]
            if end_idx < start_idx:  # 空 step
                continue
            
            step_len = end_idx - start_idx + 1
            if step_len <= 0:
                continue
            
            # 提取该 step 的 rewards 和 values
            step_rewards = token_level_rewards[:, start_idx:end_idx + 1]
            step_values = values[:, start_idx:end_idx + 1]
            
            # 在该 step 内部计算 GAE（从后往前）
            lastgaelam = 0
            step_advantages_reversed = []
            
            for t_local in reversed(range(step_len)):
                # 如果是 step 的最后一个 token，nextvalues 是 0（因为未来 cost 已经加到当前 cost 上了）
                # 否则，nextvalues 是下一个 token 的 value
                if t_local < step_len - 1:
                    nextvalues = step_values[:, t_local + 1]
                else:
                    # step 的最后一个 token，未来 cost 已经加到 reward 上了，所以 nextvalues = 0
                    nextvalues = torch.zeros_like(step_values[:, t_local])
                
                delta = step_rewards[:, t_local] + gamma * nextvalues - step_values[:, t_local]
                lastgaelam = delta + gamma * lambd * lastgaelam
                step_advantages_reversed.append(lastgaelam)
            
            step_advantages = torch.stack(step_advantages_reversed[::-1], dim=1)
            step_returns = step_advantages + step_values
            
            # 将结果写回
            advantages[:, start_idx:end_idx + 1] = step_advantages
            returns[:, start_idx:end_idx + 1] = step_returns
            
            if logger:
                logger.debug(
                    f"[compute_step_level_gae] Step {step_idx} GAE完成: "
                    f"start={start_idx}, end={end_idx}, "
                    f"adv_mean={step_advantages.mean().item():.4f}, "
                    f"ret_mean={step_returns.mean().item():.4f}"
                )
    
    return advantages, returns


def expand_to_token_level(data: "DataProto"):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    batch_size = data.batch.batch_size[0]
    # expand as token_level_rewards
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]
    if position_ids.dim() == 3:
        # qwen2vl, (bsz, 3, seqlen), 0/1/2 is same for text, while values of
        # position_ids for text cannot stand for index of tokens, thus use the
        # right padding attention_mask to calculate eos index or `argmax` rather
        # than `max` of position_ids to calculate eos index
        position_ids = position_ids[:, 0]
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    token_level_rewards = torch.zeros_like(attention_mask, dtype=response_level_rewards.dtype)  # (bsz, seqlen)

    token_level_rewards[torch.arange(batch_size), eos_mask_idx] = response_level_rewards

    try:
        logger.info(
            f"expand_to_token_level: batch={batch_size}, response_level_mean={response_level_rewards.mean().item():.6f}, "
            f"min={response_level_rewards.min().item():.6f}, max={response_level_rewards.max().item():.6f}, eos_idx_sample={eos_mask_idx[:4].tolist()}"
        )
    except Exception:
        # defensive: avoid logging failures
        pass

    # select the response part
    token_level_rewards = token_level_rewards[:, 1:]

    return token_level_rewards


def reward_norm(response_level_rewards: torch.Tensor, n_sample=-1, running_ctrl={}, norm_mean_type=None, norm_std_type=None):
    group_mode = (norm_mean_type == "group") or (norm_std_type == "group")
    if group_mode and n_sample > 0:
        reshape_reward = response_level_rewards.reshape(*response_level_rewards.size()[:-1], -1, n_sample)
    if norm_mean_type == "running" or norm_std_type == "running":
        running = running_ctrl["domain"]
        running.update(response_level_rewards)
    # 均值计算
    if norm_mean_type == "batch":
        reward_mean = response_level_rewards.mean()
    elif norm_mean_type == "group":
        reward_mean = reshape_reward.mean(dim=-1, keepdim=True)
    elif norm_mean_type == "running":
        reward_mean = running.mean
    elif norm_mean_type == None:
        reward_mean = 0.0
    # 标准差计算
    if norm_std_type == "batch":
        reward_std = response_level_rewards.std()
    elif norm_std_type == "group":
        reward_std = torch.std(reshape_reward, dim=-1, keepdim=True)
    elif norm_std_type == "running":
        reward_std = running.std
    # 选择基础奖励值
    rewards = reshape_reward if norm_mean_type == "group" else response_level_rewards
    # 标准化奖励
    if norm_std_type is not None:
        normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-6) 
    else: 
        normalized_rewards = (rewards - reward_mean)
    
    # 如果是对 group mean 归一化，需要恢复原始形状
    if norm_mean_type == "group":
        normalized_rewards = normalized_rewards.reshape(*response_level_rewards.size())
    return normalized_rewards


def difficulty_mask(data: "DataProto", n_sample=-1, low_threshold=0.1, high_threshold=0.95):
    if n_sample > 1:
        scores = data.batch["scores"].clone().detach()
        reshape_score = scores.reshape(*scores.size()[:-1], -1, n_sample)
        reshape_score_mean = reshape_score.mean(dim=-1, keepdim=True).expand_as(reshape_score).reshape(*scores.size())
        data.batch["difficulty_mask"] = (reshape_score_mean > low_threshold) * (reshape_score_mean < high_threshold)
    else:
        data.batch["difficulty_mask"] = torch.ones_like(data.batch["scores"])
    return data


@torch.no_grad()
def compute_token_reward(data: "DataProto", pipeline_config: RLVRConfig, kl_ctrl: AdaptiveKLController):
    token_level_rewards = expand_to_token_level(data)
    beta = 0
    kld = compute_approx_kl(
        log_probs=data.batch["old_log_probs"],
        log_probs_base=data.batch["ref_log_probs"],
        action_mask=data.batch["response_mask"][:, 1:],
        kl_penalty=pipeline_config.kl_penalty,
    )
    # 是否添加token level kl
    if pipeline_config.add_token_level_kl and "ref_log_probs" in data.batch.keys():
        beta = kl_ctrl.value
        token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=data.batch["response_mask"][:, 1:], dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=data.batch.batch_size[0])
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")
    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=token_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        metrics["critic/token_reward_clip_frac"] = reward_clip_frac
        token_level_rewards = torch.clamp(
            token_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    try:
        logger.info(
            f"compute_token_reward: token_reward_mean={token_level_rewards.mean().item():.6f}, "
            f"min={token_level_rewards.min().item():.6f}, max={token_level_rewards.max().item():.6f}, current_kl={current_kl:.6f}, beta={beta:.6f}"
        )
    except Exception:
        pass

    data.batch["token_level_rewards"] = token_level_rewards
    return data, metrics


@torch.no_grad()
def reward_postprocess(data: "DataProto", pipeline_config: RLVRConfig, running_ctrl):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    response_level_metrics = {"critic/reward_clip_frac": 0.0}
    # # 对reward进行处理: 可以灵活定义不同的normalization方法
    # if pipeline_config.adv_estimator == "grpo":
    #     pipeline_config.norm_mean_type, pipeline_config.norm_std_type = "group", "group"

    # 先暂时关闭 GRPO 的 group 归一化，直接用原始 reward 训练
    # if pipeline_config.adv_estimator == "grpo":
    #     pipeline_config.norm_mean_type, pipeline_config.norm_std_type = "group", "group"

    # 暂时设置为：不归一化
    pipeline_config.norm_mean_type, pipeline_config.norm_std_type = None, None


    response_level_rewards = reward_norm(
                    response_level_rewards, 
                    n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
                    running_ctrl=running_ctrl,
                    norm_mean_type=pipeline_config.norm_mean_type,
                    norm_std_type=pipeline_config.norm_std_type
                    )

    # 对reward进行clip
    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=response_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        response_level_rewards = torch.clamp(
            response_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

        response_level_metrics = {"critic/reward_clip_frac": reward_clip_frac}

    try:
        logger.info(
            f"reward_postprocess: resp_level_mean={response_level_rewards.mean().item():.6f}, "
            f"min={response_level_rewards.min().item():.6f}, max={response_level_rewards.max().item():.6f}, "
            f"clip_frac={response_level_metrics.get('critic/reward_clip_frac', 0.0):.6f}"
        )
    except Exception:
        pass

    data.batch["response_level_rewards"] = response_level_rewards
    return data, response_level_metrics


@torch.no_grad()
def get_sample_level_mask(data: "DataProto", pipeline_config: RLVRConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # mask相关策略
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()
    true_response_length = response_mask.sum(-1).float()
    max_response_length = data.batch["responses"].shape[-1]

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    # 1. max_len_mask: 过滤掉超过最大长度的样本
    if pipeline_config.max_len_mask:
        max_len_mask = (max_response_length != true_response_length).float()
        final_sample_mask = final_sample_mask * max_len_mask
        mask_metrics["actor/max_len_mask_ratio"] = max_len_mask.mean().item()
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    # 2. difficulty_mask: 基于难度的过滤
    if pipeline_config.difficulty_mask:
        data = difficulty_mask(
            data,
            n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
            low_threshold=pipeline_config.difficulty_low_threshold,
            high_threshold=pipeline_config.difficulty_high_threshold,
        )
        if "difficulty_mask" in data.batch:
            difficulty_mask_tensor = data.batch["difficulty_mask"].float()
            final_sample_mask = final_sample_mask * difficulty_mask_tensor
            mask_metrics["actor/difficulty_mask_ratio"] = difficulty_mask_tensor.mean().item()
        else:
            mask_metrics["actor/difficulty_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/difficulty_mask_ratio"] = 1.0

    # 3. error_max_len_clip: 基于错误和长度的过滤
    if pipeline_config.error_max_len_clip:
        scores = data.batch["scores"]
        error_len_mask = ((scores == 0) & (true_response_length < pipeline_config.error_max_len_threshold)) | (
            scores == 1
        )
        error_len_mask = error_len_mask.float()
        final_sample_mask = final_sample_mask * error_len_mask
        mask_metrics["actor/error_len_mask_ratio"] = error_len_mask.mean().item()
    else:
        mask_metrics["actor/error_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics


@torch.no_grad()
def apply_kl_penalty(data: "DataProto", kl_ctrl: AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"][:, 1:]

    token_level_rewards = expand_to_token_level(data)
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")

    batch_size = data.batch.batch_size[0]

    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )  # (batch_size, seq_len-1)
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    return data, metrics


@torch.no_grad()
def compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
    cost_mode=False,  # 新增：是否为cost模式
):
    """
    Compute advantages and returns for either reward or cost.
    
    Args:
        cost_mode: If True, compute cost advantages/returns using cost-related keys.
                   If False (default), compute reward advantages/returns.
    """
    import time
    start_time = time.time()
    mode_str = "COST" if cost_mode else "REWARD"
    logger.info(f"[DEBUG] [{mode_str}] compute_advantage 开始执行...")
    
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]
    
    batch_size = response_mask.shape[0]
    seq_len = response_mask.shape[1]
    valid_tokens = int(response_mask.sum().item())
    logger.info(f"[DEBUG] [{mode_str}] 数据形状: batch_size={batch_size}, seq_len={seq_len}, valid_tokens={valid_tokens}")
    
    if response_mask.sum() == 0:
        whiten_rewards = False
        whiten_advantages = False
        logger.info("Warning: domain final_response_mask.sum() == 0! All masked_whiten will be skipped.")

    # 根据cost_mode选择不同的key
    if cost_mode:
        # Cost模式：使用cost相关的key
        token_level_key = "token_level_costs"  # 需要从episode_costs扩展得到
        values_key = "cost_values"
        advantages_key = "cost_advantages"
        returns_key = "cost_returns"
        raw_advantages_key = "raw_cost_advantages"
        
        # 如果没有token_level_costs，需要从step_costs扩展（改进：使用step-level cost而非episode-level cost）
        if token_level_key not in data.batch:
            logger.info(f"[DEBUG] [{mode_str}] 开始从step_costs扩展为token_level_costs（step-level分配）...")
            step_start = time.time()
            # 优先使用step_costs（每个step的cost），而不是episode_costs（累加的cost）
            if "step_costs" in data.non_tensor_batch and "step_response_lengths" in data.non_tensor_batch:
                # 从step_costs和step_response_lengths构造token_level_costs
                step_costs_raw = data.non_tensor_batch["step_costs"]
                step_response_lengths_raw = data.non_tensor_batch["step_response_lengths"]
                
                # 处理step_costs：转换为列表
                # step_costs_raw是np.array([costs], dtype=object)，其中costs是列表[cost1, cost2, ...]
                if isinstance(step_costs_raw, np.ndarray) and step_costs_raw.dtype == object:
                    if len(step_costs_raw) > 0:
                        first_item = step_costs_raw[0]  # 取第一个batch的step_costs列表
                        if isinstance(first_item, (list, tuple, np.ndarray)):
                            step_costs = [float(x) for x in (first_item.tolist() if isinstance(first_item, np.ndarray) else list(first_item))]
                        else:
                            step_costs = [float(first_item)]
                    else:
                        step_costs = []
                else:
                    step_costs = [float(x) for x in np.array(step_costs_raw).flatten()] if len(step_costs_raw) > 0 else []
                
                # 处理step_response_lengths：转换为列表
                # step_response_lengths_raw是np.array([lengths], dtype=object)，其中lengths是列表[len1, len2, ...]
                if isinstance(step_response_lengths_raw, np.ndarray) and step_response_lengths_raw.dtype == object:
                    if len(step_response_lengths_raw) > 0:
                        first_item = step_response_lengths_raw[0]  # 取第一个batch的step_response_lengths列表
                        if isinstance(first_item, (list, tuple, np.ndarray)):
                            step_response_lengths = [int(x) for x in (first_item.tolist() if isinstance(first_item, np.ndarray) else list(first_item))]
                        else:
                            step_response_lengths = [int(first_item)]
                    else:
                        step_response_lengths = []
                else:
                    step_response_lengths = [int(x) for x in np.array(step_response_lengths_raw).flatten()] if len(step_response_lengths_raw) > 0 else []
                
                logger.info(f"[DEBUG] [{mode_str}] step_costs: {step_costs}, step_response_lengths: {step_response_lengths}")
                
                # 验证step_costs和step_response_lengths长度一致
                if len(step_costs) != len(step_response_lengths):
                    logger.warning(
                        f"[DEBUG] [{mode_str}] step_costs长度({len(step_costs)})与step_response_lengths长度({len(step_response_lengths)})不一致，"
                        f"使用fallback方案（episode_costs）"
                    )
                    step_costs = []
                    step_response_lengths = []
                
                # 构造token_level_costs：将每个step的cost放在对应step的最后一个token上
                batch_size = response_mask.shape[0]
                token_level_costs = torch.zeros_like(response_mask, dtype=torch.float32)
                
                if len(step_costs) > 0 and len(step_response_lengths) > 0:
                    # 计算每个step的最后一个token位置（在response_mask中的索引）
                    current_idx = 0
                    for step_idx, (step_cost, step_len) in enumerate(zip(step_costs, step_response_lengths)):
                        if step_len > 0:
                            # 当前step的最后一个token位置（在response中的索引，从0开始）
                            step_last_token_idx = current_idx + step_len - 1
                            # 确保索引在有效范围内
                            if step_last_token_idx < response_mask.shape[1]:
                                # 将step_cost放在该step的最后一个token上（对所有batch样本）
                                token_level_costs[:, step_last_token_idx] = float(step_cost)
                                logger.debug(
                                    f"[DEBUG] [{mode_str}] Step {step_idx}: cost={step_cost:.4f} "
                                    f"放在token位置 {step_last_token_idx} (step_len={step_len})"
                                )
                            else:
                                logger.warning(
                                    f"[DEBUG] [{mode_str}] Step {step_idx}的最后一个token位置 {step_last_token_idx} "
                                    f"超出response_mask长度 {response_mask.shape[1]}"
                                )
                        current_idx += step_len
                    
                    logger.info(
                        f"[DEBUG] [{mode_str}] token_level_costs构造完成（将使用step-level GAE）: "
                        f"shape={token_level_costs.shape}, mean={token_level_costs.mean().item():.6f}, "
                        f"sum={token_level_costs.sum().item():.6f}, 非零位置数={(token_level_costs != 0).sum().item()}"
                    )
                else:
                    # Fallback: 如果step_costs不可用，使用episode_costs（向后兼容）
                    logger.warning(f"[DEBUG] [{mode_str}] step_costs不可用，使用fallback方案（episode_costs）")
                    if "episode_costs" in data.non_tensor_batch:
                        episode_costs_raw = data.non_tensor_batch["episode_costs"]
                        if isinstance(episode_costs_raw, np.ndarray) and episode_costs_raw.dtype == object:
                            episode_costs_list = []
                            for item in episode_costs_raw:
                                if isinstance(item, (list, tuple)):
                                    episode_costs_list.append(float(item[0]) if len(item) > 0 else 0.0)
                                elif isinstance(item, np.ndarray):
                                    episode_costs_list.append(float(item.item()) if item.size == 1 else float(item[0]))
                                elif isinstance(item, (int, float, np.number)):
                                    episode_costs_list.append(float(item))
                                else:
                                    episode_costs_list.append(float(item))
                            episode_costs = torch.tensor(episode_costs_list, dtype=torch.float32, device=response_mask.device)
                        else:
                            try:
                                episode_costs = torch.tensor(episode_costs_raw, dtype=torch.float32, device=response_mask.device)
                            except (TypeError, ValueError):
                                episode_costs_list = [float(x) for x in np.array(episode_costs_raw).flatten()]
                                episode_costs = torch.tensor(episode_costs_list, dtype=torch.float32, device=response_mask.device)
                        
                        # Fallback: 将cost放在最后一个非padding位置
                        last_valid_idx = (response_mask.sum(dim=1) - 1).long().clamp(min=0)
                        token_level_costs[torch.arange(batch_size), last_valid_idx] = episode_costs.squeeze(-1) if episode_costs.dim() > 1 else episode_costs
                    else:
                        raise ValueError(
                            "cost_mode=True requires 'step_costs' and 'step_response_lengths' in data.non_tensor_batch "
                            "or 'episode_costs' in data.non_tensor_batch "
                            "or 'token_level_costs' in data.batch"
                        )
                
                data.batch[token_level_key] = token_level_costs
                logger.info(f"[DEBUG] [{mode_str}] step_costs扩展总耗时: {time.time() - step_start:.3f}秒")
            elif "episode_costs" in data.non_tensor_batch:
                # Fallback: 如果step_costs不可用，使用旧的episode_costs方式（向后兼容）
                logger.warning(f"[DEBUG] [{mode_str}] step_costs或step_response_lengths不可用，使用fallback方案（episode_costs）")
                episode_costs_raw = data.non_tensor_batch["episode_costs"]
                if isinstance(episode_costs_raw, np.ndarray) and episode_costs_raw.dtype == object:
                    episode_costs_list = []
                    for item in episode_costs_raw:
                        if isinstance(item, (list, tuple)):
                            episode_costs_list.append(float(item[0]) if len(item) > 0 else 0.0)
                        elif isinstance(item, np.ndarray):
                            episode_costs_list.append(float(item.item()) if item.size == 1 else float(item[0]))
                        elif isinstance(item, (int, float, np.number)):
                            episode_costs_list.append(float(item))
                        else:
                            episode_costs_list.append(float(item))
                    episode_costs = torch.tensor(episode_costs_list, dtype=torch.float32, device=response_mask.device)
                else:
                    try:
                        episode_costs = torch.tensor(episode_costs_raw, dtype=torch.float32, device=response_mask.device)
                    except (TypeError, ValueError):
                        episode_costs_list = [float(x) for x in np.array(episode_costs_raw).flatten()]
                        episode_costs = torch.tensor(episode_costs_list, dtype=torch.float32, device=response_mask.device)
                
                batch_size = response_mask.shape[0]
                token_level_costs = torch.zeros_like(response_mask, dtype=torch.float32)
                last_valid_idx = (response_mask.sum(dim=1) - 1).long().clamp(min=0)
                token_level_costs[torch.arange(batch_size), last_valid_idx] = episode_costs.squeeze(-1) if episode_costs.dim() > 1 else episode_costs
                data.batch[token_level_key] = token_level_costs
            else:
                raise ValueError(
                    "cost_mode=True requires 'step_costs' and 'step_response_lengths' in data.non_tensor_batch "
                    "or 'episode_costs' in data.non_tensor_batch "
                    "or 'token_level_costs' in data.batch"
                )
    else:
        # Reward模式（原有逻辑）
        token_level_key = "token_level_rewards"
        values_key = "values"
        advantages_key = "advantages"
        returns_key = "returns"
        raw_advantages_key = "raw_advantages"

    logger.info(f"[DEBUG] [{mode_str}] 开始处理token_level数据...")
    process_start = time.time()
    token_level_rewards = data.batch[token_level_key].float()
    logger.info(f"[DEBUG] [{mode_str}] token_level_rewards原始: shape={token_level_rewards.shape}, mean={token_level_rewards.mean().item():.6f}, sum={token_level_rewards.sum().item():.6f}")
    
    if whiten_rewards:
        logger.info(f"[DEBUG] [{mode_str}] 执行whiten_rewards...")
        whiten_start = time.time()
        token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
        logger.info(f"[DEBUG] [{mode_str}] whiten_rewards完成，耗时: {time.time() - whiten_start:.3f}秒")
    
    token_level_rewards = token_level_rewards * response_mask
    data.batch[token_level_key] = token_level_rewards
    logger.info(f"[DEBUG] [{mode_str}] token_level_rewards处理后: mean={token_level_rewards.mean().item():.6f}, sum={token_level_rewards.sum().item():.6f}")
    
    if adv_estimator == "gae":
        logger.info(f"[DEBUG] [{mode_str}] 使用GAE估计器计算advantage...")
        gae_start = time.time()
        if values_key not in data.batch:
            raise ValueError(f"cost_mode={cost_mode} requires '{values_key}' in data.batch")
        values = data.batch[values_key].float()
        logger.info(f"[DEBUG] [{mode_str}] values: shape={values.shape}, mean={values.mean().item():.6f}, sum={values.sum().item():.6f}")
        data.batch[values_key] = values * response_mask
        
        # 检查是否使用 step-level GAE（cost_mode 且使用了 step_costs）
        use_step_level_gae = False
        step_response_lengths_for_gae = None
        if cost_mode and "step_costs" in data.non_tensor_batch and "step_response_lengths" in data.non_tensor_batch:
            step_response_lengths_raw = data.non_tensor_batch["step_response_lengths"]
            # 解析 step_response_lengths
            if isinstance(step_response_lengths_raw, np.ndarray) and step_response_lengths_raw.dtype == object:
                if len(step_response_lengths_raw) > 0:
                    first_item = step_response_lengths_raw[0]
                    if isinstance(first_item, (list, tuple, np.ndarray)):
                        step_response_lengths_for_gae = [int(x) for x in (first_item.tolist() if isinstance(first_item, np.ndarray) else list(first_item))]
                    else:
                        step_response_lengths_for_gae = [int(first_item)]
                else:
                    step_response_lengths_for_gae = []
            else:
                step_response_lengths_for_gae = [int(x) for x in np.array(step_response_lengths_raw).flatten()] if len(step_response_lengths_raw) > 0 else []
            
            if len(step_response_lengths_for_gae) > 0:
                use_step_level_gae = True
                logger.info(f"[DEBUG] [{mode_str}] 使用step-level GAE计算（step数量={len(step_response_lengths_for_gae)}）...")
        
        if use_step_level_gae:
            # 使用 step-level GAE：每个 step 的 cost 单独用来优化本轮的 token
            advantages, returns = compute_step_level_gae_advantage_return(
                token_level_rewards=token_level_rewards,
                values=values,
                gamma=gamma,
                lambd=lambd,
                step_response_lengths=step_response_lengths_for_gae,
                logger=logger
            )
            logger.info(f"[DEBUG] [{mode_str}] Step-level GAE计算完成，耗时: {time.time() - gae_start:.3f}秒")
        else:
            # 使用标准的 trajectory-level GAE（原有逻辑）
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
            )
            logger.info(f"[DEBUG] [{mode_str}] Trajectory-level GAE计算完成，耗时: {time.time() - gae_start:.3f}秒")
    elif adv_estimator in ["reinforce", "grpo", "gigpo", "step_reinforce"]:
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    else:
        raise NotImplementedError

    logger.info(f"[DEBUG] [{mode_str}] advantages原始: shape={advantages.shape}, mean={advantages.mean().item():.6f}, sum={advantages.sum().item():.6f}")
    logger.info(f"[DEBUG] [{mode_str}] returns: shape={returns.shape}, mean={returns.mean().item():.6f}, sum={returns.sum().item():.6f}")

    data.batch[raw_advantages_key] = advantages
    if whiten_advantages:
        logger.info(f"[DEBUG] [{mode_str}] 执行whiten_advantages...")
        whiten_adv_start = time.time()
        # TODO whiten过程中是否要考虑response的长度？
        advantages = masked_whiten(values=advantages, mask=response_mask)
        logger.info(f"[DEBUG] [{mode_str}] whiten_advantages完成，耗时: {time.time() - whiten_adv_start:.3f}秒")
    advantages = advantages * response_mask

    if advantage_clip is not None:
        logger.info(f"[DEBUG] [{mode_str}] 执行advantage clipping (clip={advantage_clip})...")
        clip_start = time.time()
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        metric_key = f"critic/{'cost_' if cost_mode else ''}advantage_clip_frac"
        data.meta_info["metrics"] = {metric_key: adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)
        logger.info(f"[DEBUG] [{mode_str}] advantage clipping完成，clip_frac={adv_clip_frac:.6f}，耗时: {time.time() - clip_start:.3f}秒")

    data.batch[advantages_key] = advantages
    data.batch[returns_key] = returns
    
    total_time = time.time() - start_time
    logger.info(f"[DEBUG] [{mode_str}] token_level处理总耗时: {time.time() - process_start:.3f}秒")
    
    try:
        # log short summary for debugging
        resp_mask_sum = int(response_mask.sum().item())
        prefix = "compute_cost_advantage" if cost_mode else "compute_advantage"
        logger.info(
            f"{prefix}: token_reward_mean={token_level_rewards.mean().item():.6f}, "
            f"advantages_mean={advantages.mean().item():.6f}, returns_mean={returns.mean().item():.6f}, "
            f"resp_mask_sum={resp_mask_sum}, adv_sum={advantages.sum().item():.6f}"
        )
        logger.info(f"[DEBUG] [{mode_str}] compute_advantage 总耗时: {total_time:.3f}秒")
    except Exception as e:
        logger.warning(f"[DEBUG] [{mode_str}] 日志记录失败: {e}")
    
    return data


class GenerateRequestType(enum.Enum):
    ADD = enum.auto()
    ABORT = enum.auto()
    STOP = enum.auto()
    ALIVE_CHECK = enum.auto()


def postprocess_generate(
    prompts: "DataProto",
    output: torch.Tensor,
    num_return_sequences,
    sequence_length,
    eos_token_id,
    pad_token_id,
    fill_eos_token=False,
    output_logprobs: Optional[list[list[float]]]=None,
    pad_to_seq_len=True,
) -> "DataProto":
    from roll.distributed.scheduler.protocol import DataProto

    if fill_eos_token:
        # yali: 如果output最后一个token不是pad_token_id，则替换成eos_token_id,
        #  TODO: 需要消融这个变化的影响
        last_token_index = output.size(1) - 1
        need_replace_mask = output[:, last_token_index] != pad_token_id
        output[need_replace_mask, last_token_index] = eos_token_id

    input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
    attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
    prompt_id = prompts.batch.get("prompt_id", None)

    # input_batch_size * num_return_sequences
    output_batch_size = output.size(0)
    input_batch_size = input_ids.size(0)
    prompt_length = input_ids.size(1)

    if pad_to_seq_len:
        output = pad_to_length(output, sequence_length, pad_token_id)
        assert output.shape[1] == sequence_length, f"output shape {output.shape} != {sequence_length}"
    sequence_length = output.shape[1]

    prompt = output[:, :prompt_length].clone()  # (bs, prompt_length)
    response = output[:, prompt_length:].clone()  # (bs, response_length)

    attention_mask = (
        attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, prompt_length)
    )
    response_mask = get_pad_mask(response_id=response, pad_token=pad_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    position_ids = prompts.batch["position_ids"]
    # if is_num_return_sequences_expand=True, num_return_sequences here equals 1
    if position_ids.dim() == 3:  # qwen2vl mrope, maybe can support in other ways
        position_ids = (
            position_ids.unsqueeze(1)
            .repeat(1, num_return_sequences, 1, 1)
            .view(output_batch_size, *position_ids.shape[-2:])
        )
        delta_position_id = torch.arange(1, (sequence_length - prompt_length) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, 1, -1).expand(output_batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        # left padding for prompt and right padding for response, to be converted
        # to right padding which is consistent with output
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

    assert attention_mask.any(dim=1).all(), f"has all 0 attention_mask, {attention_mask} {input_ids}"
    first_one = attention_mask.float().argmax(dim=1)
    new_response_mask = torch.zeros_like(attention_mask)  # response mask for cat input_ids
    logprobs = torch.zeros([output_batch_size, sequence_length - 1], dtype=torch.float32) if output_logprobs is not None else None
    for i in range(output_batch_size):
        shift = first_one[i].item()
        if shift > 0:
            output[i, :-shift] = output[i, shift:].clone()
        else:
            output[i, :] = output[i, :].clone()
        valid_length = attention_mask[i].sum().int().item()
        response_length = response_mask[i].sum().int().item()
        attention_mask[i][:valid_length] = 1
        attention_mask[i][valid_length:] = 0
        prompt_len = valid_length - response_length
        new_response_mask[i][prompt_len : valid_length] = 1
        if logprobs is not None:
            logprobs[i][prompt_len - 1 : valid_length - 1] = torch.tensor(
                output_logprobs[i][:response_length], dtype=logprobs.dtype
            )
        if position_ids.dim() == 3 and shift > 0:
            # shift as output to convert to right padding
            # NOTE: left shift without clear right might lead to unclean values
            # in right part, which especially is the case when using long prompt
            # length and short response length. This usually makes no effect if
            # mask is right, while it might make trouble to for multi-modal model
            # like Qwen2-vl, since extra image_token would be left which might
            # cause error: Image features and image tokens do not match
            output_position_ids[i, ..., :-shift] = output_position_ids[i, ..., shift:].clone()
            # only clean in VLM(qwen2-vl) to make no effect on LLM
            if prompt_length > response_length:
                output[i, -shift:] = pad_token_id

    prompt_mask = (attention_mask == 1) & (new_response_mask == 0)
    if position_ids.dim() == 3:
        position_ids = output_position_ids
    else:  # normal position_ids
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    batch = TensorDict(
        {
            "prompts": prompt,
            "responses": response,
            "input_ids": output,  # right pad
            "attention_mask": attention_mask,  # right pad
            "position_ids": position_ids,
            "prompt_mask": prompt_mask,
            "response_mask": new_response_mask,  # right pad, response tokens
        },
        batch_size=output_batch_size,
    )
    if prompt_id is not None:
        prompt_id = (
            prompt_id.squeeze().unsqueeze(1).repeat(1, num_return_sequences).view(output_batch_size, -1).squeeze(-1)
        )
        batch["prompt_id"] = prompt_id
    if logprobs is not None:
        batch["infer_logprobs"] = logprobs
    return DataProto(batch=batch)


def get_dist_info_from_comm_plan(comm_plan, rank_in_cluster, rank_in_worker):
    for src_rank, comm_plan_args in comm_plan.items():
        start_rank = 0
        for tgt_device in comm_plan_args["tgt_devices"]:
            start_rank += 1
            if tgt_device["rank"] == rank_in_cluster and tgt_device["device"]["rank"] == rank_in_worker:
                return start_rank, comm_plan_args
    return None, None


def separate_prompt_response(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor, pad_id: int
):
    prompt_mask = attention_mask.bool() & ~response_mask.bool()
    response_mask_valid = attention_mask.bool() & response_mask.bool()
    prompt_ids = torch.where(prompt_mask, input_ids, torch.full_like(input_ids, pad_id))
    response_ids = torch.where(response_mask_valid, input_ids, torch.full_like(input_ids, pad_id))
    return prompt_ids, response_ids

def filter_func_args(func, forward_args):
    signature = inspect.signature(func)
    forward_params = signature.parameters.keys()
    valid_args = {k: v for k, v in forward_args.items() if k in forward_params}
    return valid_args


def aggregate_metrics(history_metrics: List[Dict], metrics_agg_mode: Dict[str, str]) -> Dict[str, float]:
    """
    Aggregate metrics from history based on the specified aggregation modes.

    Args:
        history_metrics: List of dictionaries containing metrics for each step
        metrics_agg_mode: Dictionary mapping metric names to aggregation modes
                         Supported modes: "sum", "mean", "min", "max", "last", "first"

    Returns:
        Dictionary of aggregated metrics
    """
    # Collect all metrics from history
    all_metrics = {}
    for metrics in history_metrics:
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(float(v))

    # Aggregate metrics based on mode
    aggregated_metrics = {}
    for metric_name, values in all_metrics.items():
        mode = metrics_agg_mode.get(metric_name, "mean")  # default to mean
        if mode == "sum":
            aggregated_metrics[metric_name] = float(np.sum(values))
        elif mode == "mean":
            aggregated_metrics[metric_name] = float(np.mean(values))
        elif mode == "min":
            aggregated_metrics[metric_name] = float(np.min(values))
        elif mode == "max":
            aggregated_metrics[metric_name] = float(np.max(values))
        elif mode == "last":
            aggregated_metrics[metric_name] = float(values[-1])
        elif mode == "first":
            aggregated_metrics[metric_name] = float(values[0])
        else:
            # Default to mean for unknown modes
            aggregated_metrics[metric_name] = float(np.mean(values))

    return aggregated_metrics


def group_reward_norm(data: "DataProto", n_sample=-1, div_std=True, div_std_global=False):
    assert n_sample > 1, "n_sample must > 1"
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    reshape_reward = response_level_rewards.reshape(*response_level_rewards.size()[:-1], -1, n_sample)
    reshape_reward = reshape_reward - reshape_reward.mean(dim=-1, keepdim=True)
    if div_std:
        if not div_std_global:
            reshape_reward = reshape_reward / (torch.std(reshape_reward, dim=-1, keepdim=True) + 1e-6)
        else:
            reshape_reward = reshape_reward / (torch.std(reshape_reward) + 1e-6)
    data.batch["response_level_rewards"] = reshape_reward.reshape(*response_level_rewards.size())
    return data
