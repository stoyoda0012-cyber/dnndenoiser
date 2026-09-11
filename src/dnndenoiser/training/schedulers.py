"""Learning-rate schedules used by the command-line trainer.

One callable, extracted from a larger training module that is not part of the
supported surface. It depends on ``LambdaLR`` and numpy and nothing else.
"""
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

__all__ = ['get_cosine_schedule_with_warmup']


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     num_cycles=0.5, min_lr_ratio=0.01):
    """Create scheduler with linear warmup and cosine decay

    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 = half cycle, ends at min)
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
