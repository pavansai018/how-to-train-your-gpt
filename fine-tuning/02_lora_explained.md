# LoRA: Low-Rank Adaptation

## The short answer

LoRA fine-tunes a model without changing its original weights. It adds
small trainable matrices to specific layers. At the end you have the
original model plus a tiny adapter file. The adapter is usually a few
megabytes. You can train adapters for different tasks and swap them
instantly. One model. Many skills.

## Where it sits

LoRA is inserted into the linear layers of the transformer. Typically
the query and value projections in attention and sometimes the feed
forward layers. The original weight matrix stays frozen. LoRA adds two
small matrices that are trained from scratch.

```
Without LoRA:
  output = input @ W  (W is large, 768 × 768, 589,824 params)

With LoRA:
  output = input @ W + input @ A @ B
                      ^^^^^^^^^^^^^^^^
                      W is frozen. A and B are trained.
                      A is 768 × 16 (12,288 params)
                      B is 16 × 768 (12,288 params)
                      LoRA adds only 24,576 params per layer
```

The rank r is small. Typically 8 16 or 32. Our example uses 16. The
original weight matrix has rank up to 768. LoRA constrains the update
to be low rank. The hypothesis is that fine-tuning updates are low rank
anyway. You do not need the full rank to adapt a pretrained model to a
new task. A small subspace is enough.

## Why it works

Pretrained models have rich representations. The weights already encode
the structure of language. Fine-tuning does not need to teach the model
language again. It only needs to teach a new behavior on top of the
existing understanding. Teaching a new behavior on top of a rich
representation requires only small adjustments. LoRA captures those
small adjustments in a compact form.

The rank r controls the tradeoff. Higher rank means more capacity to
learn complex patterns. But also more parameters and slower training.
For most instruction tuning tasks a rank of 8 or 16 is sufficient.
For tasks that require learning new formats or patterns a rank of 32
or 64 might help.

## The math

A full fine-tuning update to a weight matrix W is ΔW. The new weight
is W + ΔW. ΔW has the same shape as W. For a 768 by 768 matrix ΔW
has 589824 independent parameters.

LoRA approximates ΔW as the product of two smaller matrices A and B.
A has shape 768 by r. B has shape r by 768. Their product A times B has
shape 768 by 768 but only 2 times 768 times r independent parameters.
For r equals 16 that is 24576 parameters. A compression ratio of 24.

```
ΔW ≈ A × B

ΔW[i][j] = sum over k from 0 to r-1: A[i][k] × B[k][j]
```

During training W is frozen. Gradients flow only through A and B. A
is initialized with random normal values. B is initialized to zero so
that at the start of training the adapter does nothing. The model
behaves exactly like the base model until A and B start learning.

## The alpha parameter

LoRA has a scaling factor called alpha. The LoRA output is scaled by
alpha divided by rank before being added to the original output.

```
output = input @ W + (alpha / r) × (input @ A @ B)
```

Alpha controls how much influence the adapter has. A larger alpha means
the adapter has more impact on the output. The standard value is twice
the rank. So for rank 16 alpha is usually 32.

When you change the rank the alpha to rank ratio determines the
effective learning rate of the adapter. Keeping alpha equal to twice
the rank means the ratio stays 2 regardless of rank. This makes
hyperparameter tuning transferable across rank values.

## The dropout

LoRA adapters usually include a small dropout on the A matrix output.
Dropout is 0.05 or 0.1. It prevents the adapter from overfitting to
the limited fine-tuning data. The base model provides regularization
because its weights are frozen. The adapter needs its own small
regularization.

## Which layers to adapt

The standard choice is query and value projections in every attention
layer. Some configurations also adapt the output projection and the
feed forward layers. More adapted layers means more capacity but also
more parameters.

```
Minimal (recommended for most tasks):
  Query projection (W_q)
  Value projection (W_v)

Standard (good for instruction tuning):
  Query projection (W_q)
  Value projection (W_v)
  Key projection (W_k)
  Output projection (W_o)

Full (maximum capacity):
  All linear layers including feed forward
```

For our model with 12 layers the minimal configuration adds about 0.3
million parameters on top of 152 million frozen parameters. The adapter
is about 2 megabytes. A full fine-tuning checkpoint would be 600
megabytes for the same model.

## Merging adapters

At inference time you have two choices. Keep the adapter separate and
compute the LoRA update on the fly. Or merge the adapter into the base
weights for faster inference.

```
Separate (training and experimentation):
  output = input @ W + input @ A @ B
  Flexible. Easy to swap adapters. Slightly slower inference.

Merged (deployment):
  W_merged = W + A @ B
  output = input @ W_merged
  No extra computation. No slowdown. Permanent.
```

Merging is a one way operation. After merging you cannot separate the
adapter from the base weights. But inference is as fast as the original
model. Most deployments merge before serving.

## A tiny code example

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.rank = rank
        self.alpha = alpha

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_b(self.lora_a(x)) * (self.alpha / self.rank)
        return base + lora

# Replace a 768x768 linear layer with LoRA
layer = LoRALinear(768, 768, rank=16)

frozen_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

print(f"Frozen parameters: {frozen_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Compression ratio: {frozen_params / trainable_params:.0f}x")
```

## What you need to remember

LoRA adds small trainable matrices to frozen pretrained layers. The
update is constrained to be low rank. The rank controls capacity versus
parameter count. The alpha parameter scales the adapter influence.
Adapters are tiny files that can be swapped at will. Merging eliminates
inference overhead. LoRA makes fine-tuning accessible on consumer GPUs
while maintaining most of the quality of full fine-tuning.
