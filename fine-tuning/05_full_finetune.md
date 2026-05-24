# Full Fine-Tuning

## The short answer

Full fine-tuning updates every weight in the model. Nothing is frozen.
Every parameter is trained on the new task. This is the most powerful
approach but requires the most hardware. It is also the most expensive
and the least flexible because you must store a complete copy of the
model for every task you fine-tune for.

## Where it sits

Full fine-tuning takes the pretrained checkpoint and continues
training. The only difference from pretraining is the data. Instead of
raw internet text the model sees instruction response pairs. The
training loop is identical. Forward pass. Compute loss. Backward pass.
Update weights. The same code that trained the base model trains the
fine-tuned model.

## Why you might need it

LoRA works for most tasks. But some tasks require changing the model's
fundamental behavior in ways that low rank updates cannot capture.
Teaching a model a new language that was not in its pretraining data.
Radically changing the output format or style. Correcting systematic
biases or errors in the base model. These cases might justify full
fine-tuning.

Full fine-tuning also produces the highest quality results when
measured on standard benchmarks. The gap between full fine-tuning and
LoRA has been shrinking as base models improve but for the most
demanding applications full fine-tuning still wins by a small margin.

## Why you probably do not need it

Most real world use cases do not need full fine-tuning. Instruction
following. Chat behavior. Summarization. Classification. Translation.
All of these work well with LoRA. The additional cost of full
fine-tuning rarely justifies the marginal quality improvement.

Full fine-tuning also introduces operational complexity. Each fine-tuned
variant is a separate multi-gigabyte checkpoint. Storing serving and
versioning these checkpoints is expensive. With LoRA you store one base
model and many small adapter files.

## The hardware requirement

Full fine-tuning needs enough GPU memory to store the model weights the
optimizer states and the activations for a training batch. The optimizer
states are the largest cost because AdamW stores two values per
parameter.

```
Memory for a 7B parameter model during full fine-tuning:

Model weights (bfloat16):         14 GB  (7B × 2 bytes)
Optimizer states (float32):       56 GB  (7B × 2 × 4 bytes)
Gradients (float32):              28 GB  (7B × 4 bytes)
Activations (bfloat16, batch=1):  ~2 GB (varies by seq length)
-------------------------------------------------
Total:                           ~100 GB
```

This requires multiple GPUs. An RTX 4090 has 24 gigabytes. You would
need four or five of them just to load the model and optimizer. This is
why full fine-tuning is done in data centers not on desktops.

## The process

Full fine-tuning follows the same steps as pretraining but with
different hyperparameters.

```
Learning rate:   1e-5 to 5e-5  (much smaller than pretraining)
Warmup steps:    100 to 500     (less warmup needed)
Total steps:     1 to 5 epochs  (fewer epochs because less data)
Batch size:      As large as fits in memory
Weight decay:    0.1            (same as pretraining)
```

The learning rate is ten to thirty times smaller than pretraining. The
model is already close to a good solution. Large steps would destroy
the pretrained knowledge. Small steps refine the behavior without
disrupting the foundation.

## Catastrophic forgetting

Full fine-tuning has a risk that LoRA avoids. When every weight can
change the model can forget its pretrained abilities. A model
fine-tuned on only medical questions might lose the ability to discuss
sports or politics or history. The weights drift away from the
pretrained configuration.

Mitigations exist. Mix general data into the fine-tuning set. Use a
very small learning rate. Stop training as soon as the validation loss
stops improving. But even with mitigations some forgetting is
inevitable. The model trades some of its broad knowledge for deep
knowledge of the fine-tuning task.

LoRA avoids this by freezing the base weights. The original knowledge
is preserved exactly because the original weights never change. Only
the adapters learn the new task. This is a fundamental architectural
advantage of LoRA over full fine-tuning for most applications.

## When it is worth it

Full fine-tuning makes sense in a few specific scenarios. When you are
building a foundation model that others will fine-tune. When the task
requires changing the model's knowledge not just its behavior. When
the marginal quality improvement from full fine-tuning translates to
real business value. When you have the hardware budget and the
operational infrastructure to manage large model files.

For everyone else LoRA or QLoRA is the right choice. Less hardware.
Less storage. Less time. Nearly the same quality. The gap continues to
shrink as base models improve and adapters get better at capturing
task specific behavior.

## What you need to remember

Full fine-tuning updates every weight in the model on task specific
data. It requires expensive hardware because optimizer states consume
four times the memory of the model itself. It risks catastrophic
forgetting where the model loses its general abilities. The quality
improvement over LoRA is marginal for most tasks. Full fine-tuning is
the right tool only when you are building foundation models or need
the absolute maximum performance regardless of cost.
