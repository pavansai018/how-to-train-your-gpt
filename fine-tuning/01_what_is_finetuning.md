# What Is Fine-Tuning

## The short answer

Fine-tuning takes a model that already knows language and teaches it
a specific skill. The model arrives knowing grammar and facts and
conversation patterns from pretraining. Fine-tuning adds a new layer
of training on a focused dataset. The result is a model that follows
instructions or classifies sentiment or translates languages or any
other task you can describe with examples.

## Where it sits in the pipeline

```
Raw text from the internet
  → Pretraining (expensive, general)
    → Base model (knows language, not chat)
      → Fine-tuning (cheaper, focused)
        → Chat model (helpful, follows instructions)
```

Pretraining takes weeks on thousands of GPUs. Fine-tuning takes hours
on a single GPU. The base model does the heavy lifting. Fine-tuning just
points it in the right direction.

## Why we need it

A base model can complete text. Feed it *The capital of France is* and
it finishes with *Paris.* Feed it *Summarize this article* and it
finishes with *in a few sentences* instead of actually summarizing. The
base model does not know it is supposed to follow instructions. It was
trained to predict the next word in internet text. Internet text
contains examples of people asking for summaries but also contains
examples of people writing completions for *Summarize this article.* The
model learned to complete. Not to follow.

Fine-tuning teaches the model to distinguish between the two. By
training on examples of instructions paired with the desired responses
the model learns to recognize when it is being asked to do something
and how to produce the correct output.

## The three approaches

| Approach | What happens | Hardware needed | How long |
|---|---|---|---|
| Full fine-tuning | Update every weight in the model | 4-8 A100 GPUs | Hours to days |
| LoRA | Train small adapter matrices. Original weights frozen | Single 3090/4090 | Minutes to hours |
| QLoRA | Quantize the base model to 4-bit. Train LoRA adapters | Single laptop GPU | Hours |

Full fine-tuning updates all 152 million weights. This is the most
powerful approach but requires hardware that most people do not own.
The model is fully adapted to the new task but the original model is
lost. You must store a separate full copy of the model for each task
you fine-tune for.

LoRA does not change the original weights. It adds small trainable
matrices alongside them. At the end you have the original model plus
tiny adapter files that are a few megabytes each. You can swap adapters
in and out like changing a lens on a camera. One base model. Many
adapters. Each adapter makes the model good at a different thing.

QLoRA is LoRA with one extra trick. It compresses the base model to
four bits per weight before applying LoRA. The compression reduces the
memory needed to load the model by four to eight times. A seven billion
parameter model that normally needs fourteen gigabytes can run in under
four gigabytes with QLoRA. This makes fine-tuning accessible on
consumer hardware.

## When to use which

If you have access to a cluster of GPUs and need maximum performance
for a mission critical task use full fine-tuning.

If you have a single good GPU and want to teach the model a new skill
use LoRA. This covers almost everyone building applications. Chatbots
customer support agents code assistants and content generators are all
routinely built with LoRA.

If you have a laptop or an older GPU and still want to fine-tune use
QLoRA. The quality is slightly lower than LoRA but the model still
learns the task successfully. The difference shrinks as base models
get better.

## The data format

Fine-tuning data is simple. You provide prompt and response pairs.

```
{
  "instruction": "Translate to French: Hello, how are you?",
  "response": "Bonjour, comment allez-vous?"
}
```

Every example teaches the model one thing. Given this input produce this
output. After seeing thousands of examples the model generalizes and can
produce correct outputs for inputs it has never seen before.

The format varies by task but the principle is the same. Show the model
what to do. Let it figure out the pattern. Do not tell it the rules
explicitly. Let it learn from examples like it learned language from
sentences.

## What fine-tuning cannot do

Fine-tuning cannot teach the model new knowledge that was not in its
pretraining data. If the model was never trained on medical records it
cannot suddenly become a doctor. Fine-tuning can only rearrange and
apply knowledge the model already has.

Fine-tuning cannot fix fundamental architectural limitations. A model
with a small context window stays small. A model that hallucinates will
still hallucinate. Fine-tuning can reduce certain failure modes but
cannot eliminate them.

Fine-tuning cannot make a bad model good. The base model must already
be competent. Fine-tuning on a tiny incompetent model produces a tiny
competent model. The gap between base models of different sizes
persists after fine-tuning. A fine-tuned 7B model never catches a
fine-tuned 70B model.

## What you need to remember

Fine-tuning adapts a pretrained model to a specific task. It is cheaper
and faster than pretraining because it starts from a model that already
knows language. LoRA is the standard approach because it is fast and
efficient and produces tiny adapter files. QLoRA extends LoRA to run
on consumer hardware. The data format is simple instruction response
pairs. Fine-tuning does not add new knowledge. It rearranges existing
knowledge to follow patterns.
