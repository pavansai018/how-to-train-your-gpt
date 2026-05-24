# Data Preparation for Fine-Tuning

## The short answer

Fine-tuning data is a collection of examples. Each example shows the
model what to do. Give it an input and the desired output. After seeing
thousands of examples the model learns the pattern and can produce
correct outputs for new inputs. The quality of the data matters more
than the quantity. Cleaning the data and formatting it consistently is
most of the work.

## The basic format

Every example has at least an instruction and a response. Some examples
also have a system prompt or context.

```json
{
  "instruction": "Summarize this paragraph in one sentence.",
  "input": "The cat sat on the mat for three hours...",
  "response": "A cat stayed on a mat for an extended period."
}
```

The input field is optional. Many tasks only need an instruction. The
model learns from the pattern of instruction followed by response.

## Chat templates

Most fine-tuned models use a chat format. The instruction and response
are wrapped in special tokens that tell the model where the user
message ends and the assistant response begins.

```
<|user|>
What is the capital of France?
<|assistant|>
The capital of France is Paris.
<|endoftext|>
```

The exact tokens depend on the base model. LLaMA uses `[INST]` and
`[/INST]`. Mistral uses `<s>[INST]` and `[/INST]`. Open source chat
models often use `<|user|>` and `<|assistant|>` markers. Using the
wrong tokens means the model does not recognize the format and
produces garbage.

## How much data you need

The minimum is around one hundred examples. Below that the model cannot
generalize. It memorizes the examples and fails on new inputs. A
thousand examples is solid for most tasks. Ten thousand is good for
complex tasks like multi-turn conversation or code generation. Beyond
ten thousand the returns diminish quickly.

```
100 examples:   Bare minimum. Model might not generalize.
1,000 examples: Good for most classification and simple QA tasks.
5,000 examples: Good for instruction following and summarization.
10,000+ examples: Complex tasks. Diminishing returns after this.
```

The quality of each example matters more than the total count. A
thousand carefully written examples beat ten thousand sloppy ones.
Every mistake in the training data teaches the model to make that
mistake. The model learns to replicate patterns. Not to judge them.

## Cleaning the data

Before training go through your data and remove anything that would
teach the model bad habits.

Examples of what to remove. Responses that are cut off or incomplete.
Responses that contradict the instruction. Instructions that are
ambiguous or impossible to follow. Repeated examples. Examples where
the response is in the wrong language or format. Examples with
offensive or harmful content.

```
Good example:
  Instruction: "What is 2 + 2?"
  Response: "2 + 2 equals 4."

Bad example (vague):
  Instruction: "Tell me about math."
  Response: "It's cool."

Bad example (contradicts):
  Instruction: "Give a short answer."
  Response: "Let me explain this in great detail over several paragraphs..."
```

## Balancing the data

Fine-tuning can unbalance the model. If you train only on one task the
model gets worse at everything else. This is called catastrophic
forgetting. The model forgets how to have a normal conversation because
every training example is a specific task.

Mix in some general conversation examples. About ten percent of your
data should be normal chat. This keeps the model from losing its
general abilities while learning the new task.

## The prompt matters

The instruction text affects the results. Different phrasings produce
different behaviors. If you train with instructions like *Summarize
this* the model learns to summarize when it sees those words. If
someone asks *Give me a short version* instead the model might not
recognize it as a summarization request.

Write instructions in multiple ways. For each task provide three or
four different phrasings of the same instruction. This teaches the
model the intent behind the words rather than the exact wording.

```
"Summarize this paragraph."
"Can you summarize this?"
"Give me a summary of the following text."
"Briefly summarize what this says."
```

All four mean the same thing. Training on all four variants makes the
model robust to different phrasings.

## Splitting the data

Divide your data into training and validation sets. Train on ninety
percent. Evaluate on ten percent. The validation set tells you if the
model is learning the task or just memorizing the examples.

If the training loss goes down but the validation loss goes up the
model is memorizing. Stop training. Reduce the number of steps or
increase dropout. If both go down you are on the right track.

## A real example

Here is a small instruction tuning dataset in the format used by Alpaca.

```json
[
  {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced diet with plenty of vegetables..."
  },
  {
    "instruction": "What are the three primary colors?",
    "input": "",
    "output": "The three primary colors are red blue and yellow."
  },
  {
    "instruction": "Describe the following city in three sentences.",
    "input": "Tokyo",
    "output": "Tokyo is the capital of Japan and one of the most..."
  }
]
```

Each example is a self-contained lesson. The model sees the instruction
and learns to produce the output. After thousands of examples it can
handle variations of instructions it has never seen before.

## What you need to remember

Fine-tuning data is instruction response pairs. The format uses special
tokens specific to each base model. A thousand good examples beats ten
thousand bad ones. Clean the data carefully. The model learns to
replicate patterns not to judge their quality. Mix in general
conversation to prevent catastrophic forgetting. Vary the phrasing of
instructions to build robustness. Split into training and validation
sets to monitor for memorization.
