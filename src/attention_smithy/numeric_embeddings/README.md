# Numeric Embeddings

## Position Embeddings
In a basic transformer setup, data is input in parallel. For example, if provided the following sentence (presuming words are tokens):

```
The dog chased the cat
```

Each word in the sentence - `the`, `dog`, `chased`, `the`, `cat` - is fed into the transformer simultaneously. 
This has its upsides - computationally, you can run more data through in a single pass. 
However, this also renders the transformer permutation-invariant - you can scramble the words and the transformer wouldn't know the difference, because they were not entered sequentially.
Thus, if you swapped two of the words, changing the meaning of the sentence entirely:

```
The dog chased the cat
The cat chased the dog
```

The transformer would not natively distinguish a difference. 
Further, if you provided it a truly scrambled version of the sentence:

```
chased cat dog the the
```

Again, the transformer would not natively interpret it any differently, even though the sentence is syntactically invalid.

Obviously, order - word order - matters in natural language. 
It matters in other data types as well - pixel coordinates in a picture, for instance.
Thus, positional encodings must be artificially encoded into the dataset itself.

## How to Encode Numbers

Numerous strategies have been proposed for encoding position. 
Generally, position is encoded as integers, such as the rank of words in a sentence:

| the | dog | chased | the | cat |
|-----|-----|--------|-----|-----|
|  0  |  1  |   2    |  3  |  4  |

And then these integers are given their own encoding and combined with the token or word embedding.
Thus, the word embedding - its "definition," so to speak, from the computer's perspective - takes on new, positional meaning.
"Dog" does not just mean "dog" anymore - it means "dog" as a second word in a sentence.

These strategies can often be extended to any numeric data, not just position, depending on the dataset being analyzed.
For example, clinical time series data would more accurately be reflected as precise floats (such as `1.24`) rather than position such as (`index 3`).
Thus, they could be more accurately called "numeric" encodings rather than specifically positional encodings.

## Where Embedding Strategies Apply

Different strategies are often applied at different locations in a transformer architecture. 
Below is a summary of where 4 popular methods - sinusoidal, learned, rotary, and ALiBi - apply in the process.

![Screenshot 2025-02-10 at 3 49 43 PM](https://github.com/user-attachments/assets/4a7d1983-6594-4953-8936-2b438de5de82)

To simplify the process, `AttentionSmithy` is designed to allow for turning on or off any number of numeric embedding strategies at model initialization.
The `NumericEmbeddingManager` is provided the necessary or desired strategies at initialization.
It then calls those strategies at the appropriate time in `AttentionSmithy` component blocks.
Below is a coded example, activating all 4 strategies simultaneously.

```python
from attention_smithy.numeric_embeddings import (
    SinusoidalPositionEmbedding, LearnedPositionEmbedding,
    RotaryPositionEmbedding, ALiBiPositionEmbedding,
    NumericEmbeddingManager
)

NumericEmbeddingManager([
    SinusoidalPositionEmbedding(self.config['embedding_dimension']),
    LearnedPositionEmbedding(max_sequence_length, self.config['embedding_dimension']),
    RotaryPositionEmbedding(self.config['embedding_dimension'] // self.config['number_of_heads']),
    ALiBiPositionEmbedding(self.config['number_of_heads']),
])

```

## Encoded Strategies

Below is a summary of each positional encoding method available in AttentionSmithy.

### Sinusoidal

### Learned

### Rotary

### ALiBi
