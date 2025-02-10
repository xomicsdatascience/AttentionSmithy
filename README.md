# AttentionSmithy

The [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) paper completely revolutionized the AI industry. After inspiring such programs like GPT and BERT, it seems all deep learning research began exclusively focusing on the attention mechanism behind transformers. This has created a great deal of research surrounding the topic, spawning hundreds of variations to the original paper meant to enhance the original program or tailor it to new applications. Most of these developments happen in isolation, disconnected from the broader community and incompatible with tools made by other developers. For developers that want to experiment with combining these ideas to fit a new problem, such a disjointed state is frustrating.

AttentionSmithy was designed as a platform that allows for flexible experimentation with the attention mechanism in a variety of applications. This includes the ability to use a multitude of positional embeddings, variations on the attention mechanism, and others.

The baseline code was originally inspired by [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) blog code.

## AttentionSmithy Components

Here is a visual depiction of the different components of a transformer model, using Figure 1 from Attention Is All You Need as reference.

![Screenshot 2025-02-10 at 3 53 42 PM](https://github.com/user-attachments/assets/29acea66-c865-48fd-9cde-ce55a2be08af)

## AttentionSmithy Numeric Embedding

Here is a visual depiction of where each positional or numeric embedding fits in to the original model.

![Screenshot 2025-02-10 at 3 49 43 PM](https://github.com/user-attachments/assets/4a7d1983-6594-4953-8936-2b438de5de82)

## AttentionSmithy Attention Methods

Here is a basic visual of possible attention mechanisms AttentionSmithy has been designed to incorporate in future development efforts. The provided examples include [Longformer attention](https://arxiv.org/abs/2004.05150) and [Big Bird attention](https://arxiv.org/abs/2007.14062).

![Screenshot 2025-02-10 at 3 45 58 PM](https://github.com/user-attachments/assets/8dc33378-4b53-456e-8abb-22ab42c7f6c1)

