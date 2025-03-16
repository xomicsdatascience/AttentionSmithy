# Machine Translation Model

This repository contains an example of running a machine translation transformer model built using AttentionSmithy. It will create a model that translates sentences from German to English.

The model parameters mostly align with the configuration described in Vaswani et al (2023). However, for the sake of demonstrating how positional encoding methods are implemented, all 4 strategies described in the paper are utilized in this example.

The model is created using pytorch lightning to streamline data loading, GPU allocation etc. See `model_import.py` and `data_import.py` for implementation details.

## Prerequisites

To run this project efficiently, your computer should have a GPU with CUDA (for NVIDIA GPUs) or MPS (for MacBooks with Apple Silicon). Running on a CPU is possible but significantly slower.

## Installation

Before running the model, install the necessary dependencies using `pip` in a pip or conda environment:

```sh
pip install attention_smithy datasets
```

## Running the Pipeline

### Step 1: Download the Data

Execute the following script to download and preprocess the WMT-14 German-English dataset:

```sh
python data_download.py
```

This script will fetch the required dataset and prepare it for training. The files should have a total size of approximately 1.5 GB.

### Step 2: Run the Model

Run the main script to start training the translation model:

```sh
python main.py
```

## Expected Output

At the end of each epoch, the script should print out 50 of the English validation sentences and their translated versions.

## Notes

- The model is configured to run **100,000 sentences** for **10 epochs** with a **maximum sequence length of 100** tokens.
- This setup is meant for demonstration purposes and can be adjusted for more extensive training.
- Some design choices were made that promoted readability over efficiency. This includes functionality that groups sentences by length in each batch.

Happy translating! 🚀

