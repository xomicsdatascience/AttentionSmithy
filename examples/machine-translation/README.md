# Machine Translation Model

This repository contains an example of running a machine translation transformer model built using AttentionSmithy. It will create a model that translates sentences from German to English.

The model parameters mostly align with the configuration described in Vaswani et al (2023). However, for the sake of demonstrating how positional encoding methods are implemented, all 4 strategies described in the paper are utilized in this example.

The model is created using pytorch lightning to streamline data loading, GPU allocation etc. See `model_import.py` and `data_import.py` for implementation details.

## Prerequisites

To run this project efficiently, your computer should have a GPU with CUDA (for NVIDIA GPUs) or MPS (for MacBooks with Apple Silicon). Running on a CPU is possible but significantly slower.

## Setting Up the Python Environment

To ensure compatibility, use Python 3.9 or greater. You can set up the environment using either Conda or a virtual environment with `venv`.

### Using Conda

1. Create a new Conda environment:
   ```sh
   conda create --name mtm_env python
   ```
2. Activate the environment:
   ```sh
   conda activate mtm_env
   ```

### Using venv

1. Create a virtual environment:
   ```sh
   python -m venv mtm_env
   ```
2. Activate the environment:
   - On macOS/Linux:
     ```sh
     source mtm_env/bin/activate
     ```
   - On Windows:
     ```sh
     mtm_env\Scripts\activate
     ```

### Installation

Once the environment is set up and activated, install the necessary packages:

```sh
pip install attention-smithy==1.2.1 datasets
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

For demonstration purposes, the script is configured to run for ten epochs. At the end of each epoch, the script will print ten English validation sentences alongside their corresponding translations.

It is important to note that the WMT-14 German-English dataset includes a range of politically themed statements. Consequently, there remains a noticeable bias towards topics prevalent in the dataset, especially in early epochs. However, understandable grammatical structure still emerges in the translations.

By the end of the first epoch, the output may resemble the following:

```text
Reference: who came up with this idea?                                                                                                                                                                                
Output: this is why this issue?

Reference: yes, it does.
Output: however, there is no longer that.

Reference: a king with 14 wives
Output: a great deal with regard to the same time.

Reference: monitoring
Output: mr president.

Reference: everybody fought for everybody.
Output: this has been taken.

Reference: where do the weapons come from?
Output: why?

Reference: weight
Output: the vote will take place tomorrow.

Reference: friends were baffled.
Output: mr president.

Reference: i just ignore them.
Output: i would like to say it.

Reference: homex long term
Output: i would like to ask you.
```

By the tenth epoch, the translations typically exhibit greater alignment with the reference sentences, as demonstrated below:

```text
Reference: who came up with this idea?                                                                                                                                                                                
Output: what is the idea?

Reference: yes, it does.
Output: yes, that is true.

Reference: a king with 14 wives
Output: a king with 14 member states

Reference: monitoring
Output: monitoring

Reference: everybody fought for everybody.
Output: everyone has employed the use of everyone.

Reference: where do the weapons come from?
Output: where do the weapons come?

Reference: weight
Output: attention

Reference: friends were baffled.
Output: my friends were confusing.

Reference: i just ignore them.
Output: i ignore that.

Reference: homex long term
Output: in the long - term
```

Extending the training duration and incorporating additional data further enhances translation quality over time.

## Notes

- The model is configured to run **100,000 sentences** for **10 epochs** with a **maximum sequence length of 100** tokens.
- This setup is meant for demonstration purposes and can be adjusted for more extensive training.

Happy translating! 🚀

