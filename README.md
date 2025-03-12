# HubermanGPT

This project focuses on generating and processing data from Huberman Lab podcast transcripts, followed by fine-tuning a language model (Mistral-7B) to create a specialized AI assistant, **HubermanGPT**, which can communicate complex neuroscience concepts in an accessible manner.

## Overview

The project consists of two main components:
1. **Data Generation**: Extracting and processing transcripts from Huberman Lab podcast videos.
2. **Model Fine-Tuning**: Fine-tuning the Mistral-7B model using the processed data to create HubermanGPT.

The goal is to create an AI assistant that can provide concise and detailed explanations of neuroscience topics, adapt to user input, and respond thoughtfully to feedback.

---

## Project Structure

### 1. **Data Generation**
- **Input**: Metadata and video IDs from Huberman Lab podcast episodes.
- **Process**:
  - Extract transcripts using the `youtube_transcript_api`.
  - Clean and preprocess the transcripts.
  - Split transcripts into chunks for model input.
  - Format the data for fine-tuning.
- **Output**: A dataset of formatted transcripts ready for model training.

### 2. **Model Fine-Tuning**
- **Model**: Mistral-7B (GPTQ quantized version).
- **Fine-Tuning**:
  - Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
  - Train the model on the processed Huberman Lab transcripts.
  - Push the fine-tuned model to the Hugging Face Hub.
- **Output**: A fine-tuned model (`HubermanGPT`) capable of generating neuroscience-related responses.

---

## Key Features

### Data Generation
- **Transcript Extraction**: Automatically fetch transcripts from YouTube videos using the `youtube_transcript_api`.
- **Chunking**: Split long transcripts into smaller chunks (12,000 characters) with an overlap of 2,000 characters for context.
- **Formatting**: Format the data into a structured prompt-response format for fine-tuning.

### Model Fine-Tuning
- **LoRA Configuration**: Use LoRA for efficient fine-tuning with the following parameters:
  - Rank (`r`): 8
  - Alpha (`lora_alpha`): 32
  - Target Modules: `q_proj`
  - Dropout: 0.05
- **Training**: Fine-tune the model using the processed dataset with the following hyperparameters:
  - Learning Rate: 2e-4
  - Batch Size: 4
  - Epochs: 10
  - Gradient Accumulation Steps: 4
  - Optimizer: Paged AdamW 8-bit

### HubermanGPT
- **Functionality**:
  - Communicate complex neuroscience concepts in an accessible manner.
  - Provide concise or detailed explanations based on user input.
  - Adapt responses to user feedback and requests for deeper insights.

---

## Requirements

### Python Libraries
- **Data Generation**:
  - `pandas`
  - `youtube_transcript_api`
  - `datasets`
- **Model Fine-Tuning**:
  - `transformers`
  - `peft`
  - `bitsandbytes`
  - `optimum`
  - `auto-gptq`
  - `accelerate`
  - `torch`

## Usage

To use the fine-tuned **HubermanGPT** model, follow these steps:

1. **Install Required Libraries**:
   Make sure you have the necessary libraries installed:
   ```bash
   pip install transformers peft torch

2. **Run the Inference Script:**
   Use the inference.py script to load the model and generate responses.
  ## Example Python Code

Here's a simple Python script to print "Hello, World!":

```python
   python inference.py
```
This will load the fine-tuned model and generate a response to the example prompt.

3. **Customize the Prompt:**
    ```python
    prompt = '''<s>[INST] HubermanGPT, functioning as a virtual neuroscience expert, communicates complex scientific concepts in an accessible manner.
    It escalates to deeper details on request and responds to feedback thoughtfully. HubermanGPT adapts the length of its responses based on the user's input, providing concise answers for brief comments or deeper explanations for detailed inquiries.

   Please respond to the following question: "How does neuroplasticity work?" [/INST]'''
   ```

4. **Example Prompts:**

    Here are some example prompts you can use with HubermanGPT:

    General Neuroscience:
    "Explain the concept of neuroplasticity in simple terms."

    Health and Wellness:
    "What are some science-based tools for improving sleep quality?"

    Mental Health:
    "How does psilocybin affect the brain and mental health?"

    Learning and Memory:
    "What are the best ways to improve focus and memory?"
