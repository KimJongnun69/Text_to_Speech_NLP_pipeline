# Speech-to-Text NLP Pipeline

This project implements and compares two Automatic Speech Recognition (ASR) pipelines for converting speech audio into text transcripts.

The project evaluates transcription quality across two speech recognition approaches:

1. OpenAI Whisper (Pretrained Baseline Model)  
2. Fine-tuned Wav2Vec2 Transformer Model  

The objective is to compare model performance using **Word Error Rate (WER)** and build a reproducible end-to-end speech recognition pipeline.

---

# Repository Structure

```text
Text_to_Speech_NLP_pipeline/

│── README.md

└── Consolidated_resources/
      │
      ├── datasets/
      │     └── Dataset source / dataset link
      │
      ├── notebooks/
      │     ├── fixed_main_script.ipynb
      │     └── fixed_execution_script.ipynb
      │
      ├── pdfs_of_notebooks/
      │     ├── main_script.pdf
      │     └── fixed_execution_script.pdf
      │
      └── results/
            └── wav2vec_eval_results.csv
```

---

# Dataset Used

Dataset used:

**LibriSpeech ASR Dataset**

The dataset was downloaded from the LibriSpeech speech corpus and processed for speech-to-text training and evaluation.

Dataset source:

https://www.openslr.org/12

---

# Dataset Structure

The downloaded dataset follows the following hierarchical structure:

```text
LibriSpeech/

└── test-clean/
      │
      ├── 61/
      │     └── 70968/
      │            ├── 61-70968-0000.flac
      │            ├── 61-70968-0001.flac
      │            ├── 61-70968.trans.txt
      │
      ├── 1089/
      │     └── 134686/
      │            ├── 1089-134686-0000.flac
      │            ├── 1089-134686-0001.flac
      │            ├── 1089-134686.trans.txt
      │
      └── ...
```

### Explanation of Structure

Each folder corresponds to:

```text
speaker_id/
      chapter_id/
```

Example:

```text
61/
   70968/
```

Inside each chapter folder:

### Audio Files

```text
61-70968-0000.flac
61-70968-0001.flac
61-70968-0002.flac
```

These are speech recordings.

### Transcript File

```text
61-70968.trans.txt
```

Contains mappings between audio file and transcript.

Example:

```text
61-70968-0000 HE EXPECTED TO FIND PEACE THERE
61-70968-0001 THE MACHINE LEARNING MODEL PERFORMED WELL
```

The notebook parses this file and creates a structured dataset.

---

# Processed Dataset Format

After preprocessing, the raw dataset is converted into a structured dataframe.

Format:

```text
audio_path | transcript
```

Example:

```text
librispeech_test_clean/61/70968/61-70968-0000.flac

HE EXPECTED TO FIND PEACE THERE
```

The pipeline automatically extracts audio-transcript pairs from the dataset.

Approximately **400 audio samples** were used for experimentation.

---

# Main Script (Training Pipeline)

File:

```text
Consolidated_resources/notebooks/fixed_main_script.ipynb
```

This notebook contains the complete model development pipeline.

---

## Workflow

```text
Download and extract LibriSpeech dataset
                ↓
Read transcript files (.trans.txt)
                ↓
Create structured audio-transcript dataset
                ↓
Load Whisper pretrained model
                ↓
Generate Whisper transcriptions
                ↓
Calculate Whisper Word Error Rate
                ↓
Load pretrained Wav2Vec2 model
                ↓
Convert audio to 16 kHz
                ↓
Prepare transformer input pipeline
                ↓
Fine-tune Wav2Vec2 model
                ↓
Evaluate model performance
                ↓
Save trained model
                ↓
Export evaluation results
```

---

## What happens in Main Script

The notebook performs:

### Dataset Preprocessing

- Reads `.flac` audio files
- Reads `.trans.txt` transcript files
- Creates structured dataframe

### Whisper Baseline Evaluation

- Loads pretrained Whisper model
- Generates transcript predictions
- Calculates WER on evaluation data

### Wav2Vec2 Fine-Tuning

- Loads pretrained transformer model
- Converts audio to required 16 kHz format
- Tokenizes audio and text labels
- Fine-tunes model using transfer learning
- Evaluates final performance

---

# Models Used

## 1. Whisper Baseline

Model:

OpenAI Whisper

Purpose:

Used as baseline model for comparison.

Loaded using:

```python
model = whisper.load_model("small")
```

Process:

```text
Input audio
      ↓
Whisper transcription
      ↓
Normalize output
      ↓
Calculate WER
```

This model was used without additional training.

---

## 2. Fine-Tuned Wav2Vec2 Model

Base model:

facebook/wav2vec2-base-960h

Approach used:

Transfer Learning + Fine-Tuning

Training pipeline:

```text
Load pretrained model
        ↓
Freeze feature encoder
        ↓
Prepare speech dataset
        ↓
Convert audio to 16 kHz
        ↓
Create tokenized labels
        ↓
Fine-tune transformer layers
        ↓
Evaluate model
```

Audio preprocessing:

```text
16 kHz resampling
Mono conversion
Padding
Tensor conversion
```

This model was trained and fine-tuned on the dataset.

---

# Execution Script (Inference Pipeline)

File:

```text
Consolidated_resources/notebooks/fixed_execution_script.ipynb
```

This notebook is used to test the models on unseen audio samples.

---

## Workflow

```text
Upload audio file
        ↓
Play uploaded audio
        ↓
Load Whisper model
        ↓
Generate Whisper transcript
        ↓
Load saved Wav2Vec2 model
        ↓
Generate Wav2Vec2 transcript
        ↓
Input actual transcript manually
        ↓
Calculate Whisper WER
        ↓
Calculate Wav2Vec2 WER
        ↓
Compare results
```

---

# How to Run Execution Script

Open:

```text
fixed_execution_script.ipynb
```

Steps:

### Step 1

Upload any speech audio file.

### Step 2

Notebook plays uploaded audio.

### Step 3

Both models generate transcripts.

Example:

```text
Whisper Output:

HELLO MY NAME IS GIRISH
```

```text
Wav2Vec2 Output:

HELLO MY NAME IS GIRISH
```

### Step 4

Enter correct transcript manually.

Example:

```text
HELLO MY NAME IS GIRISH
```

### Step 5

WER is automatically calculated.

Example:

```text
Whisper WER: 0.06

Wav2Vec2 WER: 0.01
```

---

# Results

Results generated during evaluation are stored inside:

```text
Consolidated_resources/results/
```

Stored file:

```text
wav2vec_eval_results.csv
```

The results file contains:

```text
Audio file path
Actual transcript
Predicted transcript
WER score
```

---

# Evaluation Metric

Metric used:

**Word Error Rate (WER)**

Formula:

```text
WER = (Substitutions + Insertions + Deletions) / Total Words
```

Interpretation:

```text
Lower WER = Better transcription accuracy
```

---

# Technologies Used

```text
Python
PyTorch
Transformers
Librosa
OpenAI Whisper
HuggingFace Transformers
Pandas
NumPy
JiWER
```

---

# Summary

Implemented a complete speech recognition pipeline.

Project includes:

Dataset preprocessing from LibriSpeech corpus  
Whisper pretrained baseline evaluation  
Transformer-based Wav2Vec2 fine-tuning  
Audio inference pipeline for custom audio  
Word Error Rate evaluation  
Comparison between two ASR architectures  
Reproducible project structure
