# Assessing Open LLMs’ Ability to Identify Biomedical Taxonomic Relationships  
### SNOMED CT-Based Experimental Evaluation — Code Repository

This repository contains the code used in the article:  
**“Assessing Open LLMs’ Ability to Identify Biomedical Taxonomic Relationships: A SNOMED CT-Based Experimental Evaluation”**.

The software implements the full experimental pipeline described in the manuscript, enabling reproducible evaluation of multiple Large Language Models (LLMs) on the task of identifying taxonomic *“is-a”* relationships in **SNOMED CT**.

---

## 🔍 Project Overview

The goal of this repository is to evaluate whether **open, general-purpose LLMs** can correctly determine if two biomedical concepts are taxonomically related.

This codebase:

- Executes **3,300 SNOMED CT concept-pair queries** across five test archetypes (R, R⁺, R⊕, R⁻¹, (R⁺)⁻¹).  
- Sends each query to **multiple LLMs** using optimized prompts and temperature settings.  
- Enforces **strict boolean outputs** (`true` / `false`) with schema-constrained decoding.  
- Applies an internal **self-refinement loop** to correct malformed outputs.  
- Stores **all predictions and metadata** for statistical analysis.  
- Ensures full **reproducibility** through standardized configurations.

---

## 🧠 Scientific Context

The associated study demonstrates that:

- General-purpose LLMs can detect many biomedical taxonomic relations **without domain-specific fine-tuning**.
- Chain-of-Thought reasoning significantly improves performance.
- Models excel on direct/transitive “is-a” relations but struggle on reversed directionality cases.

---

## 📦 Installation

### Requirements
- Python ≥ 3.12  
- Ollama  25.02
- DSPy  
- (Optional) CUDA-supported GPU  

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start Ollama server
```bash
ollama serve
```

### Pull models
```bash
ollama pull deepseek-r1
ollama pull gemma2
ollama pull qwen2.5
ollama pull phi3
```

---

## 🧪 Dataset Description

The dataset contains **3,300 SNOMED CT concept pairs**, divided into five archetypes:

| Datacasetype | Symbol | Description | Expected label |
|---|--------|-------------|----------------|
| 0 | **R** | Direct “is-a” relations | True |
| 1 |**R⁺** | Transitive relations | True |
| 2 |**¬R⁺** | Taxonomy-unrelated random pairs | False |
| 3 |**R⁻¹** | Reversed direct relation | False |
| 4 |**(R⁺)⁻¹** | Reversed transitive relation | False |

---

## ▶️ How to Run

### Evaluate a single model
```bash
python main.py 
```
All results are saved under `data/results/`.

---

## 📊 Outputs

Example row:

| model | child | parent | prompt | temp | prediction | gold | correct |
|-------|--------|--------|--------|-------|------------|-------|----------|
| deepseek-r1 | Burn scar | Scar | p2 | 0.25 | true | true | 1 |

---

## ⚙️ Configuration
Example of config file:
```
{
  "workflow": "evaluate",
  "MODEL_NAMES": [
    {"model": "ollama/gemma2:9b", "temperature": 0.25},
    {"model": "ollama/deepseek-r1:14b", "temperature": 0.5},
    {"model": "ollama/qwen2.5:14b", "temperature": 0.75}
  ],
  "TASKS": [      
      "Determines if ${childdesc} is a type of ${parentdesc}",
      "Identify if there is a taxonomic relationship between ${childdesc} and ${parentdesc} fields"
  ],
  "OPTIMIZERS": [
      [ "dspy.teleprompt.MIPROv2", {"auto": "light", "num_threads": 24} ]
  ]  
}
```

You may customize:

- Model name  
- Prompt type  
- Temperature  

---

## 📄 License

Licensed under the **Apache License 2.0**.  
A `NOTICE` file is included to preserve attribution to the scientific article.
