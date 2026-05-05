**Course:** CS 5365 – Deep Learning  
**Institution:** University of Texas at El Paso  
**Low-Rank Adaptation of Large Language Models**  
**Group Members:** Veronica Aragon, Angela Martinez  

# Introduction:
This repository contains our re-implementation of **LoRA: Low-Rank Adaptation of Large Language Models** *(Hu et al., ICLR 2022)*. The paper introduces a parameter-efficient fine-tuning method that freezes the weights of a pretrained model and injects trainable low-rank matrices into each Transformer layer — dramatically reducing trainable parameters while maintaining competitive performance.

Our goal is to reproduce the **rank ablation experiment** from Table 5 of the paper, studying how the choice of rank `r` affects adaptation quality and efficiency across GLUE benchmark tasks.

# Data
This project uses the GLUE benchmark dataset, accessed via the Hugging Face `datasets` library.
To download automatically, run:
    python code/train_lora.py --task sst2 --rank 4
The dataset will be downloaded on first run. No manual setup required.
## Datasets Used

| Task | Dataset | Type | Metric |
|------|---------|------|--------|
| SST-2 | Stanford Sentiment Treebank | Sentiment classification | Accuracy |
| MRPC | Microsoft Research Paraphrase Corpus | Paraphrase detection | F1 / Accuracy |

## How Data is Loaded

Both datasets are downloaded automatically on first run:

```bash
python code/train_lora.py --task sst2 --rank 4
python code/train_lora.py --task mrpc --rank 4
```

Hugging Face will cache the datasets locally at `~/.cache/huggingface/datasets/` after the first download.

## Dataset Details

### SST-2
- **Source:** Socher et al. (2013), Stanford NLP
- **Train size:** ~67,000 sentences
- **Validation size:** ~872 sentences
- **Task:** Binary sentiment classification (positive / negative)

### MRPC
- **Source:** Dolan & Brockett (2005), Microsoft Research
- **Train size:** ~3,700 sentence pairs
- **Validation size:** ~408 sentence pairs
- **Task:** Binary paraphrase detection (paraphrase / not paraphrase)

# Notes
- No data files are stored in this repository
- All data is fetched at runtime via the Hugging Face `datasets` library
- Internet connection required on first run
- Subsequent runs use the local cache
  
# References
- Wang, A., et al. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* EMNLP 2018.
- Socher, R., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP 2013.
- Dolan, W. B., & Brockett, C. (2005). *Automatically Constructing a Corpus of Sentential Paraphrases.* IWP 2005.
