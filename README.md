**Course:** CS 5365 – Deep Learning
**Institution:** University of Texas at El Paso
**Low-Rank Adaptation of Large Language Models**
**Group Members:** Veronica Aragon, Angela Martinez

## Introduction:
This repository contains our re-implementation of **LoRA: Low-Rank Adaptation of Large Language Models** *(Hu et al., ICLR 2022)*. The paper introduces a parameter-efficient fine-tuning method that freezes the weights of a pretrained model and injects trainable low-rank matrices into each Transformer layer — dramatically reducing trainable parameters while maintaining competitive performance.

Our goal is to reproduce the **rank ablation experiment** from Table 5 of the paper, studying how the choice of rank `r` affects adaptation quality and efficiency across GLUE benchmark tasks.
