# cs5365-lora-project
LoRA: Low-Rank Adaptation of Large Language Models—Re-implementation
Course: CS5365—Deep Learning
Institution: University of Texas at El Paso
Group Members: Veronica Aragon, Angela Martinez

Introduction
This repository contains our re-implementation of LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022). The paper introduces a parameter-efficient fine-tuning method that freezes the weights of a pretrained model and injects trainable low-rank matrices into each layer of the Transformer architecture. This dramatically reduces the number of trainable parameters while maintaining performance competitive with full fine-tuning.
Our goal is to reproduce the core rank ablation experiment from the paper, studying how the choice of rank r affects adaptation quality and training efficiency across different tasks.
