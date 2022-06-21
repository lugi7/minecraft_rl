# minecraft_rl

A repository for RL and vision experiments within MineRL dataset

## Folders:
- **conv_encoder_evaluation**: compares 3 different vision backbones for a CLIP-like objective of determining if two Minecraft video frames are consecutive - mobilenet_v3_small (pretrained and random weights) and a simple convnet
- **decision_transformer**: an implementation of Decision Transformer (Chen et al., 2021) for trajectory modeling; backbone used is Vision Transformer (Dosovitskiy et al., 2021)
- **vit_next_state_correlation**: same task as #1, this time with Vision Transformer as the backbone
- **vit_patch_ordering**: an evaluation of self supervised pretraining objective based on predicting location of scrambled input patches

The non-custom Transformer layers used here come from Andrej Karpathy's minGPT implementation (https://github.com/karpathy/minGPT)
