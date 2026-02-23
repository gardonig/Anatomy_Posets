# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

## Overview
Integrates explicit anatomical knowledge into deep learning-based medical image segmentation using Partially Ordered Sets (Posets). Addresses errors like anatomically impossible organ positions from models such as TotalSegmentator and VIBESegmentator.

## Goal
Bridge classical anatomical knowledge with modern deep learning to create accurate and anatomically coherent segmentation models.

## Workflow
1. **Clinical Knowledge Extraction** – interactive GUI for clinicians to encode spatial relations.
2. **Post-Processing Correction** – enforce spatial rules to clean model outputs.
3. **Weakly-Supervised Training** – use cleaned outputs as pseudo-labels to train 3D networks.

## Related Models
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [VIBESegmentator](https://github.com/robert-graf/VIBESegmentator/tree/main)
- [Segment Anything Model 3 (SAM3)](https://ai.meta.com/research/sam3/)

## Key References
- Your other Left! (MICCAI 2025)
- KG-SAM (2025)
- Learning to Zoom with Anatomical Relations (NeurIPS 2025)
- 3D Spatial Priors (STIPPLE)
