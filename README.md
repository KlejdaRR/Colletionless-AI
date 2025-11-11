# Collaborative Classroom for Collectionless AI

*A Master's Thesis Project on Emergent Collective Intelligence in Neural Networks*

## Overview

This project implements a **Collaborative Classroom** system that demonstrates core principles of **Collectionless AI** - where multiple neural networks learn collectively from a data stream without storing historical data. The system shows how emergent collective intelligence can surpass individual performance through peer-to-peer knowledge transfer.

## Key Features

- **Collectionless Learning**: Online processing without data storage
- **Multi-Agent Collaboration**: 8 neural networks learning together
- **Two-Phase Training**: Teacher-guided → Peer-driven learning
- **Knowledge Distillation**: Soft target learning between agents
- **Emergent Intelligence**: Collective performance > Individual performance

## Experimental Results

| Metric | Value |
|--------|-------|
| Final Class Accuracy | **97.33%** |
| Final Best Student Accuracy | **95.45%** |
| **Performance Improvement** | **+1.88%** |
| Breakthrough Step | Step 1000 |
| Teacher Usage | 1000/1000 calls |

## Project Structure
Collectionless-AI/
├── Models/
│ ├── Teacher.py # Teacher model with limited supervision
│ ├── Student.py # Individual student neural network
│ └── init.py
├── collaborative_classroom.py # Main collaborative system
├── experiments.py # Experiment runner and visualization
├── requirements.txt # Python dependencies
└── README.md # This file

## Student Neural Network
3-layer fully connected network (784-128-64-10)
ReLU activations with dropout
LogSoftmax output for classification

## Teacher
Provides supervised labels for limited steps (1000 calls)
Enables initial knowledge bootstraping

## Learning Process
Phase 1: Teacher-Guided (Steps 0-2000)
All students learn from teacher's true labels
Individual models develop diverse representations
Best student identified via moving average loss

Phase 2: Peer-Driven (Steps 2000-5000)
Best student becomes teacher via knowledge distillation
Other students learn from soft targets
Collective intelligence emerges through collaboration

## Requirements
Python 3.8+
PyTorch 1.9+
torchvision
matplotlib
numpy

## Research Context
This work aligns with Collectionless AI principles as described in:
"Collectionless Artificial Intelligence" by Gori & Melacci
"Continual Learning through Hamilton Equations" by Betti et al.

## Key Findings
Collective Superiority: 8-agent system achieves +1.88% over best individual
Early Emergence: Collective advantage appears by step 500
Sustained Performance: Improvement maintained throughout training
Knowledge Transfer: Effective peer learning without external supervision

Author: Klejda Rrapaj
MSc in Artificial Intelligence and Automation Engineering
University of Siena

📝 License
This project is part of Master's Thesis research at University of Siena, Department of Information Engineering and Mathematics
