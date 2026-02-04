# Topic: A Privacy-Preserving Federated Learning Framework for Context-Aware Electricity Theft Detection
This repository contains the official implementation of a Privacy-Preserving Federated Learning (FL) framework designed to detect electricity theft in smart grids. The system leverages Self-Attention mechanisms and Differential Privacy to balance high detection accuracy with consumer data protection.

## 🧐 Overview
Electricity theft results in billions of dollars in losses annually. While AI models can detect theft patterns, they often require access to granular consumption data, which reveals sensitive details about a user's lifestyle.

SecureGrid solves this by:

- Keeping data localized on smart meters via Federated Learning.

- Adding mathematical noise via Differential Privacy to prevent reverse-engineering of user habits.

- Incorporating Contextual Features (like temperature) to reduce false alarms.

## ✨ Key Features
- Federated Averaging (FedAvg): Collaborative training without raw data exchange.

- Attention Mechanism: A Transformer-based layer that prioritizes "peak hours" for more accurate detection.

- Context-Awareness: Integrated weather and temperature data to differentiate between holiday/seasonal patterns and actual theft.

- Class Imbalance Handling: Localized SMOTE implementation to address the scarcity of "thief" data in real-world scenarios.

## 🏗 Architecture
The system follows a Client-Server topology:

- Client (Smart Meter): Performs local training, applies SMOTE for balancing, and injects Gaussian noise for Differential Privacy.

- Server (Utility Provider): Aggregates local weights to update the global detection model without ever seeing individual usage logs.

## 🛠 Technical Contributions

- Privacy Trilemma: Successfully balances Privacy, Accuracy, and Communication efficiency.

- Temporal Weighting: Uses Attention to identify suspicious power drops during high-demand periods.

- DP-Enforced Weights: Implements gradient clipping and noise injection to meet $(\epsilon, \delta)$-differential privacy standards.

## 📊 Results

The model achieves a robust F1-score even under high privacy constraints ($\epsilon = 0.01$).

<img width="569" height="650" alt="image" src="https://github.com/user-attachments/assets/d6e1683d-ec21-4c45-9165-cd9ffbd224b0" />

## 📈 Vizualization

<img width="1905" height="561" alt="image" src="https://github.com/user-attachments/assets/d8f63c6d-b5d1-49af-9671-e2b818f37116" />



