# Generalized real-time state of health estimation for lithium-ion batteries using simulation-augmented multi-objective dual-stream fusion of multi-Bi-LSTM-attention

This repository contains the implementation of a novel deep learning framework for generalized real-time state-of-health (SOH) estimation of lithium-ion batteries. The proposed architecture leverages a multi-loss optimized dual-stream fusion of attention-integrated multi-BiLSTM (multi-ABiLSTM) networks, designed to achieve robust performance across heterogeneous battery datasets with varying discharge profiles.

## ⚙️ Key Components

- 🔹 Energy Discrepancy–Aware Variable Cycle Length Synchronization (EDVCS): Synchronizes variable-length discharge cycles across heterogeneous datasets by compensating for energy flow discrepancies, ensuring uniform temporal representation.

- 🔹 Grid Encoding (GE): Transforms multi-cycle battery data into structured grid representations, enhancing cross-domain feature consistency and network interpretability.

- 🔹 ODS-multi-ABiLSTM Stream: Extracts dynamic degradation features from overlapped discharge segments to capture intra-cycle variations and fine-grained SOH patterns.

- 🔹 PCS-multi-ABiLSTM Stream: Integrates historical SOH evolution across previous cycles to model long-term degradation dependencies.

- 🔹 Dual-Stream Fusion Network (DSFN): Combines complementary features from both ODS and PCS streams using attention-driven fusion for enhanced estimation accuracy and generalization.

- 🔹 Simulation-Based Data Augmentation: Incorporates physics-based battery simulation models to generate synthetic yet realistic training samples, improving robustness to unseen battery profiles.

## 🧠 Abstract
To maintain the safe and reliable operation of lithium-ion batteries and manage their timely replacement, accurate state of health (SOH) estimation is critically important. This paper presents a novel deep-learning framework based on multi-loss optimized dual stream fusion of attention integrated multi-Bi-LSTM networks (multi-ABi-LSTM), for generalized real-time SOH estimation of lithium-ion batteries. Battery sensor data is first preprocessed utilizing novel energy discrepancy aware variable cycle length synchronization and grid encoding schemes to achieve generalizability considering battery sets with different discharge profiles and then passed through two parallel networks: overlapped data splitting (ODS)-based attention integrated multi-Bi-LSTM network (ODS-multi-ABi-LSTM) and past cycles’ SOHs (PCSs)-based attention integrated multi-Bi-LSTM (PCS-multi-ABi-LSTM) network. The complementary features extracted from these two networks are effectively combined by a proposed fusion network to achieve high SOH estimation accuracy. Furthermore, a lithium-ion battery simulation model is employed for data augmentation during training, enhancing the generalizability of the proposed data-driven model. The suggested technique outperforms previous methods by a remarkable margin achieving 0.716% MAPE, 0.005 MAE, 0.653% RMSE, and 0.992 on a combined dataset consisting of four different battery sets with varying specifications and discharge profiles, indicating its generalization capability. Appliances using lithium-ion batteries can adopt the proposed SOH prediction framework to predict battery health conditions in real-time, ensuring operational safety and reliability.

 ## 🔁 Install and Compile the Prerequisites

- python 3.8
- PyTorch >= 1.8
- pywavelets
- neurokit2
- Python packages: numpy, pandas, scipy

## 📌 Citation

If you find this work useful, please cite using:

```
@article{TASNIM2025100870,
title = {Generalized real-time state of health estimation for lithium-ion batteries using simulation-augmented multi-objective dual-stream fusion of multi-Bi-LSTM-attention},
journal = {e-Prime - Advances in Electrical Engineering, Electronics and Energy},
volume = {11},
pages = {100870},
year = {2025},
issn = {2772-6711},
doi = {https://doi.org/10.1016/j.prime.2024.100870},
url = {https://www.sciencedirect.com/science/article/pii/S2772671124004479},
author = {Jarin Tasnim and Md. Azizur Rahman and Md. Shoaib Akhter Rafi and Muhammad Anisuzzaman Talukder and Md. Kamrul Hasan}
}
```
