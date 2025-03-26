# FashionGAN - CSE465 Spring 2025

## 📌 Project Overview
This project trains a **Convolutional Neural Network (CNN)** on **Fashion MNIST** with **Data Augmentation** and **5-Fold Cross-Validation**.

## 👥 Contribution Table
| Name  | Role |
|-------|------|
| Sarith Chowdhury | Data Augmentation, Model Training, Documentation |
| Amir Hamza Alvi | Model Optimization, Testing |

## 🔄 Data Augmentation
We used **ImageDataGenerator** to generate **18,000 new images** by:
- Rotating images (±15°)
- Zooming (±15%)
- Horizontal flipping
- Brightness adjustment

## 📊 Final Results (5-Fold Cross Validation)
| Fold | Accuracy | Precision | Recall | F1-Score |
|------|----------|----------|--------|----------|
| 1  | 92.1%  | 91.8% | 92.3% | 92.0% |
| 2  | 92.3%  | 92.0% | 92.5% | 92.2% |
| 3  | 92.0%  | 91.6% | 92.2% | 91.9% |
| 4  | 92.4%  | 92.1% | 92.6% | 92.3% |
| 5  | 92.2%  | 91.9% | 92.4% | 92.1% |
