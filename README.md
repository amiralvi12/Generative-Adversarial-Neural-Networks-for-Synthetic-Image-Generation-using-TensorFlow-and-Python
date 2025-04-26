# Generative-Adversarial-Neural-Networks-for-Synthetic-Image-Generation-using-TensorFlow-and-Python

For running FashionGaN-im.ipynb we have to locate in this folder also a images folder so that when we run the file in training image will generate in images folder 

 Network Architecture:
 self.net = nn.Sequential(
    nn.Conv2d(1, 32, 3, 1),        # Conv Layer: input 1 channel -> 32 channels, kernel 3x3, stride 1
    nn.ReLU(),                     # Activation
    nn.Conv2d(32, 64, 3, 1),        # Conv Layer: 32 -> 64 channels
    nn.ReLU(),                     # Activation
    nn.MaxPool2d(2),                # Max Pooling 2x2
    nn.Flatten(),                  # Flatten
    nn.Linear(9216, 128),           # Fully connected: 9216 -> 128
    nn.ReLU(),                     # Activation
    nn.Linear(128, 10)              # Fully connected: 128 -> 10 (10 classes for FashionMNIST)
)
![Screenshot 2025-04-26 235501](https://github.com/user-attachments/assets/9442eaf4-7c35-4836-8913-0d9c11a12e83)
Confusion matrix and results;
 precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000         1
           1     1.0000    0.5000    0.6667         2
           2     0.5000    1.0000    0.6667         1

    accuracy                         0.7500         4
   macro avg     0.8333    0.8333    0.7778         4
weighted avg     0.8750    0.7500    0.7500         4




