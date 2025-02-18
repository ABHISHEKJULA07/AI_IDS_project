# AI-Based Network Intrusion Detection System (NIDS) using Machine Learning and Deep Learning

## Overview

This repository contains an AI-based Network Intrusion Detection System (NIDS) that leverages Machine Learning (ML) and Deep Learning (DL) techniques to detect and prevent malicious activities in network traffic. The system is designed to identify various types of network attacks, such as DDoS, port scanning, malware, and more, by analyzing network packet data.

The project aims to provide a robust, scalable, and efficient solution for network security, utilizing state-of-the-art ML/DL algorithms to improve detection accuracy and reduce false positives.


## Features

- **Real-time Network Traffic Analysis**: Monitors and analyzes network traffic in real-time to detect anomalies.
- **Multiple ML/DL Models**: Implements various machine learning and deep learning models, including:
  - Random Forest
  - Support Vector Machines (SVM)
  - Neural Networks (CNN, RNN, LSTM)
  - Autoencoders for anomaly detection
- **Dataset Support**: Compatible with popular network intrusion datasets such as:
  - [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
  - [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
  - [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- **Customizable**: Easily extendable to support new datasets, models, or features.
- **Visualization**: Includes tools for visualizing network traffic and detection results.


####Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- Scikit-learn
- Pandas, NumPy, Matplotlib
- Jupyter Notebook (optional, for experimentation)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Based-Network-IDS_ML-DL.git
   cd AI-Based-Network-IDS_ML-DL
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (e.g., CICIDS2017) and place it in the `data/` directory.

4. Run the preprocessing script to prepare the data:
   ```bash
   python scripts/preprocess_data.py
   ```

5. Train the model:
   ```bash
   python scripts/train_model.py
   ```

6. Evaluate the model:
   ```bash
   python scripts/evaluate_model.py
   ```

7. Run the real-time detection system:
   ```bash
   python scripts/real_time_detection.py
   ```

---

## Usage

### Training a Model

To train a model, modify the `scripts/train_model.py` file to select the desired algorithm and dataset. For example:

```python
from models.random_forest import RandomForestModel
from data.load_data import load_cicids2017

# Load dataset
X_train, X_test, y_train, y_test = load_cicids2017()

# Train model
model = RandomForestModel()
model.train(X_train, y_train)

# Save model
model.save("saved_models/random_forest_cicids2017.pkl")
```

### Real-Time Detection

To run the real-time detection system, use the `scripts/real_time_detection.py` script. Ensure that the trained model is saved and loaded correctly.

```python
from models.random_forest import RandomForestModel
from detection.real_time import RealTimeDetector

# Load trained model
model = RandomForestModel.load("saved_models/random_forest_cicids2017.pkl")

# Start real-time detection
detector = RealTimeDetector(model)
detector.start()
```

---

## Results

The system has been tested on the CICIDS2017 dataset, achieving the following performance metrics:

| Model            | Accuracy | Precision | Recall  | F1-Score |
|------------------|----------|-----------|---------|----------|
| Random Forest    | 99.2%    | 98.9%     | 99.1%   | 99.0%    |
| CNN              | 98.7%    | 98.5%     | 98.6%   | 98.5%    |
| LSTM             | 99.0%    | 98.8%     | 98.9%   | 98.8%    |

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Special thanks to the creators of the CICIDS2017, NSL-KDD, and UNSW-NB15 datasets.
- Inspired by research papers and open-source projects in the field of network security and AI.

---

## Contact

For questions or feedback, please reach out to:

- ABHISHEK JULA  
  Email: your- abhishekjula018@gmail.com


