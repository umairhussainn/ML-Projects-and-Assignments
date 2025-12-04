# League of Legends Match Predictor 🎮

A PyTorch-based logistic regression model that predicts League of Legends match outcomes using in-game statistics.

## 📋 Overview

This project implements a binary classification model to predict match outcomes (Win/Loss) based on League of Legends game statistics. Built with PyTorch, it includes data preprocessing, L2 regularization, hyperparameter tuning, and comprehensive performance evaluation.

## ✨ Features

- Data preprocessing with StandardScaler
- Custom PyTorch logistic regression model
- L2 regularization (Ridge regression)
- Model evaluation with confusion matrix, ROC curve, and AUC
- Hyperparameter tuning for optimal learning rate
- Feature importance analysis
- Model saving and loading

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lol-match-predictor.git
cd lol-match-predictor

# Install dependencies
pip install pandas numpy scikit-learn torch matplotlib
```

## 💻 Usage

**Run Jupyter Notebook:**
```bash
jupyter notebook "Final Project League of Legends Match Predictor.ipynb"
```

**Or run Python script:**
```bash
python lol_match_predictor.py
```

## 📊 Dataset

- **Source**: [League of Legends Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv)
- **Features**: In-game statistics (kills, deaths, assists, gold, towers, etc.)
- **Target**: Binary (Win/Loss)
- **Split**: 80% training, 20% testing

## 📈 Results

- **Test Accuracy**: ~XX.XX%
- **AUC Score**: ~0.XX
- **Best Learning Rate**: 0.0X

## 🛠️ Technologies

- Python 3.8+
- PyTorch 2.8.0
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## 📁 Project Structure

```
lol-match-predictor/
├── Final Project League of Legends Match Predictor.ipynb
├── lol_match_predictor.py
├── README.md
└── requirements.txt
```

## 🎯 Key Tasks

1. Data loading and preprocessing
2. Logistic regression model implementation
3. Model training (1000 epochs)
4. L2 regularization
5. Performance visualization
6. Model saving/loading
7. Hyperparameter tuning
8. Feature importance analysis

## 📄 License

MIT License

## 👏 Acknowledgments

Course: Introduction to Neural Networks with PyTorch - IBM Skills Network

---

⭐ Star this repo if you find it helpful!

**Contact**: your.email@example.com | [GitHub](https://github.com/yourusername)