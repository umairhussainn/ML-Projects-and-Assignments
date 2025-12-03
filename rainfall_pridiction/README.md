#  Rainfall Prediction - Melbourne

Machine learning project to predict rainfall in Melbourne area using Australian weather data.

##  Results
- **84% accuracy** with Random Forest
- **7,500+ weather observations** (2008-2017)
- **Handles imbalanced data** (22% rain days)

##  Quick Start

### Install & Run
```bash
# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# Run the model
python rainfall_prediction.py
```

##  What's Inside
- `rainfall_prediction.py` - Complete ML pipeline
- Weather data from Melbourne area (3 locations)
- Random Forest vs Logistic Regression comparison

## 🔬 Features Used
- **Temperature**: Min/Max daily temperatures
- **Humidity**: Morning and afternoon humidity
- **Pressure**: Atmospheric pressure readings
- **Wind**: Speed and direction data
- **Seasons**: Engineered from dates

##  Model Performance

| Model | Accuracy | Best For |
|-------|----------|----------|
| Random Forest | **84%** | Overall prediction |
| Logistic Regression | 83% | Interpretability |

### Key Insights
- **Most important**: Humidity and pressure
- **Seasonal patterns**: Winter = more rain
- **Class balance**: 78% no rain, 22% rain

##  Technical Stack
- Python, Pandas, Scikit-learn
- Pipeline with preprocessing
- GridSearchCV for optimization
- Stratified cross-validation

##  Data Source
Australian Bureau of Meteorology via [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

##  Skills Demonstrated
 Data preprocessing & cleaning  
 Feature engineering  
 Model selection & tuning  
 Pipeline creation  
 Performance evaluation  
 Handling imbalanced data  

---

