# Calories_prediction
## 🚀 Workflow

### 1. Import Libraries
- NumPy, Pandas, Matplotlib, Seaborn  
- Scikit-learn (Label Encoding, KFold CV, Metrics)  
- XGBoost & LightGBM Regressors  

### 2. Load Data
- Reads `train.csv` and `test.csv`  
- Drops unnecessary `id` column  

### 3. Exploratory Data Analysis (EDA)
- `.info()` → check data types  
- `.isnull().sum()` → check missing values  
- Visualizations with **Matplotlib** and **Seaborn**  

### 4. Preprocessing
- Encode categorical features using `LabelEncoder`  
- Handle missing values (if any)  

### 5. Modeling
Trains regression models:
- **XGBoost Regressor**  
- **LightGBM Regressor**  
- Uses **K-Fold Cross Validation** for evaluation  

### 6. Evaluation Metrics
- Mean Absolute Error (**MAE**)  
- Mean Squared Log Error (**MSLE**)  

---

## 📊 Results
- Compares performance of **XGBoost** and **LightGBM** models  
- Best-performing model can be used to generate predictions for `test.csv`  
