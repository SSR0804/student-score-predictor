
# Student Score Predictor (Linear Regression)

A simple ML project to predict student exam scores based on study hours.
This project demonstrates data loading, EDA, model training, evaluation, a saved model, and a tiny Streamlit app. 
While using these files and folders make sure to create a folder named 'notebooks' as per the project structure.  

## Learnings
- Data preprocessing & sanity checks
- Train/test split and Linear Regression
- Metrics: MAE, RMSE, R^2
- Visualization (Seaborn/Matplotlib)
- Optional mini deployment with Streamlit

## Project Structure
```
student-score-predictor/
├── data/
│   └── study_hours.csv
|── notebooks/
│   └── 01_eda_and_model.ipynb  # (optional)
├── src/
│   ├── train.py
│   └── utils.py
|── app/
│   └── streamlit_app.py
|── artifacts/          # created after training (plots + metrics + model)
|── README.md
└── requirements.txt
```

## Setup
```bash / cmd
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train & Evaluate
```bash / cmd
python -m src.train
```
Artifacts (plots, metrics, trained model) will be saved in artifacts/.

## Run the Streamlit App (optional)
```bash /cmd
streamlit run app/streamlit_app.py
```

## Results
- Metrics (example): MAE ~ 2-5, R^2 ~ 0.95+ on this toy dataset.
- Plots: scatter_hours_score.png, linear_fit.png.
- Model: artifacts/linear_model.joblib

## Notes
- Dataset is synthetic and for learning only.
- Extend the project by adding polynomial regression, cross-validation, or data from a real source.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
