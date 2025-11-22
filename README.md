# Based on Text Company Type Classification Model Development

A machine learning project to classify company descriptions into **11 industry categories** for investment analysis.

## 🚀 Key Results
*   **Test Accuracy:** **86.49%** (Target: >80%)
*   **Stability:** 86.38% (5-Fold Cross-Validation)
*   **Highlight:** Successfully handles severe class imbalance using weighted loss.

## 🛠️ Tech Stack
*   **Preprocessing:** Character-level cleaning & tokenization.
*   **Features:** TF-IDF (N-gram 1-2).
*   **Model:** Logistic Regression (GridSearch optimized).
*   **Pipeline:** `sklearn.pipeline` used to prevent data leakage.

## 🏃 Quick Start

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Model**
    Place your dataset  in the root folder and run:
    ```bash
    python main.py
    ```
    *This will train the model, generate evaluation plots, and save the best model as `.pkl`.*

3.  **Inference**
    ```python
    import joblib
    model = joblib.load('best_company_classifier_pipeline.pkl')
    print(model.predict(["Company description text here..."]))
    ```
