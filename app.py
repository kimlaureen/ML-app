import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
import io
import joblib


# Page settings
PAGE_TITLE = "Train your own Machine Learning Model"
PAGE_LAYOUT = "wide"

# Sample datasets available for use
SAMPLE_DATASETS = ["iris", "tips", "titanic", "planets"]
DATASET_SOURCE_OPTIONS = ["Sample Datasets", "Upload CSV"]

# Hyperparameter ranges and defaults for Random Forest
RF_N_ESTIMATORS_RANGE = (10, 500, 100, 10)    # (min, max, default, step)
RF_MAX_DEPTH_RANGE = (1, 50, 10, 1)
RF_MIN_SAMPLES_SPLIT_RANGE = (2, 10, 2, 1)
RF_MIN_SAMPLES_LEAF_RANGE = (1, 10, 1, 1)
RF_MAX_FEATURES_OPTIONS = ["sqrt", "log2", "None"]

# Hyperparameter ranges for Linear Model (Logistic Regression regularization for classification)
LM_C_RANGE = (0.01, 10.0, 1.0, 0.01)

# Hyperparameter ranges for KNN
KNN_N_NEIGHBORS_RANGE = (1, 30, 5, 1)
KNN_WEIGHT_OPTIONS = ["uniform", "distance"]

# Test set size (slider range and default as percentage)
TEST_SIZE_RANGE = (10, 50, 20, 5)  # values in percent



# Set page title and configuration
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
st.title(PAGE_TITLE)

# Initialize session state variables
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "target_type" not in st.session_state:
    st.session_state.target_type = None

@st.cache_data
def load_seaborn_dataset(dataset_name):
    if dataset_name in SAMPLE_DATASETS:
        return sns.load_dataset(dataset_name)
    return None

# Sidebar: Dataset selection
with st.sidebar:
    st.header("Dataset Selection")
    dataset_source = st.radio("Select data source", DATASET_SOURCE_OPTIONS)

    if dataset_source == "Sample Datasets":
        dataset_name = st.selectbox("Select Dataset", SAMPLE_DATASETS)
        if st.button("Load Dataset"):
            st.session_state.dataset = load_seaborn_dataset(dataset_name)
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None and st.button("Load Uploaded Dataset"):
            st.session_state.dataset = pd.read_csv(uploaded_file)

    if st.button("Reset Results"):
        st.session_state.trained_model = None
        st.session_state.model_results = None

# Main Application
if st.session_state.dataset is not None:
    df = st.session_state.dataset

    st.header("Dataset Preview")
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head())

    st.subheader("Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    with st.form(key="feature_selection_form"):
        st.header("Feature Selection and Model Configuration")
        col1, col2 = st.columns(2)

        # --- Feature Selection ---
        with col1:
            st.subheader("Features")
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            selected_numerical = st.multiselect("Select Numerical Features", numerical_cols, default=numerical_cols[:2])
            selected_categorical = st.multiselect("Select Categorical Features", categorical_cols, default=categorical_cols[:1])
            target_variable = st.selectbox("Select Target Variable", df.columns.tolist(), index=len(df.columns)-1)

            # Determine task type: Classification vs Regression
            n_unique = df[target_variable].nunique()
            is_categorical = (n_unique <= 10) or (df[target_variable].dtype == 'object')
            task_type = st.radio("Task Type", ["Classification", "Regression"], index=0 if is_categorical else 1)

        # --- Model Configuration ---
        with col2:
            st.subheader("Model Configuration")
            test_size_slider = st.slider("Test Set Size (%)", *TEST_SIZE_RANGE)
            test_size = test_size_slider / 100.0

            model_type = st.selectbox("Select Model Type", ["Random Forest", "Linear Model", "KNN"])

            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Estimators", *RF_N_ESTIMATORS_RANGE)
                max_depth = st.slider("Max Depth", *RF_MAX_DEPTH_RANGE)
                min_samples_split = st.slider("Min Samples Split", *RF_MIN_SAMPLES_SPLIT_RANGE)
                min_samples_leaf = st.slider("Min Samples Leaf", *RF_MIN_SAMPLES_LEAF_RANGE)
                max_features_option = st.selectbox("Max Features", RF_MAX_FEATURES_OPTIONS)
                max_features = None if max_features_option == "None" else max_features_option

                rf_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features
                }

            elif model_type == "Linear Model":
                if task_type == "Classification":
                    c_param = st.slider("C (Regularization)", *LM_C_RANGE)
                    lm_params = {"C": c_param}
                else:
                    lm_params = {}

            elif model_type == "KNN":
                n_neighbors = st.slider("Number of Neighbors", *KNN_N_NEIGHBORS_RANGE)
                weight = st.selectbox("Weight Function", KNN_WEIGHT_OPTIONS)
                knn_params = {"n_neighbors": n_neighbors, "weights": weight}

            fit_button = st.form_submit_button("Fit Model")

        if fit_button:
            st.session_state.target_type = task_type
            X_features = selected_numerical + selected_categorical

            if not X_features:
                st.error("Please select at least one feature.")
            else:
                with st.spinner("Training model..."):
                    # 1. Prepare features and target
                    X = df[X_features].copy()
                    y = df[target_variable].copy()

                    # 2. Handle missing values using imputation
                    num_cols = X.select_dtypes(include=[np.number]).columns
                    cat_cols = X.select_dtypes(exclude=[np.number]).columns

                    numeric_imputer = SimpleImputer(strategy='mean')
                    if len(num_cols) > 0:
                        X[num_cols] = numeric_imputer.fit_transform(X[num_cols])

                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    if len(cat_cols) > 0:
                        X[cat_cols] = categorical_imputer.fit_transform(X[cat_cols])

                    # If target has missing values, drop those rows
                    if y.isnull().any():
                        y = y.dropna()
                        X = X.loc[y.index]

                    # 3. Encode categorical features
                    for cat_col in selected_categorical:
                        le = LabelEncoder()
                        X[cat_col] = le.fit_transform(X[cat_col].astype(str))

                    # 4. Process the target variable
                    if task_type == "Classification":
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
                    elif y.dtype == 'object':
                        try:
                            y = y.astype(float)
                        except ValueError:
                            st.error("Target variable contains non-numeric values. Please encode or choose a different target for regression.")
                            st.stop()

                    # 5. Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    # 6. Initialize and train the selected model
                    if model_type == "Random Forest":
                        if task_type == "Classification":
                            model = RandomForestClassifier(**rf_params, random_state=42)
                        else:
                            model = RandomForestRegressor(**rf_params, random_state=42)
                    elif model_type == "Linear Model":
                        if task_type == "Classification":
                            model = LogisticRegression(**lm_params, random_state=42, max_iter=1000)
                        else:
                            model = LinearRegression()
                    elif model_type == "KNN":
                        if task_type == "Classification":
                            model = KNeighborsClassifier(**knn_params)
                        else:
                            model = KNeighborsRegressor(**knn_params)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    results = {
                        'model': model,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'features': X_features
                    }

                    if task_type == "Classification":
                        results['accuracy'] = accuracy_score(y_test, y_pred)
                        results['conf_matrix'] = confusion_matrix(y_test, y_pred)
                        if len(np.unique(y)) == 2:
                            try:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                results['roc'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
                            except Exception:
                                st.warning("ROC Curve could not be generated.")
                    else:
                        results['mse'] = mean_squared_error(y_test, y_pred)
                        results['r2'] = r2_score(y_test, y_pred)
                        results['residuals'] = y_test - y_pred

                    if hasattr(model, 'feature_importances_'):
                        results['feature_importance'] = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        results['feature_importance'] = model.coef_

                    st.session_state.trained_model = model
                    st.session_state.model_results = results

@st.fragment
def show_model_results():
    st.header("Model Results")
    results = st.session_state.model_results
    task_type = st.session_state.target_type

    st.subheader("Performance Metrics")
    if task_type == "Classification":
        st.write(f"Accuracy: {results['accuracy']:.4f}")
    else:
        st.write(f"Mean Squared Error: {results['mse']:.4f}")
        st.write(f"R²: {results['r2']:.4f}")

    st.subheader("Model Visualizations")
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        if task_type == "Regression":
            st.write("Residual Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(results['residuals'], bins=30, alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            st.pyplot(fig)
        else:
            st.write("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(results['conf_matrix'], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

    with fig_col2:
        if 'feature_importance' in results:
            st.write("Feature Importance")
            importance_array = np.array(results['feature_importance'])
            if len(importance_array.shape) > 1 and importance_array.shape[0] > 1:
                importance_array = importance_array.mean(axis=0)
            else:
                importance_array = importance_array.flatten()
            feature_importance = pd.DataFrame({
                'Feature': results['features'],
                'Importance': importance_array
            })
            feature_importance = feature_importance.reindex(
                feature_importance['Importance'].abs().sort_values(ascending=False).index
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)

        if task_type == "Classification" and 'roc' in results:
            st.write("ROC Curve")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(results['roc']['fpr'], results['roc']['tpr'],
                    label=f'AUC = {results["roc"]["auc"]:.3f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)

    st.subheader("Export Model")
    model_info = {
        "Model Type": type(st.session_state.trained_model).__name__,
        "Features": results['features'],
        "Target": task_type
    }
    st.json(model_info)

    model_bytes = io.BytesIO()
    joblib.dump(st.session_state.trained_model, model_bytes)
    model_bytes.seek(0)
    st.download_button("Download Model", model_bytes, file_name="trained_model.pkl")

if st.session_state.model_results is not None:
    show_model_results()
else:
    st.info("Please select a dataset from the sidebar to begin.")
    st.header("Welcome to the ML Model Trainer")
    st.write("""
    This application allows you to:
    1. Select from preloaded datasets or upload your own CSV
    2. Choose features for model training
    3. Configure and train machine learning models with built-in missing value handling
    4. Visualize performance metrics and model results

    Get started by selecting a dataset in the sidebar!
    """)


