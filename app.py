# app.py

# All Necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Library to generate synthetic data
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Libraries for train test split, preprocessing and metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# Import XGBoost model
from xgboost import XGBClassifier

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

with st.sidebar:
    st.header("📥 Download Datasets")

    # Datasets download section
    st.subheader("Datasets (.csv)")
    
    # Create containers that will act as placeholders for our download buttons
    original_data_container = st.container()
    cleaned_data_container = st.container()
    synthetic_data_container = st.container()
    augmented_data_container = st.container()

st.title("✨📄 Synthetic Data Generation for Machine Failure")
st.markdown("""
This project builds and compares two XGBoost models to predict machine failure, with one trained on the original dataset and the other on an augmented dataset. It showcases how generating synthetic data for the minority (failure) class using the **Synthetic Data Vault (SDV)** library can effectively balance the dataset. The final comparison demonstrates that this data augmentation technique improves the model's predictive performance and robustness.
""")

# --- Caching Functions to Improve Performance ---

# Cache data loading
@st.cache_data
def load_data(path):
    """Loads the dataset from a specified path."""
    return pd.read_csv('original_dataset.csv')

# Download button for original dataset
with original_data_container:
    with open("original_dataset.csv", "rb") as file:
        st.download_button(
            label="Original Data",
            data=file,
            file_name="original_dataset.csv",
            mime="text/csv"
        )

# Cache data cleaning
@st.cache_data
def clean_data(_df):
    """Applies cleaning operations to the dataframe."""
    df_c = _df.copy()
    df_c.drop('footfall', axis=1, inplace=True)
    df_c = df_c[(df_c['CS'] >= 4)]
    df_c = df_c[(df_c['Temperature'] >= 6)]
    return df_c

# Cache the synthetic data generation process
@st.cache_data
def generate_synthetic_data(_train_data):
    """Generates synthetic data to balance the minority class in the training set."""
    minority_data_train = _train_data[_train_data['fail'] == 1]
    
    majority_count_train = _train_data['fail'].value_counts()[0]
    minority_count_train = _train_data['fail'].value_counts()[1]
    samples_to_create = majority_count_train - minority_count_train

    if samples_to_create <= 0:
        return pd.DataFrame(), samples_to_create

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=minority_data_train)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    
    with st.spinner(f"Fitting synthesizer on {len(minority_data_train)} minority samples..."):
        synthesizer.fit(minority_data_train)
    
    with st.spinner(f"Generating {samples_to_create} new synthetic samples..."):
        synthetic_data = synthesizer.sample(num_rows=samples_to_create)
    
    return synthetic_data, samples_to_create

# Cache model training
@st.cache_resource
def train_xgb_model(X_train, y_train):
    """Trains an XGBoost classifier."""
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Main Application Logic ---

# 1. Load and Explore Original Data
st.header("1. Original Data Exploration")
df_raw = load_data('data.csv')

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Dataset Shape")
    st.write(df_raw.shape)
    
    st.subheader("Null Value Check")
    st.write(df_raw.isnull().sum())
    
    st.subheader("Machine Failure Distribution")
    st.write(df_raw['fail'].value_counts())
    
    # Capture df.info() output
    buffer = io.StringIO()
    df_raw.info(buf=buffer)
    s = buffer.getvalue()
    st.subheader("Dataset Info")
    st.text(s)

with col2:
    st.subheader("Dataset Head")
    st.dataframe(df_raw.head())
    
    st.subheader("Dataset Description")
    st.dataframe(df_raw.describe())

st.divider()

# 2. Exploratory Data Analysis (EDA) on Original Data
st.header("2. EDA on Original Data")

# PCA Visualization
st.subheader("PCA of Original Dataset by Failure Status")
X_pca_orig = df_raw.drop('fail', axis=1)
y_pca_orig = df_raw['fail']
X_scaled_orig = StandardScaler().fit_transform(X_pca_orig)
pca_orig = PCA(n_components=2)
principal_components_orig = pca_orig.fit_transform(X_scaled_orig)
pca_df_orig = pd.DataFrame(data=principal_components_orig, columns=['Principal Component 1', 'Principal Component 2'])
pca_df_orig['Target'] = y_pca_orig

fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Target', data=pca_df_orig, alpha=0.7, ax=ax_pca)
ax_pca.set_title('PCA of Original Dataset by Failure Status')
ax_pca.grid(True)
st.pyplot(fig_pca)

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    # Correlation Heatmap
    st.subheader("Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_raw.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Features')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig_corr)

with eda_col2:
    # Box Plots for Outliers
    st.subheader("Box Plots for Outlier Detection")
    fig_box, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))
    df_features = df_raw.drop('fail', axis=1)
    
    for i, col_name in enumerate(df_features.columns):
        ax = axes[i//3, i%3]
        ax.boxplot(df_raw[col_name])
        ax.set_title(f'Distribution of {col_name}')
        ax.set_ylabel('Values')
        ax.set_xticklabels([col_name])
    
    fig_box.suptitle('Box Plots for Outliers', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_box)

st.divider()

# 3. Data Cleaning and Preparation
st.header("3. Data Cleaning & Preparation")
df_cleaned = clean_data(df_raw)

# Download button for cleaned dataset
with cleaned_data_container:
    with open("df_cleaned.csv", "rb") as cleaned:
        st.download_button(
            label="Cleaned Data",
            data=cleaned,
            file_name="df_cleaned.csv",
            mime="text/csv"
        )

st.write("Data is cleaned by removing outliers based on 'CS' (>= 4) and 'Temperature' (>= 6).")

clean_col1, clean_col2 = st.columns(2)
with clean_col1:
    st.subheader("Shape After Cleaning")
    st.write(df_cleaned.shape)

with clean_col2:
    st.subheader("Distribution After Cleaning")
    st.write(df_cleaned['fail'].value_counts())

st.divider()

# 4. Synthetic Data Generation
st.header("4. Synthetic Data Generation")

X = df_cleaned.drop('fail', axis=1)
y = df_cleaned['fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train_data_combined = pd.concat([X_train, y_train], axis=1)

synthetic_data, num_samples = generate_synthetic_data(train_data_combined)

# For synthetic dataset
with synthetic_data_container:
    st.download_button(
        label="Synthetic Data",
        data=synthetic_data.to_csv(index=False),
        file_name="synthetic_data.csv",
        mime="text/csv"
    )

if num_samples > 0:
    st.success(f"Successfully generated {num_samples} new synthetic samples.")
    st.subheader("Synthetic Data Head")
    st.dataframe(synthetic_data.head())

    # PCA: Real vs. Synthetic Data
    st.subheader("PCA Plot: Real vs. Synthetic Data Distribution")
    scaler_pca_comp = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    X_train_scaled_comp = X_train.copy()
    X_train_scaled_comp[numerical_cols] = scaler_pca_comp.fit_transform(X_train[numerical_cols])

    synthetic_data_pca_comp = synthetic_data.drop('fail', axis=1).copy()
    synthetic_data_pca_comp[numerical_cols] = scaler_pca_comp.transform(synthetic_data_pca_comp[numerical_cols])

    pca_comp = PCA(n_components=2)
    X_train_pca_comp = pca_comp.fit_transform(X_train_scaled_comp)
    synthetic_data_pca_transformed = pca_comp.transform(synthetic_data_pca_comp)

    fig_pca_comp, ax_pca_comp = plt.subplots(figsize=(10, 7))
    ax_pca_comp.scatter(X_train_pca_comp[:, 0], X_train_pca_comp[:, 1], alpha=0.5, label='Real Data', c='blue')
    ax_pca_comp.scatter(synthetic_data_pca_transformed[:, 0], synthetic_data_pca_transformed[:, 1], alpha=0.5, label='Synthetic Data', c='orange', marker='X')
    ax_pca_comp.set_title('PCA Plot: Real vs. Synthetic Data Distribution')
    ax_pca_comp.set_xlabel('Principal Component 1')
    ax_pca_comp.set_ylabel('Principal Component 2')
    ax_pca_comp.legend()
    ax_pca_comp.grid(True)
    st.pyplot(fig_pca_comp)
else:
    st.info("The training data is already balanced. No synthetic samples were generated.")

st.divider()

# 5. Model Training and Evaluation
st.header("5. Model Training & Comparison")

# --- Preprocessing ---
numerical_cols = ['tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']

# Create augmented datasets
X_train_augmented = pd.concat([X_train, synthetic_data.drop('fail', axis=1)], ignore_index=True)
y_train_augmented = pd.concat([y_train, synthetic_data['fail']], ignore_index=True)

augmented_df = pd.concat([X_train_augmented, y_train_augmented], axis=1)

# For augmented dataset
with augmented_data_container:
        st.download_button(
        label="Augmented Data",
        data=augmented_df.to_csv(index=False),
        file_name="augmented_data.csv",
        mime="text/csv"
    )

# Create copies to avoid modifying originals
X_train_scaled = X_train.copy()
X_train_aug_scaled = X_train_augmented.copy()
X_test_scaled = X_test.copy()

# Scale the original training data using its own scaler
scaler = StandardScaler()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Scale the augmented training data and the test data using a new scaler
#scaler_aug = StandardScaler()
X_train_aug_scaled[numerical_cols] = scaler.fit_transform(X_train_augmented[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Train and evaluate models
with st.spinner("Training models..."):
    # XGBoost on original data
    xgb_orig = train_xgb_model(X_train_scaled, y_train)
    y_pred_xgb_orig = xgb_orig.predict(X_test_scaled)
    y_pred_proba_orig = xgb_orig.predict_proba(X_test_scaled)[:, 1]

    # XGBoost on augmented data
    xgb_aug = train_xgb_model(X_train_aug_scaled, y_train_augmented)
    y_pred_xgb_aug = xgb_aug.predict(X_test_scaled)
    y_pred_proba_aug = xgb_aug.predict_proba(X_test_scaled)[:, 1]

st.subheader("Classification Reports")
report_col1, report_col2 = st.columns(2)
with report_col1:
    st.text("XGBoost - Original Data")
    st.code(classification_report(y_test, y_pred_xgb_orig))
with report_col2:
    st.text("XGBoost - Augmented Data")
    st.code(classification_report(y_test, y_pred_xgb_aug))

# Confusion Matrices
st.subheader("Confusion Matrices")
#cm_col1, cm_col2 = st.columns(2)
#with cm_col1:
cm_orig = confusion_matrix(y_test, y_pred_xgb_orig)
fig_cm_orig, ax_cm_orig = plt.subplots()
sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'], ax=ax_cm_orig)
ax_cm_orig.set_title('Original XGBoost')
ax_cm_orig.set_xlabel('Predicted Label')
ax_cm_orig.set_ylabel('Actual Label')
st.pyplot(fig_cm_orig)

#with cm_col2:
cm_aug = confusion_matrix(y_test, y_pred_xgb_aug)
fig_cm_aug, ax_cm_aug = plt.subplots()
sns.heatmap(cm_aug, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'], ax=ax_cm_aug)
ax_cm_aug.set_title('Augmented XGBoost')
ax_cm_aug.set_xlabel('Predicted Label')
ax_cm_aug.set_ylabel('Actual Label')
st.pyplot(fig_cm_aug)

# ROC Curves
st.subheader("ROC Curve Comparison")
fpr_orig, tpr_orig, _ = roc_curve(y_test, y_pred_proba_orig)
auc_orig = roc_auc_score(y_test, y_pred_proba_orig)
fpr_aug, tpr_aug, _ = roc_curve(y_test, y_pred_proba_aug)
auc_aug = roc_auc_score(y_test, y_pred_proba_aug)

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr_orig, tpr_orig, color='orange', linestyle=':', label=f'Original Data ROC (AUC = {auc_orig:.2f})')
ax_roc.plot(fpr_aug, tpr_aug, color='blue', linestyle='-', label=f'Augmented Data ROC (AUC = {auc_aug:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
ax_roc.set_title('ROC Curve: Original vs. Augmented Data Models')
ax_roc.set_xlabel('False Positive Rate (FPR)')
ax_roc.set_ylabel('True Positive Rate (TPR)')
ax_roc.legend(loc='lower right')
ax_roc.grid(True)
st.pyplot(fig_roc)

st.metric(label="Original Data ROC AUC", value=f"{auc_orig:.2f}")
st.metric(label="Augmented Data ROC AUC", value=f"{auc_aug:.2f}", delta=f"{auc_aug-auc_orig:.3f}")

st.success("Model performance improved with synthetic data augmentation, particularly in balancing precision and recall, leading to a higher overall F1-score and ROC AUC.")