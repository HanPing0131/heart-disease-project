import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import custom modules
from src.preprocessing import clean_data, handle_outliers
from src.feature_engineering import get_feature_pipeline
from src.models import get_model_zoo
from src.clustering import run_clustering_analysis

# --- 1. Load Data with Robust Pathing ---
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'data', 'heart_disease_uci.csv')

try:
    df = pd.read_csv(csv_path)
    # Recode target column into binary 
    df['target'] = (df['num'] > 0).astype(int)
    print(f"Successfully loaded {csv_path}")
except FileNotFoundError:
    print(f"ERROR: File not found at {csv_path}. Please check your 'data' folder.")
    exit()

# --- 2. Clean and Preprocess ---
df = clean_data(df)       # Handle missing values 
df = handle_outliers(df)  # Remove outliers

X = df.drop('target', axis=1)
y = df['target']

# --- 3. Stratified Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42 
)

# --- 4. Comparative Classification Analysis ---
feature_pipe = get_feature_pipeline() # Scaling + PCA 
models = get_model_zoo()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

print("\n" + "="*50)
print(f"{'Model Name':<25} | {'Mean CV Accuracy':<18}")
print("-" * 50)

for name, model in models.items():
    full_pipeline = Pipeline(steps=[
        ('features', feature_pipe),
        ('classifier', model)
    ])
    # Perform Cross-Validation 
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"{name:<25} | {cv_scores.mean():.4f}")

# --- 5. Unsupervised Clustering ---
print("\n" + "="*50)
print("Running Clustering Analysis...")
X_transformed = feature_pipe.fit_transform(X) # Data after PCA
cluster_results = run_clustering_analysis(X_transformed)

for algo, labels in cluster_results.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"{algo:<25} | Found {n_clusters} clusters")

# --- 6. Final Evaluation for ALL Models ---
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION FOR ALL MODELS")
print("="*60)

#store results to compare them later
final_comparison = []

for name, model in models.items():
    # Create and fit the full pipeline for each model
    final_pipeline = Pipeline(steps=[
        ('features', feature_pipe),
        ('classifier', model)
    ])
    
    final_pipeline.fit(X_train, y_train)
    y_pred = final_pipeline.predict(X_test)
    
    # Calculate metrics
    acc = final_pipeline.score(X_test, y_test)
    recall = recall_score(y_test, y_pred)
    
    final_comparison.append({'Model': name, 'Accuracy': acc, 'Recall': recall})
    
    print(f"\n>>> Model: {name}")
    print(f"Test Accuracy: {acc:.4f} | Test Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

# --- 7. Summary Table ---
print("\n" + "="*60)
print(f"{'Model Name':<25} | {'Test Acc':<10} | {'Test Recall':<10}")
print("-" * 60)
for res in final_comparison:
    print(f"{res['Model']:<25} | {res['Accuracy']:.4f}    | {res['Recall']:.4f}")

    # --- Step 8: Final Model Export for Deployment ---

# Initialize the final KNN pipeline using the best-performing configuration
# use n_neighbors=5 as it provided the highest Recall in our tests
final_knn_pipe = Pipeline(steps=[
    ('features', feature_pipe),         
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

# Fit the pipeline on the training data 
# ensures the Scaler and PCA are mapped to the correct feature distributions
final_knn_pipe.fit(X_train, y_train)

# Persist the model to a physical file using joblib
model_filename = 'heart_disease_knn_model.pkl'
joblib.dump(final_knn_pipe, model_filename)

import shap

# --- 9. Prepare and Save SHAP Metadata ---
print("\n>>> Preparing SHAP metadata for the explainable AI component...")

try:
    # 1. Access the inner pipeline (named 'features')
    inner_pipeline = final_pipeline.named_steps['features']
    
    # 2. Access the real ColumnTransformer (named 'preprocessor' inside the inner pipeline)
    # THIS WAS THE MISSING LINK
    ct = inner_pipeline.named_steps['preprocessor']
    
    # 3. Reconstruct feature names
    num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    
    # Now we can safely access named_transformers_ from 'ct'
    cat_encoder = ct.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out()
    
    all_feature_names = list(num_features) + list(encoded_cat_names)
    
    # 4. Transform X_train using the full 'features' pipeline
    X_train_transformed = inner_pipeline.transform(X_train)
    
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
        
    # 5. Summarize with KMeans
    background_summary = shap.kmeans(X_train_transformed, 50)
    
    # 6. Save metadata
    shap_metadata = {
        'background_data': background_summary,
        'feature_names': all_feature_names
    }
    joblib.dump(shap_metadata, 'shap_metadata.pkl')
    
    print(f"SUCCESS: shap_metadata.pkl saved with {len(all_feature_names)} features.")

except Exception as e:
    print(f"SHAP preparation failed: {e}")