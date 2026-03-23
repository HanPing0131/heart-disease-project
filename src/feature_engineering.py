from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_pipeline():
    """
    Creates a pipeline for feature encoding, scaling, and PCA reduction[cite: 14, 15, 26, 48].
    """
    # Define features from the dataset [cite: 38]
    numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    # Preprocessing: Standardize numeric and One-Hot encode categorical variables [cite: 48]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # PCA to reduce dimensionality to 6 components [cite: 27, 50]
    feature_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=6))
    ])
    
    return feature_pipeline