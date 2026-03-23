from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_model_zoo():
    """
    Returns a comprehensive dictionary of all requested classification models.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000), # [cite: 30, 54]
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), # [cite: 32, 55]
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42)
    }