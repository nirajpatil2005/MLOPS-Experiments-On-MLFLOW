import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
from mlflow.models import infer_signature

# Initialize DagsHub
dagshub.init(repo_owner='nirajpatil2005', repo_name='MLOPS-Eeperiments-on-MLFlow', mlflow=True)

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 8
n_estimators = 5

# Set experiment
mlflow.set_experiment('YT-MLOPS-Exp2')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Creating and logging confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Log the source code
    mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tags({
        "Author": "Niraj", 
        "Project": "Wine Classification",
        "Framework": "scikit-learn"
    })

    # Log the model (updated for DagsHub compatibility)
    signature = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        signature=signature,
        registered_model_name="WineClassifier-RandomForest"
    )

    print(f"Model training complete. Accuracy: {accuracy:.4f}")