import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow
import mlflow.sklearn


df = pd.read_csv("titanic preprocessing.csv")


# Menggunakan train_test_split() untuk melakukan pembagian dataset.
mlflow.autolog()

X = df.drop('Survived', axis=1)

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print("Jumlah data total: ",len(X))
print("Jumlah data latih: ",len(X_train))
print("Jumlah data test: ",len(X_test))

random_fores_model = RandomForestClassifier(random_state=42)

# 2. Latih (fit) model dengan data training (X_train dan y_train)
random_fores_model.fit(X_train, y_train)

y_pred_rf = random_fores_model.predict(X_test)

# Tampilkan classification_report 
print("Model Performance")
print(classification_report(y_test, y_pred_rf))

print("="*50)

