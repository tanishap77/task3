import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the uploaded file
data = pd.read_csv('bank.csv')

# Preview
print("\nFirst 5 Rows of Data:")
print(data.head())

# Convert categorical variables
data = pd.get_dummies(data, drop_first=True)

# Features & Target
X = data.drop('y_yes', axis=1)
y = data['y_yes']


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix (Text):\n", confusion_matrix(y_test, y_pred))

# Visualize
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Optional: Pause script
input("✅ Press Enter to exit...")
