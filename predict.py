# Required libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

print("Movie Success Predictor script started...")

# --- Step 1: Load and Prepare Data ---
try:
    
    df = pd.read_csv('tmdb_5000_movies.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'tmdb_5000_movies.csv' not found.")
    print("Please make sure the dataset file is in the same folder as the script.")
    exit()

# --- Step 2: Feature Engineering and Cleaning ---

features = [
    'budget', 'genres', 'popularity', 'production_companies', 
    'runtime', 'vote_average', 'vote_count', 'revenue'
]
df = df[features]

df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Missing values handle 
df['runtime'].fillna(df['runtime'].mean(), inplace=True)

# JSON columns se data nikalna (e.g., genre count)
def get_json_list_count(json_str):
    try:
        return len(json.loads(json_str))
    except (TypeError, json.JSONDecodeError):
        return 0

df['genre_count'] = df['genres'].apply(get_json_list_count)
df['company_count'] = df['production_companies'].apply(get_json_list_count)

# Target variable banayein: 'Success' ya 'Flop'
# Agar movie ne apne budget se 1.5 guna zyada kamaya, to 'Success' (1), varna 'Flop' (0)
df['success'] = (df['revenue'] > df['budget'] * 1.5).astype(int)

print(f"Total movies in dataset after cleaning: {len(df)}")
print(f"Successes: {df['success'].sum()}, Flops: {len(df) - df['success'].sum()}")


final_features = [
    'budget', 'popularity', 'runtime', 'vote_average', 
    'vote_count', 'genre_count', 'company_count'
]
X = df[final_features]
y = df['success']

# --- Step 4: Train the Model ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier model 
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Step 5: Evaluate the Model ---
print("\nEvaluating the model...")
y_pred = model.predict(X_test)

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# 
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Flop', 'Success']))

# 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Flop', 'Success'], yticklabels=['Flop', 'Success'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

print("\nConfusion matrix saved as 'confusion_matrix.png'.")
print("Project execution finished.")