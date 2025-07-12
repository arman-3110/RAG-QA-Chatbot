import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load and clean data
df = pd.read_csv('Training Dataset.csv')
df.dropna(inplace=True)

# 2. Create a dictionary of label encoders
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # ✅ Store the encoder in the dictionary

# 3. Encode the target
target_encoder = LabelEncoder()
df['Loan_Status'] = target_encoder.fit_transform(df['Loan_Status'])

# 4. Train the model
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

model = RandomForestClassifier()
model.fit(X, y)

# 5. Save everything
joblib.dump(model, 'loan_model.pkl')                # ✅ ML model
joblib.dump(encoders, 'encoders_dict.pkl')          # ✅ dictionary of encoders
joblib.dump(target_encoder, 'target_encoder.pkl')   # optional

