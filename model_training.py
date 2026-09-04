import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

print("Loading dataset from local cache...")
url = "/home/devsonar/.gemini/antigravity/brain/c3afb3e0-8198-49ba-aa80-edb3d853b304/.system_generated/steps/44/content.md"
df = pd.read_csv(url, skiprows=4)

# Clean and filter the dataset
# We want these features: Time (24hr), Number of Vehicles, Road Surface, Lighting Conditions, Weather Conditions, Type of Vehicle, Age of Casualty, Sex of Casualty
df = df.dropna(subset=['Time (24hr)', 'Number of Vehicles', 'Road Surface', 'Lighting Conditions', 'Weather Conditions', 'Type of Vehicle', 'Age of Casualty', 'Sex of Casualty', 'Casualty Severity'])

# Convert target: 1 = Fatal, 2 = Serious, 3 = Slight
# We will model 'High Risk' as Severity 1 or 2 (Fatal/Serious)
df['High_Risk'] = df['Casualty Severity'].apply(lambda x: 1 if x in [1, 2] else 0)

# Filter out unknown/invalid values based on UK STATS19 schema if needed, but let's just train on raw values
# Features
X = df[['Time (24hr)', 'Number of Vehicles', 'Road Surface', 'Lighting Conditions', 'Weather Conditions', 'Type of Vehicle', 'Age of Casualty', 'Sex of Casualty']]
y = df['High_Risk'].values

# Preprocessing pipeline
numeric_features = ['Time (24hr)', 'Number of Vehicles', 'Age of Casualty']
categorical_features = ['Road Surface', 'Lighting Conditions', 'Weather Conditions', 'Type of Vehicle', 'Sex of Casualty']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X).astype(np.float32)

joblib.dump(preprocessor, 'preprocessor.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f"Training robust model on {len(X_train)} samples with {X_processed.shape[1]} features...")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Training Complete! Test Accuracy: {accuracy * 100:.2f}%, AUC: {auc:.4f}")

model.save('model.keras')
print("Saved robust model.keras and preprocessor.pkl")