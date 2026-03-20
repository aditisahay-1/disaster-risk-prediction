import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Load data
# -----------------------------
X = np.load("data/X_lstm.npy")
y = np.load("data/y_lstm.npy")

# -----------------------------
# One-hot encoding
# -----------------------------
y_categorical = to_categorical(y, num_classes=3)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# -----------------------------
# Class weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# -----------------------------
# 🔥 Improved LSTM model
# -----------------------------
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64, return_sequences=True),
    Dropout(0.3),

    LSTM(32),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Early stopping 🔥
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)

# -----------------------------
# Save model
# -----------------------------
model.save("models/lstm_model.keras")

print("✅ LSTM model saved")