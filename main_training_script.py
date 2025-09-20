import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Basic Settings ----------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------- Paths ----------------------
data_path = r"C:\Alex The Great\Project\medai-env\datasets\Microarray Innovations in LEukemia\ALL-lusemi had lanfositi\All\FINAL_balanced_all_expression_V2.csv"
labels_path = r"C:\Alex The Great\Project\medai-env\datasets\Microarray Innovations in LEukemia\ALL-lusemi had lanfositi\All\FINAL_balanced_all_metadata_V2.csv"

# ---------------------- Data Loading ----------------------
print("Loading data...")
X_df = pd.read_csv(data_path, index_col=0).T
meta = pd.read_csv(labels_path)
y = meta['subtype']

# Subtype mapping
subtype_mapping = {
    'ALL_TEL_AML1': 0, 'ALL_Hyperdiploid': 1, 'B_ALL': 2,
    'T_ALL': 3, 'ALL_E2A_PBX1': 4, 'ALL_other': 5, 
    'ALL_Ph_positive': 6, 'ALL_MLL_rearranged': 7, 'Normal': 8
}

y_numeric = np.array([subtype_mapping[label] for label in y])
num_classes = len(subtype_mapping)

# One-hot encoding
y_cat = tf.keras.utils.to_categorical(y_numeric, num_classes=num_classes)

# Convert to numpy array
X = X_df.values.astype('float32')

# ---------------------- Data Cleaning ----------------------
print(f"Cleaning data... NaN values before: {np.isnan(X).sum()}")

# Remove samples with NaN
nan_samples = np.isnan(X).any(axis=1)
if nan_samples.any():
    print(f"Removing {nan_samples.sum()} samples with NaN values")
    X = X[~nan_samples]
    y_numeric = y_numeric[~nan_samples]
    y_cat = y_cat[~nan_samples]

# Remove features with NaN
nan_features = np.isnan(X).any(axis=0)
if nan_features.any():
    print(f"Removing {nan_features.sum()} features with NaN values")
    X = X[:, ~nan_features]

print(f"After cleaning - NaN values: {np.isnan(X).sum()}")

# ---------------------- Preprocessing ----------------------
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection - select top 5000 features
K = 5000
if X_scaled.shape[1] > K:
    selector = SelectKBest(f_classif, k=K)
    X_selected = selector.fit_transform(X_scaled, y_numeric)
    print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
else:
    X_selected = X_scaled
    print(f"Using all {X_selected.shape[1]} features (less than {K})")

# Class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_numeric),
    y=y_numeric
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Data split
X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
    X_selected, y_numeric, y_cat, test_size=0.2, 
    stratify=y_numeric, random_state=SEED
)

# Reshape for deep learning models
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Class distribution: {np.unique(y_train, return_counts=True)}")

# ---------------------- Augmentation ----------------------
def augment_genomic_data(batch):
    """Mild augmentation for genomic data"""
    noise = np.random.normal(0, 0.005, batch.shape)
    scale = np.random.uniform(0.99, 1.01, (batch.shape[0], 1, 1))
    return batch * scale + noise

def data_generator(X, y, batch_size=32, augment=True):
    """Data generator"""
    n = len(X)
    indices = np.arange(n)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            if augment:
                X_batch = augment_genomic_data(X_batch)
            
            yield X_batch, y_batch

# ---------------------- Model Definitions ----------------------
def create_cnn_model(input_shape, num_classes):
    """CNN model"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='cnn')

def create_lstm_model(input_shape, num_classes):
    """LSTM model"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='lstm')

def create_dense_model(input_shape, num_classes):
    """Dense model"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='dense')

# Create models
print("Creating models...")
models_dict = {
    'cnn': create_cnn_model((X_train_reshaped.shape[1], 1), num_classes),
    'lstm': create_lstm_model((X_train_reshaped.shape[1], 1), num_classes),
    'dense': create_dense_model((X_train_reshaped.shape[1], 1), num_classes)
}

# Compile models
for name, model in models_dict.items():
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Compiled {name} model")

# ---------------------- Model Training ----------------------
batch_size = 16
epochs = 150
history_dict = {}

print("Starting model training...")

for name, model in models_dict.items():
    print(f"Training {name} model...")
    
    callbacks_list = [
        callbacks.EarlyStopping(patience=25, restore_best_weights=True, 
                              monitor='val_accuracy', mode='max'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-6),
        callbacks.ModelCheckpoint(f'best_{name}_model.keras', 
                                save_best_only=True,
                                monitor='val_accuracy', mode='max')
    ]
    
    history = model.fit(
        data_generator(X_train_reshaped, y_train_cat, batch_size=batch_size),
        steps_per_epoch=len(X_train_reshaped) // batch_size,
        validation_data=(X_test_reshaped, y_test_cat),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    history_dict[name] = history
    print(f"Finished training {name} model")

# ---------------------- Save Models ----------------------
print("Saving models before feature extraction...")
for name, model in models_dict.items():
    model.save(f'{name}_model.keras')
    print(f"Saved {name}_model.keras")

# ---------------------- Feature Extraction ----------------------
print("Extracting features from models...")

meta_features_train = []
meta_features_test = []

for name, model in models_dict.items():
    # Predict on training data
    train_preds = model.predict(X_train_reshaped, verbose=0)
    meta_features_train.append(train_preds)
    
    # Predict on test data
    test_preds = model.predict(X_test_reshaped, verbose=0)
    meta_features_test.append(test_preds)
    
    print(f"Extracted features from {name} model")

# Combine meta features
X_meta_train = np.concatenate(meta_features_train, axis=1)
X_meta_test = np.concatenate(meta_features_test, axis=1)

print(f"Meta features shape: {X_meta_train.shape}")

# ---------------------- Meta-Learner ----------------------
print("Training Meta-Learner...")

meta_learner = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=SEED
)

meta_learner.fit(X_meta_train, y_train)

# Final prediction
y_pred_meta = meta_learner.predict(X_meta_test)
y_pred_meta_prob = meta_learner.predict_proba(X_meta_test)

# ---------------------- Save Artifacts ----------------------
joblib.dump(meta_learner, 'meta_learner.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')

with open('subtype_mapping.json', 'w') as f:
    json.dump(subtype_mapping, f)

print("All models and artifacts saved!")

# ---------------------- Evaluation ----------------------
print("\n" + "="*60)
print("FINAL ENSEMBLE MODEL EVALUATION")
print("="*60)

# Find existing classes in test data
existing_classes = np.unique(y_test)
target_names = [list(subtype_mapping.keys())[i] for i in existing_classes]

# 1. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_meta, 
                           target_names=target_names,
                           labels=existing_classes))

# 2. Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_meta, labels=existing_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix - ALL Subtypes Classification\n(CNN + LSTM + Dense + Gradient Boosting)', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_all_subtypes.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. ROC Curve and AUC
plt.figure(figsize=(10, 8))
fpr = dict()
tpr = dict()
roc_auc = dict()

# One-hot encoding for y_test
y_test_onehot = np.zeros((len(y_test), len(existing_classes)))
for i, cls in enumerate(existing_classes):
    y_test_onehot[:, i] = (y_test == cls).astype(int)

# Calculate ROC only for existing classes
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta'])

for i, cls in enumerate(existing_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_pred_meta_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Class name
    class_name = list(subtype_mapping.keys())[cls]
    
    plt.plot(fpr[i], tpr[i], color=next(colors), lw=3,
             label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

# Random line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')

# Chart settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve - ALL Subtypes Classification\n(Ensemble Model)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

# Mean AUC
mean_auc = np.mean(list(roc_auc.values()))
plt.text(0.6, 0.2, f'Mean AUC: {mean_auc:.3f}', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('roc_curve_all_subtypes.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean ROC-AUC Score: {mean_auc:.4f}")

# 4. Training History Plots
for name, history in history_dict.items():
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name.upper()} Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name.upper()} Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------- Final Results ----------------------
print("\nTRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print("FINAL RESULTS:")
print(f"   • Number of subtypes: {num_classes}")
print(f"   • Training samples: {X_train.shape[0]}")
print(f"   • Test samples: {X_test.shape[0]}")
print(f"   • Selected features: {X_selected.shape[1]}")
print(f"   • Final accuracy: {np.mean(y_pred_meta == y_test):.4f}")
print(f"   • Mean ROC-AUC: {mean_auc:.4f}")

print("\nDetectable subtypes:")
for subtype, code in subtype_mapping.items():
    count = np.sum(y_numeric == code)
    print(f"   {code}: {subtype} ({count} samples)")

print("\nAll models and visualizations saved successfully!")
print("Confusion Matrix: confusion_matrix_all_subtypes.png")
print("ROC Curve: roc_curve_all_subtypes.png")
print("Training Histories: training_history_*.png")