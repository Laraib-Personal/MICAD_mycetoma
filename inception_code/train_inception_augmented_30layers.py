"""
InceptionV3 Training with Augmented Data - Fine-Tune Last 30 Layers
Dataset: Augmented (2,736 images - original + augmented)
Strategy: Frozen base → Fine-tune last 30 InceptionV3 layers
This is the BEST strategy from original data (91.97%) applied to augmented data
Expected: 94-97% accuracy
"""

import os
import numpy as np
import glob
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                             matthews_corrcoef, precision_score, recall_score,
                             roc_curve, auc, roc_auc_score)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.image as tfi
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set high-resolution plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")

print("="*80)
print("INCEPTIONV3 - AUGMENTED DATA - FINE-TUNE LAST 30 LAYERS")
print("Best Strategy (91.97% on original) + More Data = Expected 94-97%")
print("="*80)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
RESULTS_PATH = PROJECT_ROOT / "results_inceptionv3" / "05_augmented_last30"

# Create results directories
(RESULTS_PATH / "models").mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "logs").mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "plots").mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "plots" / "high_res").mkdir(parents=True, exist_ok=True)

# Data paths - AUGMENTED ONLY (contains original + augmented)
AUGMENTED_BM = DATASET_PATH / "augmented data" / "augmented data" / "BM"
AUGMENTED_FM = DATASET_PATH / "augmented data" / "augmented data" / "FM"

# Alternative paths
if not AUGMENTED_BM.exists():
    AUGMENTED_BM = DATASET_PATH / "augmented data" / "BM"
if not AUGMENTED_FM.exists():
    AUGMENTED_FM = DATASET_PATH / "augmented data" / "FM"

print(f"\n📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Dataset Path: {DATASET_PATH}")
print(f"📁 Results Path: {RESULTS_PATH}")
print(f"📁 Augmented BM Path: {AUGMENTED_BM}")
print(f"📁 Augmented FM Path: {AUGMENTED_FM}")

# Model configuration
IMAGE_SIZE = 256
BATCH_SIZE = 16
INITIAL_EPOCHS = 50
FINETUNE_EPOCHS = 50
LEARNING_RATE = 1e-4
FINETUNE_LR = 1e-5

# ============================================================================
# 2. LOAD DATA (AUGMENTED - INCLUDES ORIGINAL + AUGMENTED)
# ============================================================================

def load_image(image_path, size=256):
    """Load and preprocess single image"""
    img = load_img(str(image_path))
    img_array = img_to_array(img) / 255.0
    img_resized = tfi.resize(img_array, (size, size))
    return np.array(img_resized)

def load_images_from_directory(directory, size=256, label=""):
    """Load all images from directory (JPG only)"""
    directory = Path(directory)
    
    # Get image paths - ONLY JPG/JPEG
    image_paths = []
    for ext in ['*.jpg', '*.jpeg']:
        image_paths.extend(list(directory.glob(ext)))
    
    # Filter out mask files
    image_paths = [p for p in image_paths if not p.stem.endswith('_mask')]
    
    images = []
    print(f"\nLoading {len(image_paths)} {label} images from {directory.name}...")
    
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(image_paths)}")
        try:
            img = load_image(path, size)
            images.append(img)
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
    
    return np.array(images), image_paths

print("\n" + "="*80)
print("LOADING DATASET (AUGMENTED - INCLUDES ORIGINAL + AUGMENTED)")
print("="*80)

# Load from augmented folder only
print("\n📂 Loading from AUGMENTED data folder...")
print("Note: Augmented folder contains both original and augmented images")
X_bm, paths_bm = load_images_from_directory(AUGMENTED_BM, IMAGE_SIZE, "BM (all)")
X_fm, paths_fm = load_images_from_directory(AUGMENTED_FM, IMAGE_SIZE, "FM (all)")

y_bm = np.zeros(len(X_bm))
y_fm = np.ones(len(X_fm))

print(f"\n📊 Dataset Summary:")
print(f"  BM Total:       {len(X_bm)} images")
print(f"  FM Total:       {len(X_fm)} images")
print(f"  Grand Total:    {len(X_bm) + len(X_fm)} images")

# Combine data
X = np.concatenate([X_bm, X_fm], axis=0)
y = np.concatenate([y_bm, y_fm], axis=0)
y_categorical = to_categorical(y, num_classes=2)

# ============================================================================
# 3. SPLIT DATA
# ============================================================================

print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

# Split: 80% train, 10% val, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
)

print(f"  Train:      {len(X_train)} images ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val)} images ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {len(X_test)} images ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 4. BUILD MODEL
# ============================================================================

print("\n" + "="*80)
print("BUILDING MODEL")
print("="*80)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

print(f"\n📋 InceptionV3 Architecture:")
print(f"  Total base layers: {len(base_model.layers)}")
print(f"\n  Last 30 layers (to be unfrozen):")
for i, layer in enumerate(base_model.layers[-30:]):
    actual_idx = len(base_model.layers) - 30 + i
    print(f"    [{actual_idx}] {layer.name}")

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable_params_phase1 = sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"\n✅ Model built successfully")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Trainable (Phase 1): {trainable_params_phase1:,}")

# ============================================================================
# 5. TRAIN MODEL - PHASE 1 (FROZEN BASE)
# ============================================================================

print("\n" + "="*80)
print("TRAINING - PHASE 1: FROZEN BASE (50 EPOCHS)")
print("="*80)

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        str(RESULTS_PATH / "models" / "inception_aug30_phase1.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=INITIAL_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase1,
    verbose=1
)

# ============================================================================
# 6. FINE-TUNING - PHASE 2 (UNFREEZE LAST 30 LAYERS)
# ============================================================================

print("\n" + "="*80)
print("TRAINING - PHASE 2: FINE-TUNE LAST 30 LAYERS (50 EPOCHS)")
print("="*80)

# Unfreeze last 30 layers of BASE MODEL (not custom head)
print(f"\n🔓 Unfreezing LAST 30 InceptionV3 layers:")
for i, layer in enumerate(base_model.layers[-30:]):
    layer.trainable = True
    actual_idx = len(base_model.layers) - 30 + i
    if i % 5 == 0:  # Print every 5th layer
        print(f"  [{actual_idx}] {layer.name} → TRAINABLE")

trainable_params_phase2 = sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"\n  Total layers unfrozen: 30")
print(f"  Trainable parameters (Phase 2): {trainable_params_phase2:,}")
print(f"  Parameter increase: {trainable_params_phase2 - trainable_params_phase1:,}")

model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        str(RESULTS_PATH / "models" / "inception_aug30_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_phase2,
    verbose=1
)

# Combine histories
for key in history_phase1.history:
    history_phase1.history[key].extend(history_phase2.history[key])

history = history_phase1

# ============================================================================
# 7. EVALUATE MODEL
# ============================================================================

print("\n" + "="*80)
print("EVALUATING MODEL")
print("="*80)

# Load best model
model.load_weights(str(RESULTS_PATH / "models" / "inception_aug30_best.keras"))

# Predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = np.mean(y_pred == y_true)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
mcc = matthews_corrcoef(y_true, y_pred)

print(f"\n📊 Test Set Performance:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  MCC:       {mcc:.4f}")

# Per-class metrics
report = classification_report(y_true, y_pred, target_names=['BM', 'FM'], output_dict=True)

# Overfitting check
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
train_val_gap = (final_train_acc - final_val_acc) * 100

print(f"\n📊 Overfitting Analysis:")
print(f"  Final Train Accuracy: {final_train_acc*100:.2f}%")
print(f"  Final Val Accuracy:   {final_val_acc*100:.2f}%")
print(f"  Train-Val Gap:        {train_val_gap:.2f}%")
print(f"  Overfitting:          {'Yes' if train_val_gap > 10 else 'No'}")

# ============================================================================
# 8. GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING HIGH-RESOLUTION VISUALIZATIONS")
print("="*80)

# 1. Dataset Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
categories = ['BM\n(Bacterial)', 'FM\n(Fungal)']
counts = [len(X_bm), len(X_fm)]
colors = ['#3498db', '#e74c3c']

axes[0].bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
axes[0].set_title('Dataset Distribution (Augmented)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

for i, (cat, count) in enumerate(zip(categories, counts)):
    axes[0].text(i, count + 20, f'{count}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')

axes[1].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
axes[1].set_title('Class Distribution', fontsize=14, fontweight='bold')

total = len(X_bm) + len(X_fm)
plt.suptitle(f'Augmented Dataset - Total: {total} Images', 
            fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "01_dataset_distribution.png", 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved dataset distribution")

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

annotations = []
for i in range(2):
    row = []
    for j in range(2):
        row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
    annotations.append(row)

sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar=True,
           xticklabels=['BM', 'FM'], yticklabels=['BM', 'FM'],
           linewidths=2, linecolor='black', ax=ax,
           cbar_kws={'label': 'Count'})

ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title(f'Confusion Matrix - InceptionV3 (Augmented, Last 30 Layers)\nAccuracy: {accuracy*100:.2f}%', 
            fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "02_confusion_matrix.png", 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved confusion matrix")

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title(f'ROC Curve - InceptionV3 (Augmented, Last 30 Layers)\nAUC: {roc_auc:.3f}', 
            fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "03_roc_curve.png", 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved ROC curve")

# 4. Training History
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].axvline(x=INITIAL_EPOCHS, color='red', linestyle='--', label='Fine-tuning starts', linewidth=1.5)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].axvline(x=INITIAL_EPOCHS, color='red', linestyle='--', label='Fine-tuning starts', linewidth=1.5)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Learning Rate (if available)
axes[1, 0].text(0.5, 0.5, f'Phase 1 LR: {LEARNING_RATE}\nPhase 2 LR: {FINETUNE_LR}\n\nLayers Unfrozen: 30\nBest Strategy on Augmented Data', 
               ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
axes[1, 0].set_title('Training Configuration', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

# Final metrics
axes[1, 1].text(0.5, 0.5, 
               f'Test Accuracy: {accuracy*100:.2f}%\nPrecision: {precision*100:.2f}%\nRecall: {recall*100:.2f}%\nF1-Score: {f1*100:.2f}%\nMCC: {mcc:.4f}\nAUC: {roc_auc:.3f}',
               ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
axes[1, 1].set_title('Final Metrics', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

plt.suptitle('InceptionV3 - Augmented Data - Fine-Tune Last 30 Layers', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "04_training_history.png", 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved training history")

print(f"✅ Saved all visualizations")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    "experiment": "InceptionV3_Augmented_FineTune30Layers",
    "dataset": "Augmented Data (includes original + augmented)",
    "timestamp": datetime.now().isoformat(),
    "dataset_info": {
        "total_images": len(X),
        "bm_total": len(X_bm),
        "fm_total": len(X_fm),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "note": "Augmented folder contains both original and augmented images"
    },
    "hyperparameters": {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "initial_epochs": INITIAL_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "finetune_lr": FINETUNE_LR,
        "dropout": 0.6,
        "l2_reg": 0.001,
        "fine_tuned_layers": "Last 30 InceptionV3 layers"
    },
    "trainable_parameters": {
        "phase1": int(trainable_params_phase1),
        "phase2": int(trainable_params_phase2)
    },
    "test_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "auc": float(roc_auc)
    },
    "per_class_metrics": {
        "BM": {
            "precision": float(report['BM']['precision']),
            "recall": float(report['BM']['recall']),
            "f1_score": float(report['BM']['f1-score']),
            "support": int(report['BM']['support'])
        },
        "FM": {
            "precision": float(report['FM']['precision']),
            "recall": float(report['FM']['recall']),
            "f1_score": float(report['FM']['f1-score']),
            "support": int(report['FM']['support'])
        }
    },
    "overfitting_analysis": {
        "final_train_accuracy": float(final_train_acc),
        "final_val_accuracy": float(final_val_acc),
        "test_accuracy": float(accuracy),
        "train_val_gap": float(train_val_gap),
        "overfitting_detected": bool(train_val_gap > 10)
    },
    "confusion_matrix": cm.tolist()
}

# Save JSON
with open(RESULTS_PATH / "logs" / "inception_aug30_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved detailed results (JSON)")

# Save CSV summary
summary_df = pd.DataFrame([{
    'Experiment': 'InceptionV3 - Augmented - Fine-Tune Last 30 Layers',
    'Dataset': 'Augmented',
    'Total_Images': len(X),
    'Accuracy': f"{accuracy*100:.2f}%",
    'Precision': f"{precision*100:.2f}%",
    'Recall': f"{recall*100:.2f}%",
    'F1_Score': f"{f1*100:.2f}%",
    'MCC': f"{mcc:.4f}",
    'AUC': f"{roc_auc:.3f}",
    'Train_Val_Gap': f"{train_val_gap:.2f}%",
    'Overfitting': 'No' if train_val_gap < 10 else 'Yes',
    'Trainable_Params_Phase1': f"{trainable_params_phase1:,}",
    'Trainable_Params_Phase2': f"{trainable_params_phase2:,}"
}])

summary_df.to_csv(RESULTS_PATH / "logs" / "inception_aug30_summary.csv", index=False)
print(f"✅ Saved summary (CSV)")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📁 All results saved to: {RESULTS_PATH}")
print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
print(f"🎯 Train-Val Gap: {train_val_gap:.2f}%")
print(f"🎯 Layers Fine-tuned: Last 30 (Best Strategy)")
print(f"\n🏆 Expected: This should be your BEST result!")
print(f"   (Best strategy 91.97% + 4× more data = 94-97% expected)")
print("="*80)

