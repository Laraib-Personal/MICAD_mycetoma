"""
InceptionV3 Training with Augmented Data - Fine-Tune Last 5 Layers
Dataset: Original (684) + Augmented (2,052) = 2,736 images
Strategy: Frozen base → Fine-tune last 5 layers
Complete with high-resolution visualizations and comprehensive metrics
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
print("INCEPTIONV3 - AUGMENTED DATA - FINE-TUNE LAST 5 LAYERS")
print("="*80)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
RESULTS_PATH = PROJECT_ROOT / "results_inceptionv3" / "03_augmented_last5"

# Create results directories
(RESULTS_PATH / "models").mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "logs").mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "plots").mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "plots" / "high_res").mkdir(parents=True, exist_ok=True)

# Data paths - ORIGINAL + AUGMENTED
ORIGINAL_BM = DATASET_PATH / "BM" / "BM"
ORIGINAL_FM = DATASET_PATH / "FM" / "FM"
AUGMENTED_BM = DATASET_PATH / "augmented data" / "augmented data" / "BM"
AUGMENTED_FM = DATASET_PATH / "augmented data" / "augmented data" / "FM"

# Alternative paths
if not ORIGINAL_BM.exists():
    ORIGINAL_BM = DATASET_PATH / "BM"
if not ORIGINAL_FM.exists():
    ORIGINAL_FM = DATASET_PATH / "FM"
if not AUGMENTED_BM.exists():
    AUGMENTED_BM = DATASET_PATH / "augmented data" / "BM"
if not AUGMENTED_FM.exists():
    AUGMENTED_FM = DATASET_PATH / "augmented data" / "FM"

print(f"\n📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Dataset Path: {DATASET_PATH}")
print(f"📁 Results Path: {RESULTS_PATH}")

# Model configuration
IMAGE_SIZE = 256
BATCH_SIZE = 16
INITIAL_EPOCHS = 50
FINETUNE_EPOCHS = 50
LEARNING_RATE = 1e-4
FINETUNE_LR = 1e-5

# ============================================================================
# 2. LOAD DATA (ORIGINAL + AUGMENTED)
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
print("LOADING DATASET (AUGMENTED DATA - INCLUDES ORIGINAL + AUGMENTED)")
print("="*80)

# Load Augmented Data (already contains original + augmented images)
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
print(f"  Last 5 layers:")
for i, layer in enumerate(base_model.layers[-5:]):
    actual_idx = len(base_model.layers) - 5 + i
    print(f"    [{actual_idx}] {layer.name}")

# Freeze base model
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
print("TRAINING - PHASE 1: FROZEN BASE")
print("="*80)

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        str(RESULTS_PATH / "models" / "inception_aug5_phase1.keras"),
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
# 6. FINE-TUNING - PHASE 2 (UNFREEZE LAST 5 LAYERS)
# ============================================================================

print("\n" + "="*80)
print("TRAINING - PHASE 2: FINE-TUNE LAST 5 LAYERS")
print("="*80)

# Unfreeze last 5 base model layers
print(f"\n🔓 Unfreezing last 5 InceptionV3 layers:")
for i, layer in enumerate(base_model.layers[-5:]):
    layer.trainable = True
    actual_idx = len(base_model.layers) - 5 + i
    print(f"  [{actual_idx}] {layer.name} → TRAINABLE")

trainable_params_phase2 = sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"\n  Trainable parameters (Phase 2): {trainable_params_phase2:,}")

model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        str(RESULTS_PATH / "models" / "inception_aug5_best.keras"),
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
model.load_weights(str(RESULTS_PATH / "models" / "inception_aug5_best.keras"))

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

print(f"\n📊 Per-Class Performance:")
print(f"  BM (Bacterial):")
print(f"    Precision: {report['BM']['precision']*100:.2f}%")
print(f"    Recall:    {report['BM']['recall']*100:.2f}%")
print(f"    F1-Score:  {report['BM']['f1-score']*100:.2f}%")
print(f"  FM (Fungal):")
print(f"    Precision: {report['FM']['precision']*100:.2f}%")
print(f"    Recall:    {report['FM']['recall']*100:.2f}%")
print(f"    F1-Score:  {report['FM']['f1-score']*100:.2f}%")

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
# 8. HIGH-RESOLUTION VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING HIGH-RESOLUTION VISUALIZATIONS")
print("="*80)

# 1. DATASET DISTRIBUTION
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

categories = ['BM\n(Bacterial)', 'FM\n(Fungal)']
counts = [len(X_bm), len(X_fm)]
colors = ['#3498db', '#e74c3c']

axes[0].bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
axes[0].set_title('Dataset Distribution (Original + Augmented)', fontsize=14, fontweight='bold')
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

# 2. CONFUSION MATRIX
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
ax.set_title(f'Confusion Matrix - InceptionV3 (Augmented, Fine-Tune 5L)\nAccuracy: {accuracy*100:.2f}%', 
            fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "02_confusion_matrix.png", 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved confusion matrix")

# 3. ROC CURVES
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

colors = ['#3498db', '#e74c3c']
class_names = ['BM', 'FM']

for i, (color, name) in enumerate(zip(colors, class_names)):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=color, lw=3,
           label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('ROC Curves - InceptionV3 (Augmented, Fine-Tune 5L)', 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "03_roc_curves.png",
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved ROC curves")

# 4. TRAINING HISTORY
fig = plt.figure(figsize=(16, 6), dpi=300)
gs = GridSpec(1, 2, figure=fig)

epochs_range = range(1, len(history.history['loss']) + 1)

# Loss plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_range, history.history['loss'], 'o-', color='#3498db', 
        linewidth=2, markersize=4, label='Training Loss')
ax1.plot(epochs_range, history.history['val_loss'], 's-', color='#e74c3c',
        linewidth=2, markersize=4, label='Validation Loss')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(alpha=0.3)

# Accuracy plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_range, [acc*100 for acc in history.history['accuracy']], 
        'o-', color='#2ecc71', linewidth=2, markersize=4, label='Training Accuracy')
ax2.plot(epochs_range, [acc*100 for acc in history.history['val_accuracy']], 
        's-', color='#f39c12', linewidth=2, markersize=4, label='Validation Accuracy')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "04_training_history.png",
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved training history")

# 5. PER-CLASS METRICS
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

metrics = ['precision', 'recall', 'f1-score']
bm_scores = [report['BM'][m] * 100 for m in metrics]
fm_scores = [report['FM'][m] * 100 for m in metrics]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, bm_scores, width, label='BM', 
              color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, fm_scores, width, label='FM',
              color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.title() for m in metrics])
ax.legend(fontsize=11, frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 110])

plt.tight_layout()
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "05_per_class_metrics.png",
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved per-class metrics")

# 6. COMPREHENSIVE DASHBOARD
fig = plt.figure(figsize=(16, 10), dpi=300)
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Top-left: Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
           xticklabels=['BM', 'FM'], yticklabels=['BM', 'FM'],
           linewidths=2, linecolor='black', ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=10, fontweight='bold')

# Top-right: Metrics Summary
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
metrics_text = f"""
TEST SET PERFORMANCE
(Augmented Data - Fine-Tune 5 Layers)

Overall Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:     {accuracy*100:6.2f}%
Precision:    {precision*100:6.2f}%
Recall:       {recall*100:6.2f}%
F1-Score:     {f1*100:6.2f}%
MCC:          {mcc:7.4f}

Class-wise Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM (Bacterial):
  Precision:  {report['BM']['precision']*100:6.2f}%
  Recall:     {report['BM']['recall']*100:6.2f}%
  F1-Score:   {report['BM']['f1-score']*100:6.2f}%
  Support:    {int(report['BM']['support']):4d}

FM (Fungal):
  Precision:  {report['FM']['precision']*100:6.2f}%
  Recall:     {report['FM']['recall']*100:6.2f}%
  F1-Score:   {report['FM']['f1-score']*100:6.2f}%
  Support:    {int(report['FM']['support']):4d}

Dataset Info:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM Total:     {len(X_bm):4d} images
FM Total:     {len(X_fm):4d} images
Total:        {len(X):4d} images
Train:        {len(X_train):4d} images
Validation:   {len(X_val):4d} images
Test:         {len(X_test):4d} images

Training Info:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1: Frozen base
Phase 2: Fine-tune last 5 layers
Train-Val Gap: {train_val_gap:.2f}%
"""
ax2.text(0.1, 0.95, metrics_text, transform=ax2.transAxes,
        fontfamily='monospace', fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Bottom-left: Accuracy comparison
ax3 = fig.add_subplot(gs[1, 0])
test_acc = accuracy * 100

bars = ax3.bar(['Train', 'Validation', 'Test'], 
              [final_train_acc*100, final_val_acc*100, test_acc],
              color=['#2ecc71', '#f39c12', '#9b59b6'],
              alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
ax3.set_title('Accuracy Across Splits', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 110])

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# Bottom-right: Overfitting analysis
ax4 = fig.add_subplot(gs[1, 1])
val_test_gap = (final_val_acc*100 - test_acc)

bars = ax4.bar(['Train-Val\nGap', 'Val-Test\nGap'], 
              [train_val_gap, val_test_gap],
              color=['#e67e22', '#3498db'],
              alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axhline(y=10, color='r', linestyle='--', linewidth=2, label='10% Threshold')
ax4.set_ylabel('Gap (%)', fontsize=10, fontweight='bold')
ax4.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

plt.suptitle('InceptionV3 - Augmented Data - Fine-Tune 5 Layers - Comprehensive Dashboard', 
            fontsize=16, fontweight='bold', y=0.98)
plt.savefig(RESULTS_PATH / "plots" / "high_res" / "06_comprehensive_dashboard.png",
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✅ Saved comprehensive dashboard")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    "experiment": "InceptionV3_Augmented_FineTune5Layers",
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
        "fine_tuned_layers": 5
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
        "mcc": float(mcc)
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
    "confusion_matrix": cm.tolist(),
    "training_history": {
        "loss": [float(x) for x in history.history['loss']],
        "val_loss": [float(x) for x in history.history['val_loss']],
        "accuracy": [float(x) for x in history.history['accuracy']],
        "val_accuracy": [float(x) for x in history.history['val_accuracy']]
    }
}

# Save JSON
with open(RESULTS_PATH / "logs" / "inception_aug5_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved detailed results (JSON)")

# Save CSV summary
summary_df = pd.DataFrame([{
    'Experiment': 'InceptionV3 - Augmented - Fine-Tune 5 Layers',
    'Dataset': 'Original + Augmented',
    'Total_Images': len(X),
    'Accuracy': f"{accuracy*100:.2f}%",
    'Precision': f"{precision*100:.2f}%",
    'Recall': f"{recall*100:.2f}%",
    'F1_Score': f"{f1*100:.2f}%",
    'MCC': f"{mcc:.4f}",
    'Train_Val_Gap': f"{train_val_gap:.2f}%",
    'Overfitting': 'No' if train_val_gap < 10 else 'Yes',
    'Trainable_Params_Phase1': f"{trainable_params_phase1:,}",
    'Trainable_Params_Phase2': f"{trainable_params_phase2:,}"
}])

summary_df.to_csv(RESULTS_PATH / "logs" / "inception_aug5_summary.csv", index=False)
print(f"✅ Saved summary (CSV)")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📁 All results saved to: {RESULTS_PATH}")
print(f"\n📊 High-Resolution Plots Generated:")
print(f"  1. Dataset Distribution")
print(f"  2. Confusion Matrix")
print(f"  3. ROC Curves")
print(f"  4. Training History")
print(f"  5. Per-Class Metrics")
print(f"  6. Comprehensive Dashboard")
print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
print(f"🎯 Train-Val Gap: {train_val_gap:.2f}%")
print(f"🎯 Overfitting: {'Yes' if train_val_gap > 10 else 'No'}")
print("="*80)

