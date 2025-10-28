"""
Compare Fine-Tuning Strategies
Tests 3 scenarios:
1. No fine-tuning (frozen base)
2. Fine-tuning last 2 layers (original notebook approach)
3. Fine-tuning last 30 layers (our enhanced approach)
"""

import os
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.image as tfi

print("="*80)
print("FINE-TUNING STRATEGY COMPARISON")
print("="*80)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
RESULTS_PATH = PROJECT_ROOT / "results_finetuning_comparison"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

ORIGINAL_BM = DATASET_PATH / "BM" / "BM"
ORIGINAL_FM = DATASET_PATH / "FM" / "FM"

if not ORIGINAL_BM.exists():
    ORIGINAL_BM = DATASET_PATH / "BM"
if not ORIGINAL_FM.exists():
    ORIGINAL_FM = DATASET_PATH / "FM"

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS_FROZEN = 40
EPOCHS_FINETUNE = 50
LEARNING_RATE = 1e-4
FINETUNE_LR = 1e-5

# ============================================================================
# LOAD DATA
# ============================================================================

def load_image(image_path, size=256):
    """Load and preprocess single image"""
    img = load_img(str(image_path))
    img_array = img_to_array(img) / 255.0
    img_resized = tfi.resize(img_array, (size, size))
    return np.array(img_resized)

def load_images_from_directory(directory, size=256):
    """Load all images from directory (JPG only)"""
    directory = Path(directory)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg']:
        image_paths.extend(list(directory.glob(ext)))
    image_paths = [p for p in image_paths if not p.stem.endswith('_mask')]
    
    images = []
    print(f"\nLoading {len(image_paths)} images from {directory.name}...")
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(image_paths)}")
        try:
            img = load_image(path, size)
            images.append(img)
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
    
    return np.array(images), image_paths

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

X_bm, paths_bm = load_images_from_directory(ORIGINAL_BM, IMAGE_SIZE)
X_fm, paths_fm = load_images_from_directory(ORIGINAL_FM, IMAGE_SIZE)

y_bm = np.zeros(len(X_bm))
y_fm = np.ones(len(X_fm))

print(f"\n📊 Dataset Summary:")
print(f"  BM: {len(X_bm)} images")
print(f"  FM: {len(X_fm)} images")
print(f"  Total: {len(X_bm) + len(X_fm)} images")

X = np.concatenate([X_bm, X_fm], axis=0)
y = np.concatenate([y_bm, y_fm], axis=0)
y_categorical = to_categorical(y, num_classes=2)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train: {len(X_train)} images")
print(f"  Validation: {len(X_val)} images")

# ============================================================================
# INSPECT MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("MODEL ARCHITECTURE ANALYSIS")
print("="*80)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

print(f"\n🔍 InceptionV3 Base Model:")
print(f"  Total layers: {len(base_model.layers)}")
print(f"  Total parameters: {base_model.count_params():,}")

print(f"\n📋 Layer Breakdown:")
print(f"  First 10 layers:")
for i, layer in enumerate(base_model.layers[:10]):
    print(f"    [{i}] {layer.name} - Trainable: {layer.trainable}")

print(f"\n  Last 10 layers:")
for i, layer in enumerate(base_model.layers[-10:]):
    actual_idx = len(base_model.layers) - 10 + i
    print(f"    [{actual_idx}] {layer.name} - Trainable: {layer.trainable}")

print(f"\n  Last 2 layers (original notebook approach):")
for i, layer in enumerate(base_model.layers[-2:]):
    actual_idx = len(base_model.layers) - 2 + i
    print(f"    [{actual_idx}] {layer.name}")

print(f"\n  Last 30 layers (enhanced script approach):")
print(f"    Layers [{len(base_model.layers)-30}] to [{len(base_model.layers)-1}]")

# ============================================================================
# SCENARIO 1: NO FINE-TUNING (FROZEN BASE)
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 1: NO FINE-TUNING (ALL LAYERS FROZEN)")
print("="*80)

# Build model
base_model_1 = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model_1.layers:
    layer.trainable = False

x = base_model_1.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model_1 = Model(inputs=base_model_1.input, outputs=predictions)

trainable_params_1 = sum([np.prod(v.shape) for v in model_1.trainable_weights])
print(f"\n  Trainable parameters: {trainable_params_1:,}")
print(f"  Frozen parameters: {model_1.count_params() - trainable_params_1:,}")

model_1.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_1 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
]

print(f"\n⏳ Training with frozen base ({EPOCHS_FROZEN} epochs max)...")
history_1 = model_1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FROZEN,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_1,
    verbose=1
)

# Evaluate
y_pred_1 = np.argmax(model_1.predict(X_val, verbose=0), axis=1)
y_true = np.argmax(y_val, axis=1)

acc_1 = accuracy_score(y_true, y_pred_1)
prec_1 = precision_score(y_true, y_pred_1, average='weighted')
rec_1 = recall_score(y_true, y_pred_1, average='weighted')
f1_1 = f1_score(y_true, y_pred_1, average='weighted')
mcc_1 = matthews_corrcoef(y_true, y_pred_1)

print(f"\n✅ RESULTS (No Fine-Tuning):")
print(f"  Accuracy:  {acc_1*100:.2f}%")
print(f"  Precision: {prec_1*100:.2f}%")
print(f"  Recall:    {rec_1*100:.2f}%")
print(f"  F1-Score:  {f1_1*100:.2f}%")
print(f"  MCC:       {mcc_1:.4f}")

train_acc_1 = history_1.history['accuracy'][-1]
val_acc_1 = history_1.history['val_accuracy'][-1]
gap_1 = (train_acc_1 - val_acc_1) * 100
print(f"  Train-Val Gap: {gap_1:.2f}%")

# ============================================================================
# SCENARIO 2: FINE-TUNING LAST 2 LAYERS (ORIGINAL NOTEBOOK)
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 2: FINE-TUNING LAST 2 LAYERS (Original Notebook Approach)")
print("="*80)

# Build model
base_model_2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model_2.layers:
    layer.trainable = False

x = base_model_2.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model_2 = Model(inputs=base_model_2.input, outputs=predictions)

model_2.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n⏳ Phase 1: Training with frozen base ({EPOCHS_FROZEN} epochs max)...")
history_2a = model_2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FROZEN,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Unfreeze last 2 layers
print(f"\n🔓 Unfreezing last 2 layers:")
for i, layer in enumerate(model_2.layers[-2:]):
    layer.trainable = True
    actual_idx = len(model_2.layers) - 2 + i
    print(f"    [{actual_idx}] {layer.name}")

trainable_params_2 = sum([np.prod(v.shape) for v in model_2.trainable_weights])
print(f"\n  Now trainable parameters: {trainable_params_2:,}")

model_2.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n⏳ Phase 2: Fine-tuning ({EPOCHS_FINETUNE} epochs max)...")
history_2b = model_2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FINETUNE,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Evaluate
y_pred_2 = np.argmax(model_2.predict(X_val, verbose=0), axis=1)

acc_2 = accuracy_score(y_true, y_pred_2)
prec_2 = precision_score(y_true, y_pred_2, average='weighted')
rec_2 = recall_score(y_true, y_pred_2, average='weighted')
f1_2 = f1_score(y_true, y_pred_2, average='weighted')
mcc_2 = matthews_corrcoef(y_true, y_pred_2)

print(f"\n✅ RESULTS (Fine-Tuning Last 2 Layers):")
print(f"  Accuracy:  {acc_2*100:.2f}%")
print(f"  Precision: {prec_2*100:.2f}%")
print(f"  Recall:    {rec_2*100:.2f}%")
print(f"  F1-Score:  {f1_2*100:.2f}%")
print(f"  MCC:       {mcc_2:.4f}")

train_acc_2 = history_2b.history['accuracy'][-1]
val_acc_2 = history_2b.history['val_accuracy'][-1]
gap_2 = (train_acc_2 - val_acc_2) * 100
print(f"  Train-Val Gap: {gap_2:.2f}%")

# ============================================================================
# SCENARIO 3: FINE-TUNING LAST 30 LAYERS (ENHANCED APPROACH)
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 3: FINE-TUNING LAST 30 LAYERS (Enhanced Script Approach)")
print("="*80)

# Build model
base_model_3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model_3.layers:
    layer.trainable = False

x = base_model_3.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model_3 = Model(inputs=base_model_3.input, outputs=predictions)

model_3.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n⏳ Phase 1: Training with frozen base ({EPOCHS_FROZEN} epochs max)...")
history_3a = model_3.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FROZEN,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Unfreeze last 30 layers
print(f"\n🔓 Unfreezing last 30 layers:")
print(f"    Layers [{len(base_model_3.layers)-30}] to [{len(base_model_3.layers)-1}]")
for layer in base_model_3.layers[-30:]:
    layer.trainable = True

trainable_params_3 = sum([np.prod(v.shape) for v in model_3.trainable_weights])
print(f"\n  Now trainable parameters: {trainable_params_3:,}")

model_3.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n⏳ Phase 2: Fine-tuning ({EPOCHS_FINETUNE} epochs max)...")
history_3b = model_3.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FINETUNE,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Evaluate
y_pred_3 = np.argmax(model_3.predict(X_val, verbose=0), axis=1)

acc_3 = accuracy_score(y_true, y_pred_3)
prec_3 = precision_score(y_true, y_pred_3, average='weighted')
rec_3 = recall_score(y_true, y_pred_3, average='weighted')
f1_3 = f1_score(y_true, y_pred_3, average='weighted')
mcc_3 = matthews_corrcoef(y_true, y_pred_3)

print(f"\n✅ RESULTS (Fine-Tuning Last 30 Layers):")
print(f"  Accuracy:  {acc_3*100:.2f}%")
print(f"  Precision: {prec_3*100:.2f}%")
print(f"  Recall:    {rec_3*100:.2f}%")
print(f"  F1-Score:  {f1_3*100:.2f}%")
print(f"  MCC:       {mcc_3:.4f}")

train_acc_3 = history_3b.history['accuracy'][-1]
val_acc_3 = history_3b.history['val_accuracy'][-1]
gap_3 = (train_acc_3 - val_acc_3) * 100
print(f"  Train-Val Gap: {gap_3:.2f}%")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📊 COMPARISON SUMMARY")
print("="*80)

import pandas as pd

comparison_df = pd.DataFrame({
    'Strategy': [
        'No Fine-Tuning (Frozen)',
        'Fine-Tune Last 2 Layers',
        'Fine-Tune Last 30 Layers'
    ],
    'Trainable_Params_Phase1': [
        f"{trainable_params_1:,}",
        f"{trainable_params_1:,}",
        f"{trainable_params_1:,}"
    ],
    'Trainable_Params_Phase2': [
        f"{trainable_params_1:,} (no change)",
        f"{trainable_params_2:,}",
        f"{trainable_params_3:,}"
    ],
    'Accuracy': [
        f"{acc_1*100:.2f}%",
        f"{acc_2*100:.2f}%",
        f"{acc_3*100:.2f}%"
    ],
    'Precision': [
        f"{prec_1*100:.2f}%",
        f"{prec_2*100:.2f}%",
        f"{prec_3*100:.2f}%"
    ],
    'Recall': [
        f"{rec_1*100:.2f}%",
        f"{rec_2*100:.2f}%",
        f"{rec_3*100:.2f}%"
    ],
    'F1_Score': [
        f"{f1_1*100:.2f}%",
        f"{f1_2*100:.2f}%",
        f"{f1_3*100:.2f}%"
    ],
    'MCC': [
        f"{mcc_1:.4f}",
        f"{mcc_2:.4f}",
        f"{mcc_3:.4f}"
    ],
    'Train_Val_Gap': [
        f"{gap_1:.2f}%",
        f"{gap_2:.2f}%",
        f"{gap_3:.2f}%"
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv(RESULTS_PATH / "finetuning_comparison.csv", index=False)

# Determine winner
accuracies = [acc_1, acc_2, acc_3]
strategies = ['No Fine-Tuning', 'Fine-Tune Last 2 Layers', 'Fine-Tune Last 30 Layers']
winner_idx = np.argmax(accuracies)
winner = strategies[winner_idx]
winner_acc = accuracies[winner_idx]

print(f"\n" + "="*80)
print(f"🏆 WINNER: {winner}")
print(f"   Accuracy: {winner_acc*100:.2f}%")
print(f"   Improvement over frozen: +{(winner_acc - acc_1)*100:.2f}%")
print("="*80)

print(f"\n💡 KEY INSIGHTS:")
print(f"   1. Frozen base achieves: {acc_1*100:.2f}%")
print(f"   2. Fine-tuning 2 layers achieves: {acc_2*100:.2f}% ({'+' if acc_2>acc_1 else ''}{(acc_2-acc_1)*100:.2f}%)")
print(f"   3. Fine-tuning 30 layers achieves: {acc_3*100:.2f}% ({'+' if acc_3>acc_1 else ''}{(acc_3-acc_1)*100:.2f}%)")
print(f"\n   Overfitting risks:")
print(f"   - No fine-tuning: {gap_1:.2f}% gap {'✅ Good' if gap_1 < 10 else '⚠️ High'}")
print(f"   - 2 layers: {gap_2:.2f}% gap {'✅ Good' if gap_2 < 10 else '⚠️ High'}")
print(f"   - 30 layers: {gap_3:.2f}% gap {'✅ Good' if gap_3 < 10 else '⚠️ High'}")

print(f"\n📁 Results saved to: {RESULTS_PATH / 'finetuning_comparison.csv'}")
print("="*80)

