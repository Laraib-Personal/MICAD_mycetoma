"""
Automatic Results Table Updater
Scans for all result files and updates the comparison tables
"""

import json
import pandas as pd
from pathlib import Path
import glob

print("="*80)
print("UPDATING RESULTS TABLES")
print("="*80)

# ============================================================================
# SCAN FOR ALL RESULT FILES
# ============================================================================

results_data = []

# 1. Fine-tuning comparison results
print("\n📂 Checking fine-tuning comparison results...")
try:
    ft_df = pd.read_csv("results_finetuning_comparison/finetuning_comparison.csv")
    for idx, row in ft_df.iterrows():
        results_data.append({
            'Rank': idx + 1,
            'Experiment': f"InceptionV3 - {row['Strategy']}",
            'Dataset': 'Original Only',
            'Total_Images': 684,
            'Train_Size': 547,
            'Val_Size': 137,
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1_Score': row['F1_Score'],
            'MCC': row['MCC'],
            'Trainable_Params_Phase1': row['Trainable_Params_Phase1'],
            'Trainable_Params_Phase2': row['Trainable_Params_Phase2'],
            'Train_Val_Gap': row['Train_Val_Gap'],
            'Overfitting': 'No' if float(row['Train_Val_Gap'].strip('%')) < 10 else 'Yes',
            'Timestamp': '2025-10-27',
            'Notes': row['Strategy']
        })
    print(f"✅ Found {len(ft_df)} fine-tuning experiments")
except FileNotFoundError:
    print("⚠️  Fine-tuning comparison not found")

# 2. Original local training (baseline)
print("\n📂 Checking original baseline results...")
try:
    with open("results_inceptionv3/01_baseline/logs/inception_original_results.json") as f:
        orig_results = json.load(f)
    
    results_data.append({
        'Rank': len(results_data) + 1,
        'Experiment': 'InceptionV3 (Local - Earlier Experiment)',
        'Dataset': 'Original Only',
        'Total_Images': orig_results['total_images'],
        'Train_Size': orig_results['train_size'],
        'Val_Size': orig_results['val_size'],
        'Accuracy': f"{orig_results['metrics']['accuracy']*100:.2f}%",
        'Precision': f"{orig_results['metrics']['precision']*100:.2f}%",
        'Recall': f"{orig_results['metrics']['recall']*100:.2f}%",
        'F1_Score': f"{orig_results['metrics']['f1_score']*100:.2f}%",
        'MCC': f"{orig_results['metrics']['mcc']:.4f}",
        'Trainable_Params_Phase1': 'N/A',
        'Trainable_Params_Phase2': 'N/A',
        'Train_Val_Gap': f"{orig_results['overfitting_check']['train_val_gap']*100:.2f}%",
        'Overfitting': 'No' if not orig_results['overfitting_check']['overfitting'] else 'Yes',
        'Timestamp': orig_results['timestamp'],
        'Notes': 'Initial baseline - Less optimized hyperparameters'
    })
    print(f"✅ Found original local results")
except FileNotFoundError:
    print("⚠️  Original local results not found")

# 3. Augmented data with last 5 layers
print("\n📂 Checking augmented last 5-layer results...")
try:
    with open("results_inceptionv3/03_augmented_last5/logs/inception_aug5_results.json") as f:
        aug5_results = json.load(f)
    
    results_data.append({
        'Rank': len(results_data) + 1,
        'Experiment': 'InceptionV3 - Augmented - Fine-Tune Last 5 Layers',
        'Dataset': 'Original + Augmented',
        'Total_Images': aug5_results['dataset_info']['total_images'],
        'Train_Size': aug5_results['dataset_info']['train_size'],
        'Val_Size': aug5_results['dataset_info']['val_size'],
        'Accuracy': f"{aug5_results['test_metrics']['accuracy']*100:.2f}%",
        'Precision': f"{aug5_results['test_metrics']['precision']*100:.2f}%",
        'Recall': f"{aug5_results['test_metrics']['recall']*100:.2f}%",
        'F1_Score': f"{aug5_results['test_metrics']['f1_score']*100:.2f}%",
        'MCC': f"{aug5_results['test_metrics']['mcc']:.4f}",
        'Trainable_Params_Phase1': f"{aug5_results['trainable_parameters']['phase1']:,}",
        'Trainable_Params_Phase2': f"{aug5_results['trainable_parameters']['phase2']:,}",
        'Train_Val_Gap': f"{aug5_results['overfitting_analysis']['train_val_gap']:.2f}%",
        'Overfitting': 'No' if not aug5_results['overfitting_analysis']['overfitting_detected'] else 'Yes',
        'Timestamp': aug5_results['timestamp'],
        'Notes': 'Augmented data - Fine-tune last 5 layers'
    })
    print(f"✅ Found augmented last 5-layer results")
except FileNotFoundError:
    print("⚠️  Augmented last 5-layer results not found (run train_inception_augmented_5layers.py first)")

# 4. Augmented data with first 5 + last 5 layers
print("\n📂 Checking augmented first5+last5 results...")
try:
    with open("results_inceptionv3/04_augmented_first5_last5/logs/inception_aug_f5l5_results.json") as f:
        aug_f5l5_results = json.load(f)
    
    results_data.append({
        'Rank': len(results_data) + 1,
        'Experiment': 'InceptionV3 - Augmented - Fine-Tune First5+Last5',
        'Dataset': 'Original + Augmented',
        'Total_Images': aug_f5l5_results['dataset_info']['total_images'],
        'Train_Size': aug_f5l5_results['dataset_info']['train_size'],
        'Val_Size': aug_f5l5_results['dataset_info']['val_size'],
        'Accuracy': f"{aug_f5l5_results['test_metrics']['accuracy']*100:.2f}%",
        'Precision': f"{aug_f5l5_results['test_metrics']['precision']*100:.2f}%",
        'Recall': f"{aug_f5l5_results['test_metrics']['recall']*100:.2f}%",
        'F1_Score': f"{aug_f5l5_results['test_metrics']['f1_score']*100:.2f}%",
        'MCC': f"{aug_f5l5_results['test_metrics']['mcc']:.4f}",
        'Trainable_Params_Phase1': f"{aug_f5l5_results['trainable_parameters']['phase1']:,}",
        'Trainable_Params_Phase2': f"{aug_f5l5_results['trainable_parameters']['phase2']:,}",
        'Train_Val_Gap': f"{aug_f5l5_results['overfitting_analysis']['train_val_gap']:.2f}%",
        'Overfitting': 'No' if not aug_f5l5_results['overfitting_analysis']['overfitting_detected'] else 'Yes',
        'Timestamp': aug_f5l5_results['timestamp'],
        'Notes': 'Augmented data - Fine-tune first 5 + last 5 layers (10 total)'
    })
    print(f"✅ Found augmented first5+last5 results")
except FileNotFoundError:
    print("⚠️  Augmented first5+last5 results not found (run train_inception_augmented_first5_last5.py first)")

# ============================================================================
# CREATE UPDATED TABLES
# ============================================================================

print("\n" + "="*80)
print("CREATING UPDATED TABLES")
print("="*80)

# Create DataFrame
df = pd.DataFrame(results_data)

# Sort by accuracy
df['Accuracy_Num'] = df['Accuracy'].str.rstrip('%').astype(float)
df = df.sort_values('Accuracy_Num', ascending=False).reset_index(drop=True)
df['Rank'] = range(1, len(df) + 1)
df = df.drop('Accuracy_Num', axis=1)

# Save detailed table
df.to_csv('all_results_table.csv', index=False)
print(f"✅ Updated all_results_table.csv ({len(df)} experiments)")

# Create summary table
summary_df = df[['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
                 'MCC', 'Train_Val_Gap', 'Trainable_Params_Phase2']].copy()
summary_df.columns = ['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
                      'MCC', 'Train_Val_Gap', 'Trainable_Params']

# Add status column
def get_status(row):
    acc = float(row['Accuracy'].rstrip('%'))
    if acc >= 93:
        return 'Excellent (Clinical-grade)'
    elif acc >= 91:
        return 'Very Good (Research-grade)'
    elif acc >= 89:
        return 'Good'
    else:
        return 'Baseline'

summary_df['Status'] = summary_df.apply(get_status, axis=1)

summary_df.to_csv('all_results_summary.csv', index=False)
print(f"✅ Updated all_results_summary.csv")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "="*80)
print("CURRENT RESULTS RANKING")
print("="*80)
print()
print(df[['Rank', 'Experiment', 'Dataset', 'Total_Images', 'Accuracy', 
          'MCC', 'Train_Val_Gap', 'Overfitting']].to_string(index=False))

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*80)
print("STATISTICS")
print("="*80)

# Best result
best = df.iloc[0]
print(f"\n🏆 BEST RESULT:")
print(f"   Experiment: {best['Experiment']}")
print(f"   Accuracy:   {best['Accuracy']}")
print(f"   MCC:        {best['MCC']}")
print(f"   Dataset:    {best['Dataset']} ({best['Total_Images']} images)")

# Best for each dataset size
print(f"\n📊 BEST BY DATASET SIZE:")
for dataset_type in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset_type]
    if not subset.empty:
        best_subset = subset.iloc[0]
        print(f"   {dataset_type}:")
        print(f"     {best_subset['Experiment']}")
        print(f"     Accuracy: {best_subset['Accuracy']}")

# Overfitting analysis
no_overfit = df[df['Overfitting'] == 'No']
print(f"\n✅ NO OVERFITTING: {len(no_overfit)} experiments")
if not no_overfit.empty:
    best_no_overfit = no_overfit.iloc[0]
    print(f"   Best: {best_no_overfit['Experiment']} ({best_no_overfit['Accuracy']})")

# Check if augmented experiment exists
aug_exists = any('Augmented' in exp for exp in df['Experiment'])
if aug_exists:
    aug_row = df[df['Experiment'].str.contains('Augmented')].iloc[0]
    orig_best = df[df['Dataset'] == 'Original Only'].iloc[0]
    
    print(f"\n📈 AUGMENTATION IMPACT:")
    print(f"   Best with original (684 images):  {orig_best['Accuracy']}")
    print(f"   With augmented (2,736 images):    {aug_row['Accuracy']}")
    
    orig_acc = float(orig_best['Accuracy'].rstrip('%'))
    aug_acc = float(aug_row['Accuracy'].rstrip('%'))
    improvement = aug_acc - orig_acc
    print(f"   Improvement: {improvement:+.2f}%")

print("\n" + "="*80)
print("✅ TABLES UPDATED SUCCESSFULLY!")
print("="*80)
print(f"\n📁 Files updated:")
print(f"   - all_results_table.csv")
print(f"   - all_results_summary.csv")
print(f"\nTotal experiments: {len(df)}")
print("="*80)

