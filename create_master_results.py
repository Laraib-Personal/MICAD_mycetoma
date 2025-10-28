"""
Create Master Results CSV
Consolidates ALL InceptionV3 experiment results into one comprehensive CSV file
"""

import pandas as pd
import json
from pathlib import Path

print("="*80)
print("CREATING MASTER RESULTS CSV")
print("="*80)

# Initialize results list
all_results = []

# ============================================================================
# 1. BASELINE EXPERIMENT
# ============================================================================
print("\n📂 Loading Baseline results...")
try:
    with open("results_inceptionv3/01_baseline/logs/inception_original_results.json") as f:
        baseline = json.load(f)
    
    all_results.append({
        'Rank': len(all_results) + 1,
        'Experiment': 'InceptionV3 - Baseline (Early)',
        'Folder': 'results_inceptionv3/01_baseline/',
        'Dataset': 'Original Only',
        'Total_Images': baseline['total_images'],
        'Train_Size': baseline['train_size'],
        'Val_Size': baseline['val_size'],
        'Test_Size': 'N/A',
        'Batch_Size': 16,
        'Epochs': 40,
        'Learning_Rate': '1e-4',
        'Fine_Tuning_Strategy': 'None (frozen base)',
        'Layers_Unfrozen': 0,
        'Accuracy': f"{baseline['metrics']['accuracy']*100:.2f}%",
        'Precision': f"{baseline['metrics']['precision']*100:.2f}%",
        'Recall': f"{baseline['metrics']['recall']*100:.2f}%",
        'F1_Score': f"{baseline['metrics']['f1_score']*100:.2f}%",
        'MCC': f"{baseline['metrics']['mcc']:.4f}",
        'AUC': 'N/A',
        'Specificity': 'N/A',
        'Trainable_Params_Phase1': 'N/A',
        'Trainable_Params_Phase2': 'N/A',
        'Train_Val_Gap': f"{baseline['overfitting_check']['train_val_gap']*100:.2f}%",
        'Overfitting': 'No' if not baseline['overfitting_check']['overfitting'] else 'Yes',
        'Timestamp': baseline['timestamp'],
        'Status': 'Complete',
        'Notes': 'Initial baseline - Less optimized hyperparameters'
    })
    print(f"✅ Found baseline: {baseline['metrics']['accuracy']*100:.2f}%")
except Exception as e:
    print(f"⚠️  Baseline not found: {e}")

# ============================================================================
# 2. FINE-TUNING COMPARISON
# ============================================================================
print("\n📂 Loading Fine-Tuning Comparison results...")
try:
    ft_df = pd.read_csv("results_inceptionv3/02_finetuning_comparison/finetuning_comparison.csv")
    
    for idx, row in ft_df.iterrows():
        # Determine layers unfrozen
        if 'Frozen' in row['Strategy']:
            layers = 0
            strategy = 'No Fine-Tuning (Frozen)'
        elif '2 Layers' in row['Strategy']:
            layers = 2
            strategy = 'Fine-Tune Last 2 Layers'
        elif '30 Layers' in row['Strategy']:
            layers = 30
            strategy = 'Fine-Tune Last 30 Layers'
        else:
            layers = 'N/A'
            strategy = row['Strategy']
        
        all_results.append({
            'Rank': len(all_results) + 1,
            'Experiment': f'InceptionV3 - {strategy}',
            'Folder': 'results_inceptionv3/02_finetuning_comparison/',
            'Dataset': 'Original Only',
            'Total_Images': 684,
            'Train_Size': 547,
            'Val_Size': 137,
            'Test_Size': 137,
            'Batch_Size': 16,
            'Epochs': '50+50',
            'Learning_Rate': '1e-4, 1e-5',
            'Fine_Tuning_Strategy': strategy,
            'Layers_Unfrozen': layers,
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1_Score': row['F1_Score'],
            'MCC': row['MCC'],
            'AUC': 'N/A',
            'Specificity': 'N/A',
            'Trainable_Params_Phase1': row['Trainable_Params_Phase1'],
            'Trainable_Params_Phase2': row['Trainable_Params_Phase2'],
            'Train_Val_Gap': row['Train_Val_Gap'],
            'Overfitting': 'No' if float(row['Train_Val_Gap'].strip('%')) < 10 else 'Yes',
            'Timestamp': '2025-10-27',
            'Status': 'Complete',
            'Notes': f'{strategy} on original data'
        })
    
    print(f"✅ Found {len(ft_df)} fine-tuning experiments")
except Exception as e:
    print(f"⚠️  Fine-tuning comparison not found: {e}")

# ============================================================================
# 3. AUGMENTED - LAST 5 LAYERS
# ============================================================================
print("\n📂 Loading Augmented Last 5 Layers results...")
try:
    with open("results_inceptionv3/03_augmented_last5/logs/inception_aug5_results.json") as f:
        aug5 = json.load(f)
    
    all_results.append({
        'Rank': len(all_results) + 1,
        'Experiment': 'InceptionV3 - Augmented - Fine-Tune Last 5 Layers',
        'Folder': 'results_inceptionv3/03_augmented_last5/',
        'Dataset': 'Augmented (Original + Generated)',
        'Total_Images': aug5['dataset_info']['total_images'],
        'Train_Size': aug5['dataset_info']['train_size'],
        'Val_Size': aug5['dataset_info']['val_size'],
        'Test_Size': aug5['dataset_info']['test_size'],
        'Batch_Size': 16,
        'Epochs': '50+50',
        'Learning_Rate': '1e-4, 1e-5',
        'Fine_Tuning_Strategy': 'Fine-Tune Last 5 InceptionV3 Layers',
        'Layers_Unfrozen': 5,
        'Accuracy': f"{aug5['test_metrics']['accuracy']*100:.2f}%",
        'Precision': f"{aug5['test_metrics']['precision']*100:.2f}%",
        'Recall': f"{aug5['test_metrics']['recall']*100:.2f}%",
        'F1_Score': f"{aug5['test_metrics']['f1_score']*100:.2f}%",
        'MCC': f"{aug5['test_metrics']['mcc']:.4f}",
        'AUC': 'N/A',
        'Specificity': 'N/A',
        'Trainable_Params_Phase1': f"{aug5['trainable_parameters']['phase1']:,}",
        'Trainable_Params_Phase2': f"{aug5['trainable_parameters']['phase2']:,}",
        'Train_Val_Gap': f"{aug5['overfitting_analysis']['train_val_gap']:.2f}%",
        'Overfitting': 'No' if not aug5['overfitting_analysis']['overfitting_detected'] else 'Yes',
        'Timestamp': aug5['timestamp'],
        'Status': 'Complete',
        'Notes': 'Augmented data - Fine-tune last 5 layers of InceptionV3 base'
    })
    print(f"✅ Found augmented last 5: {aug5['test_metrics']['accuracy']*100:.2f}%")
except Exception as e:
    print(f"⚠️  Augmented last 5 not found: {e}")

# ============================================================================
# 4. AUGMENTED - FIRST 5 + LAST 5 LAYERS
# ============================================================================
print("\n📂 Loading Augmented First5+Last5 results...")
try:
    with open("results_inceptionv3/04_augmented_first5_last5/logs/inception_aug_f5l5_results.json") as f:
        aug_f5l5 = json.load(f)
    
    all_results.append({
        'Rank': len(all_results) + 1,
        'Experiment': 'InceptionV3 - Augmented - Fine-Tune First5+Last5',
        'Folder': 'results_inceptionv3/04_augmented_first5_last5/',
        'Dataset': 'Augmented (Original + Generated)',
        'Total_Images': aug_f5l5['dataset_info']['total_images'],
        'Train_Size': aug_f5l5['dataset_info']['train_size'],
        'Val_Size': aug_f5l5['dataset_info']['val_size'],
        'Test_Size': aug_f5l5['dataset_info']['test_size'],
        'Batch_Size': 16,
        'Epochs': '50+50',
        'Learning_Rate': '1e-4, 1e-5',
        'Fine_Tuning_Strategy': 'Fine-Tune First 5 + Last 5 InceptionV3 Layers',
        'Layers_Unfrozen': 10,
        'Accuracy': f"{aug_f5l5['test_metrics']['accuracy']*100:.2f}%",
        'Precision': f"{aug_f5l5['test_metrics']['precision']*100:.2f}%",
        'Recall': f"{aug_f5l5['test_metrics']['recall']*100:.2f}%",
        'F1_Score': f"{aug_f5l5['test_metrics']['f1_score']*100:.2f}%",
        'MCC': f"{aug_f5l5['test_metrics']['mcc']:.4f}",
        'AUC': 'N/A',
        'Specificity': 'N/A',
        'Trainable_Params_Phase1': f"{aug_f5l5['trainable_parameters']['phase1']:,}",
        'Trainable_Params_Phase2': f"{aug_f5l5['trainable_parameters']['phase2']:,}",
        'Train_Val_Gap': f"{aug_f5l5['overfitting_analysis']['train_val_gap']:.2f}%",
        'Overfitting': 'No' if not aug_f5l5['overfitting_analysis']['overfitting_detected'] else 'Yes',
        'Timestamp': aug_f5l5['timestamp'],
        'Status': 'Partial' if aug_f5l5['dataset_info']['total_images'] < 2500 else 'Complete',
        'Notes': 'Augmented data - Fine-tune first 5 + last 5 layers (10 total)'
    })
    print(f"✅ Found augmented first5+last5: {aug_f5l5['test_metrics']['accuracy']*100:.2f}%")
except Exception as e:
    print(f"⚠️  Augmented first5+last5 not found: {e}")

# 5. Augmented data with last 30 layers (BEST STRATEGY)
print("\n📂 Checking augmented last 30 layers results...")
try:
    with open("results_inceptionv3/05_augmented_last30/logs/inception_aug30_results.json") as f:
        aug30 = json.load(f)
    
    all_results.append({
        'Rank': len(all_results) + 1,
        'Experiment': 'InceptionV3 - Augmented - Fine-Tune Last 30 Layers',
        'Folder': 'results_inceptionv3/05_augmented_last30/',
        'Dataset': 'Augmented (Original + Generated)',
        'Total_Images': aug30['dataset_info']['total_images'],
        'Train_Size': aug30['dataset_info']['train_size'],
        'Val_Size': aug30['dataset_info']['val_size'],
        'Test_Size': aug30['dataset_info']['test_size'],
        'Batch_Size': 16,
        'Epochs': '50+50',
        'Learning_Rate': '1e-4, 1e-5',
        'Fine_Tuning_Strategy': 'Fine-Tune Last 30 InceptionV3 Layers (BEST)',
        'Layers_Unfrozen': 30,
        'Accuracy': f"{aug30['test_metrics']['accuracy']*100:.2f}%",
        'Precision': f"{aug30['test_metrics']['precision']*100:.2f}%",
        'Recall': f"{aug30['test_metrics']['recall']*100:.2f}%",
        'F1_Score': f"{aug30['test_metrics']['f1_score']*100:.2f}%",
        'MCC': f"{aug30['test_metrics']['mcc']:.4f}",
        'AUC': f"{aug30['test_metrics']['auc']:.3f}" if 'auc' in aug30['test_metrics'] else 'N/A',
        'Specificity': 'N/A',
        'Trainable_Params_Phase1': f"{aug30['trainable_parameters']['phase1']:,}",
        'Trainable_Params_Phase2': f"{aug30['trainable_parameters']['phase2']:,}",
        'Train_Val_Gap': f"{aug30['overfitting_analysis']['train_val_gap']:.2f}%",
        'Overfitting': 'No' if not aug30['overfitting_analysis']['overfitting_detected'] else 'Yes',
        'Timestamp': aug30['timestamp'],
        'Status': 'Complete',
        'Notes': 'Augmented data with BEST strategy (91.97% on original) - Expected best overall result'
    })
    print(f"✅ Found augmented last 30 layers: {aug30['test_metrics']['accuracy']*100:.2f}%")
except Exception as e:
    print(f"⚠️  Augmented last 30 layers not found: {e}")

# ============================================================================
# 6. ResNet50 Results
# ============================================================================
print("\n📂 Checking ResNet50 results...")
try:
    resnet_df = pd.read_csv("results_resnet50/resnet50_comparison.csv")
    for idx, row in resnet_df.iterrows():
        all_results.append({
            'Rank': len(all_results) + 1,
            'Experiment': f"ResNet50 - {row['Strategy']}",
            'Folder': f"results_resnet50/{row['Short_Name']}/",
            'Dataset': 'Augmented (Original + Generated)',
            'Total_Images': 2736,
            'Train_Size': 2189,
            'Val_Size': 274,
            'Test_Size': 273,
            'Batch_Size': 16,
            'Epochs': '50+50',
            'Learning_Rate': '1e-4, 1e-5',
            'Fine_Tuning_Strategy': row['Strategy'],
            'Layers_Unfrozen': row['Layers_Unfrozen'],
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1_Score': row['F1_Score'],
            'MCC': row['MCC'],
            'AUC': 'N/A',
            'Specificity': 'N/A',
            'Trainable_Params_Phase1': row['Trainable_Params_Phase1'],
            'Trainable_Params_Phase2': row['Trainable_Params_Phase2'],
            'Train_Val_Gap': row['Train_Val_Gap'],
            'Overfitting': row['Overfitting'],
            'Timestamp': 'N/A',
            'Status': 'Complete',
            'Notes': f'ResNet50 - {row["Strategy"]} on augmented data'
        })
    print(f"✅ Found {len(resnet_df)} ResNet50 experiments")
except Exception as e:
    print(f"⚠️  ResNet50 results not found: {e}")

# ============================================================================
# CREATE DATAFRAME AND SORT
# ============================================================================
print("\n" + "="*80)
print("CREATING MASTER CSV")
print("="*80)

df = pd.DataFrame(all_results)

# Sort by accuracy
df['Accuracy_Num'] = df['Accuracy'].str.rstrip('%').astype(float)
df = df.sort_values('Accuracy_Num', ascending=False).reset_index(drop=True)
df['Rank'] = range(1, len(df) + 1)
df = df.drop('Accuracy_Num', axis=1)

# Save master CSV
df.to_csv('master_results_inceptionv3.csv', index=False)
print(f"\n✅ Created master_results_inceptionv3.csv ({len(df)} experiments)")

# Print summary
print("\n" + "="*80)
print("MASTER RESULTS SUMMARY")
print("="*80)
print()
print(df[['Rank', 'Experiment', 'Dataset', 'Total_Images', 'Layers_Unfrozen', 
          'Accuracy', 'MCC', 'Train_Val_Gap', 'Status']].to_string(index=False))

# ============================================================================
# CREATE SIMPLIFIED VERSION
# ============================================================================
print("\n" + "="*80)
print("CREATING SIMPLIFIED VERSION")
print("="*80)

simple_df = df[['Rank', 'Experiment', 'Dataset', 'Total_Images', 'Layers_Unfrozen',
                'Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 
                'Train_Val_Gap', 'Status']].copy()

simple_df.to_csv('master_results_simple.csv', index=False)
print(f"✅ Created master_results_simple.csv (simplified version)")

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

best = df.iloc[0]
print(f"\n🏆 BEST RESULT:")
print(f"   {best['Experiment']}")
print(f"   Accuracy: {best['Accuracy']}")
print(f"   MCC: {best['MCC']}")
print(f"   Gap: {best['Train_Val_Gap']}")

# Best by dataset
print(f"\n📊 BEST BY DATASET:")
for dataset in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset]
    if not subset.empty:
        best_sub = subset.iloc[0]
        print(f"\n   {dataset}:")
        print(f"   └─ {best_sub['Experiment']}")
        print(f"      Accuracy: {best_sub['Accuracy']} | Layers: {best_sub['Layers_Unfrozen']}")

# Complete vs Pending
complete = df[df['Status'] == 'Complete']
partial = df[df['Status'] == 'Partial']
print(f"\n✅ COMPLETE: {len(complete)} experiments")
print(f"⏳ PARTIAL: {len(partial)} experiments")

print("\n" + "="*80)
print("✅ MASTER CSV FILES CREATED!")
print("="*80)
print(f"\n📁 Files created:")
print(f"   1. master_results_inceptionv3.csv  (comprehensive)")
print(f"   2. master_results_simple.csv       (simplified)")
print(f"\n📊 Total experiments consolidated: {len(df)}")
print("="*80)

