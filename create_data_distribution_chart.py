"""
High-Resolution Data Distribution Chart Generator
Creates publication-quality PNG charts showing original vs augmented data distribution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import seaborn as sns

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "dataset"
OUTPUT_PATH = PROJECT_ROOT / "data_distribution_charts"
OUTPUT_PATH.mkdir(exist_ok=True)

# High-resolution settings for publication
DPI = 300
FIG_SIZE = (16, 10)

# Data paths
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

# ============================================================================
# COUNT IMAGES
# ============================================================================

def count_images(directory, label=""):
    """Count JPG/JPEG images in directory"""
    directory = Path(directory)
    if not directory.exists():
        print(f"⚠️  Warning: {directory} does not exist")
        return 0
    
    count = 0
    for ext in ['*.jpg', '*.jpeg']:
        count += len(list(directory.rglob(ext)))
    
    # Filter out mask files
    all_images = []
    for ext in ['*.jpg', '*.jpeg']:
        all_images.extend(list(directory.rglob(ext)))
    
    filtered = [p for p in all_images if not p.stem.endswith('_mask')]
    
    if label:
        print(f"📊 {label}: {len(filtered)} images")
    
    return len(filtered)

print("\n" + "="*80)
print("COUNTING IMAGES IN DATASET")
print("="*80)

# Count original images
original_bm_count = count_images(ORIGINAL_BM, "Original BM")
original_fm_count = count_images(ORIGINAL_FM, "Original FM")
original_total = original_bm_count + original_fm_count

# Count augmented images (which includes original + augmented)
augmented_bm_total = count_images(AUGMENTED_BM, "Augmented BM (total)")
augmented_fm_total = count_images(AUGMENTED_FM, "Augmented FM (total)")
augmented_total = augmented_bm_total + augmented_fm_total

# Calculate augmented-only counts
augmented_only_bm = augmented_bm_total - original_bm_count
augmented_only_fm = augmented_fm_total - original_fm_count
augmented_only_total = augmented_only_bm + augmented_only_fm

print(f"\n📈 Summary:")
print(f"  Original BM:              {original_bm_count}")
print(f"  Original FM:              {original_fm_count}")
print(f"  Original Total:           {original_total}")
print(f"  Augmented-Only BM:        {augmented_only_bm}")
print(f"  Augmented-Only FM:        {augmented_only_fm}")
print(f"  Augmented-Only Total:     {augmented_only_total}")
print(f"  Augmented BM (total):     {augmented_bm_total}")
print(f"  Augmented FM (total):     {augmented_fm_total}")
print(f"  Augmented Total:          {augmented_total}")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

# Set font sizes for publication
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Color scheme
colors = {
    'BM_original': '#3498db',      # Blue
    'FM_original': '#e74c3c',      # Red
    'BM_augmented': '#5dade2',     # Light Blue
    'FM_augmented': '#ec7063',     # Light Red
}

# ============================================================================
# CHART 1: Comprehensive Stacked Bar Chart
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
fig.suptitle('Dataset Distribution: Original vs Augmented Images', 
             fontsize=22, fontweight='bold', y=0.98)

# Left plot: By Class (BM vs FM)
ax1 = axes[0]
categories = ['BM', 'FM']
original_counts = [original_bm_count, original_fm_count]
augmented_only_counts = [augmented_only_bm, augmented_only_fm]

x = np.arange(len(categories))
width = 0.6

bars1 = ax1.bar(x, original_counts, width, label='Original', 
                color=[colors['BM_original'], colors['FM_original']],
                edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x, augmented_only_counts, width, bottom=original_counts,
                label='Augmented', 
                color=[colors['BM_augmented'], colors['FM_augmented']],
                edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for i, bar in enumerate(bars):
        height = bar.get_height()
        bottom = bar.get_y()
        label_y = bottom + height/2
        if bars == bars2:
            total_height = original_counts[i] + augmented_only_counts[i]
            label_y = original_counts[i] + height/2
        
        ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{int(height)}',
                ha='center', va='center', fontweight='bold', fontsize=13)

# Add total labels at top
for i, (orig, aug) in enumerate(zip(original_counts, augmented_only_counts)):
    total = orig + aug
    ax1.text(i, total + 20, f'Total: {total}',
            ha='center', va='bottom', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('Class', fontweight='bold')
ax1.set_ylabel('Number of Images', fontweight='bold')
ax1.set_title('Distribution by Class', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(augmented_bm_total, augmented_fm_total) + 80)

# Right plot: By Type (Original vs Augmented)
ax2 = axes[1]
types = ['Original', 'Augmented']
bm_counts = [original_bm_count, augmented_only_bm]
fm_counts = [original_fm_count, augmented_only_fm]

x = np.arange(len(types))
width = 0.35

bars1 = ax2.bar(x - width/2, bm_counts, width, label='BM',
                color=colors['BM_original'], edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, fm_counts, width, label='FM',
                color=colors['FM_original'], edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=13)

# Add total labels at top
for i, (bm, fm) in enumerate(zip(bm_counts, fm_counts)):
    total = bm + fm
    ax2.text(i, total + 20, f'Total: {total}',
            ha='center', va='bottom', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2.set_xlabel('Image Type', fontweight='bold')
ax2.set_ylabel('Number of Images', fontweight='bold')
ax2.set_title('Distribution by Type', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(types)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(augmented_bm_total, augmented_fm_total) + 80)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file = OUTPUT_PATH / "01_comprehensive_distribution.png"
fig.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output_file}")
plt.close()

# ============================================================================
# CHART 2: Detailed Stacked Bar Chart
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

categories = ['BM\n(Bacterial Mycetoma)', 'FM\n(Fungal Mycetoma)']
x = np.arange(len(categories))
width = 0.7

# Stacked bars
bars_orig = ax.bar(x, original_counts, width, label='Original Images',
                   color=[colors['BM_original'], colors['FM_original']],
                   edgecolor='black', linewidth=2)
bars_aug = ax.bar(x, augmented_only_counts, width, bottom=original_counts,
                  label='Augmented Images (Generated)',
                  color=[colors['BM_augmented'], colors['FM_augmented']],
                  edgecolor='black', linewidth=2)

# Add value labels
for i, (orig_bar, aug_bar) in enumerate(zip(bars_orig, bars_aug)):
    orig_height = orig_bar.get_height()
    aug_height = aug_bar.get_height()
    
    # Label on original portion
    ax.text(orig_bar.get_x() + orig_bar.get_width()/2., orig_height/2,
            f'Original:\n{int(orig_height)}',
            ha='center', va='center', fontweight='bold', fontsize=12,
            color='white')
    
    # Label on augmented portion
    ax.text(aug_bar.get_x() + aug_bar.get_width()/2., 
            original_counts[i] + aug_height/2,
            f'Augmented:\n{int(aug_height)}',
            ha='center', va='center', fontweight='bold', fontsize=12,
            color='white')
    
    # Total at top
    total = original_counts[i] + augmented_only_counts[i]
    ax.text(i, total + 30, f'Total: {total}',
            ha='center', va='bottom', fontweight='bold', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9, edgecolor='black'))

ax.set_xlabel('Mycetoma Type', fontweight='bold', fontsize=16)
ax.set_ylabel('Number of Images', fontweight='bold', fontsize=16)
ax.set_title('Complete Dataset Distribution: Original vs Augmented Images',
             fontweight='bold', fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.legend(loc='upper left', framealpha=0.95, fontsize=13)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
ax.set_ylim(0, max(augmented_bm_total, augmented_fm_total) + 100)

# Add statistics text box
stats_text = f'Dataset Statistics:\n'
stats_text += f'• Original Total: {original_total} images\n'
stats_text += f'• Augmented Generated: {augmented_only_total} images\n'
stats_text += f'• Total Dataset: {augmented_total} images\n'
stats_text += f'• Augmentation Ratio: {augmented_only_total/original_total:.2f}x'

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round',
        facecolor='lightblue', alpha=0.8, edgecolor='black'))

plt.tight_layout()
output_file = OUTPUT_PATH / "02_detailed_stacked_chart.png"
fig.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_file}")
plt.close()

# ============================================================================
# CHART 3: Pie Charts Comparison
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Dataset Composition: Multiple Views', 
             fontsize=20, fontweight='bold', y=1.02)

# Pie Chart 1: Overall (Original vs Augmented)
ax1 = axes[0]
labels1 = ['Original\nImages', 'Augmented\nImages']
sizes1 = [original_total, augmented_only_total]
colors1 = ['#3498db', '#5dade2']
explode1 = (0.05, 0.05)

wedges1, texts1, autotexts1 = ax1.pie(sizes1, explode=explode1, labels=labels1,
                                       colors=colors1, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax1.set_title(f'Overall Distribution\n(Total: {augmented_total} images)', 
              fontweight='bold', fontsize=14, pad=15)

# Pie Chart 2: By Class (BM vs FM) - Original
ax2 = axes[1]
labels2 = ['BM\n(Bacterial)', 'FM\n(Fungal)']
sizes2 = [original_bm_count, original_fm_count]
colors2 = ['#3498db', '#e74c3c']
explode2 = (0.05, 0.05)

wedges2, texts2, autotexts2 = ax2.pie(sizes2, explode=explode2, labels=labels2,
                                       colors=colors2, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax2.set_title(f'Original Images\n(Total: {original_total} images)', 
              fontweight='bold', fontsize=14, pad=15)

# Pie Chart 3: By Class (BM vs FM) - Augmented
ax3 = axes[2]
labels3 = ['BM\n(Bacterial)', 'FM\n(Fungal)']
sizes3 = [augmented_bm_total, augmented_fm_total]
colors3 = ['#5dade2', '#ec7063']
explode3 = (0.05, 0.05)

wedges3, texts3, autotexts3 = ax3.pie(sizes3, explode=explode3, labels=labels3,
                                       colors=colors3, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts3:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax3.set_title(f'Augmented Dataset\n(Total: {augmented_total} images)', 
              fontweight='bold', fontsize=14, pad=15)

plt.tight_layout()
output_file = OUTPUT_PATH / "03_pie_charts_comparison.png"
fig.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_file}")
plt.close()

# ============================================================================
# CHART 4: Side-by-Side Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(2)  # BM and FM
width = 0.35

# Create grouped bars
bars_orig = ax.bar(x - width/2, [original_bm_count, original_fm_count], 
                   width, label='Original Images',
                   color=[colors['BM_original'], colors['FM_original']],
                   edgecolor='black', linewidth=1.5)
bars_aug_orig = ax.bar(x + width/2, [augmented_bm_total, augmented_fm_total], 
                       width, label='Augmented Dataset (Original + Generated)',
                       color=[colors['BM_augmented'], colors['FM_augmented']],
                       edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars_orig, bars_aug_orig]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_xlabel('Mycetoma Type', fontweight='bold', fontsize=16)
ax.set_ylabel('Number of Images', fontweight='bold', fontsize=16)
ax.set_title('Dataset Comparison: Original vs Augmented (Complete)', 
             fontweight='bold', fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['BM (Bacterial Mycetoma)', 'FM (Fungal Mycetoma)'], fontsize=14)
ax.legend(loc='upper left', framealpha=0.95, fontsize=13)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
ax.set_ylim(0, max(augmented_bm_total, augmented_fm_total) + 150)

# Add statistics annotations
stats_y = max(augmented_bm_total, augmented_fm_total) + 50
bm_increase = ((augmented_bm_total - original_bm_count) / original_bm_count) * 100
fm_increase = ((augmented_fm_total - original_fm_count) / original_fm_count) * 100

ax.annotate(f'+{int(augmented_bm_total - original_bm_count)} images\n({bm_increase:.1f}% increase)',
            xy=(0, augmented_bm_total), xytext=(0, stats_y),
            ha='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax.annotate(f'+{int(augmented_fm_total - original_fm_count)} images\n({fm_increase:.1f}% increase)',
            xy=(1, augmented_fm_total), xytext=(1, stats_y),
            ha='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

plt.tight_layout()
output_file = OUTPUT_PATH / "04_side_by_side_comparison.png"
fig.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_file}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ ALL CHARTS GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\n📁 Output Directory: {OUTPUT_PATH}")
print(f"\n📊 Generated Charts:")
print(f"  1. 01_comprehensive_distribution.png - Dual view (by class & by type)")
print(f"  2. 02_detailed_stacked_chart.png    - Detailed stacked bar chart")
print(f"  3. 03_pie_charts_comparison.png     - Three pie charts comparison")
print(f"  4. 04_side_by_side_comparison.png   - Side-by-side bar comparison")
print(f"\n⚙️  Settings:")
print(f"  • Resolution: {DPI} DPI (publication quality)")
print(f"  • Format: PNG")
print(f"  • Size: {FIG_SIZE[0]}\" x {FIG_SIZE[1]}\"")
print(f"\n✨ All charts are ready for extraction and use in papers/presentations!")

