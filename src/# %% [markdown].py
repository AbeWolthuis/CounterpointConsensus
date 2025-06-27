# %% [markdown]
# # Classification analysis

# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Baseline multinomial logistic regression: elbow on C + MI bars
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate



# %%


# %% [markdown]
# ## Data cleaning

# %%
# Data is located one folder up, in output/violations. (This is run inside a notebook)

RANDOM_SEED = 42
datapath = os.path.join(os.path.dirname(os.getcwd()), 'output', 'violations', 'violations_used_for_analysis_new.csv')
# datapath = os.path.join(os.path.dirname(os.getcwd()), 'output', 'violations', 'violations_output.csv')

df_uncleaned = pd.read_csv(datapath)

# Select only the columns we succesfully normalized (start with 'scaled_')
scaled_columns = [col for col in df_uncleaned.columns if col.startswith('scaled_')]
df_uncleaned = df_uncleaned[['jrpid', 'composer'] + scaled_columns]

# Remove the 'scaled_' prefix from the column names
df_uncleaned.columns = df_uncleaned.columns.str.replace('scaled_', '', regex=False)



# Merge columns that belong to the same composer
composer_map: dict[str, list[str]] = {
        "Busnoys": ['Busnoys, Antoine', "Busnoys, Antoine (COA)", "Busnoys, Antoine (COA1)"],
        "Compere": ['Compere, Loyset'],
        "Du Fay": ['Du Fay, Guillaume'],
        "Isaac": ['Isaac'],
        "Japart": ['Japart, Jean'],
        "Okeghem": ['Okeghem, Johannes', "Johannes Okeghem", "Okeghem, Johannes (COA)", "Okeghem, Johannes (COA1)"],
        "Josquin": ['Josquin des Prez', "Josquin des Prez (COA)"],
        "la Rue": ['la Rue, Pierre de', "La Rue"],
        "Martini": ['Martini, Johannes', "Martini, Johannes (COA)"],
        "Tinctoris": ['Tinctoris, Johannes'],
        "de Orto": ['de Orto, Marbrianus', "de Orto, Marbrianus (COA)"],
    }


# Drop rows with any nan values in the features
nan_count = df_uncleaned.isna().sum().sum()
print(f"Found {nan_count} NaN values in the dataset.\n")
df_uncleaned = df_uncleaned.dropna()


for canonical, alternatives in composer_map.items():
    # Replace any alternative composer label with the canonical one
    df_uncleaned.loc[df_uncleaned['composer'].isin(alternatives), 'composer'] = canonical

composer_counts = df_uncleaned['composer'].value_counts()
# Print composer counts
print(composer_counts)


# %%
dropped_composers = ["Tinctoris", "Du Fay", "Isaac", "Japart"]
df_uncleaned = df_uncleaned[~df_uncleaned['composer'].isin(dropped_composers)]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_uncleaned['composer'])
df_uncleaned['composer_encoded'] = y_encoded

# Store the mapping for later recovery
composer_classes = label_encoder.classes_

print(f"Encoded {len(composer_classes)} composers:")
composer_counts = df_uncleaned['composer'].value_counts()
for i, composer in enumerate(composer_classes):
    count = composer_counts[composer]
    print(f"  {i}: {composer} ({count} pieces)")

# %%
df_cleaned = df_uncleaned

# %% [markdown]
# ## Exploratory analysis

# %%
X = df_cleaned.drop(columns=['composer', 'jrpid', 'composer_encoded'])
y = df_cleaned['composer_encoded']


# %%
# compute composer-average feature matrix
avg_by_composer = df_cleaned.groupby('composer')[X.columns].mean()

# prepare feature numbers
feat_nos = [col.split(',')[-1] for col in avg_by_composer.columns]

# formatting widths
name_w = 25
num_w = 8

# header row: feature numbers
header = ' ' * name_w + ''.join(f"{fn:>{num_w}}" for fn in feat_nos)
print(header)

multiplier = 100

# one row per composer
# for composer, row in avg_by_composer.iterrows():
    # vals = ''.join(f"{multiplier*v:>{num_w}.3f}" for v in row.values)
    # print(f"{composer:<{name_w}}{vals}")
    
display(avg_by_composer.mul(1).round(5))

# %%
# --- Feature–label association strength  (Mutual Information) ---
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics         import mutual_info_score
from scipy.stats import pearsonr


# --------- Pearson ----------
# --- Summed absolute per‐composer Pearson correlations per feature ---
def plot_summed_abs_pearson(X: pd.DataFrame,
                            composers: list[str]) -> None:
    """
    Compute Pearson r(feature, each composer binary label),
    sum absolute values across composers, and bar‐plot the results.
    """
    # build matrix [n_features × n_composers]
    pearson_mat = np.zeros((X.shape[1], len(composers)))
    for j, comp in enumerate(composers):
        y_bin = (df_cleaned["composer"] == comp).astype(int)
        for i, feat in enumerate(X.columns):
            pearson_mat[i, j] = pearsonr(X[feat], y_bin)[0]

    # sum absolute correlations per feature
    abs_sum = np.sum(np.abs(pearson_mat), axis=1)
    df_sum = pd.DataFrame({
        "feature": X.columns,
        "sum_abs_r": abs_sum
    }).sort_values("sum_abs_r", ascending=False)

    # plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_sum,
        x="sum_abs_r", y="feature",
        palette="Greens"
    )
    plt.xlabel("Sum of |Pearson r| across composers")
    plt.ylabel("Feature")
    plt.title("Feature Discriminability: Σ |r(feature, composer)|")
    plt.tight_layout()
    plt.show()

# call the function
plot_summed_abs_pearson(X, composer_classes)


# Heatmap

pearson_matrix = np.zeros((len(X.columns), len(composer_classes)))

for j, comp_label in enumerate(composer_classes):
    y_bin = (df_cleaned['composer'] == comp_label).astype(int)
    for i, col in enumerate(X.columns):
        pearson_matrix[i,j] = pearsonr(X[col], y_bin)[0]

plt.figure(figsize=(8,6))
sns.heatmap(
    pearson_matrix,
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label':'Pearson r'},
    xticklabels=composer_classes,
    yticklabels=X.columns
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.title("Per‐composer Pearson correlation")
plt.tight_layout()
plt.show()


# --------- Mutual information ----------

mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
assoc_df = (
    pd.DataFrame({"feature": X.columns, "MI": mi})
      .sort_values("MI", ascending=False)
)

# Bar chart with Greens colormap
plt.figure(figsize=(10, 6))
bars = plt.bar(assoc_df["feature"], assoc_df["MI"])

# Apply Greens colormap to bars based on MI values, not position
# Normalize MI values to 0-1 range for colormap
mi_normalized = (assoc_df["MI"] - assoc_df["MI"].min()) / (assoc_df["MI"].max() - assoc_df["MI"].min())
colors = plt.cm.Greens(mi_normalized * 0.7 + 0.3)  # Scale to 0.3-1.0 range to avoid too light colors

for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.xticks(
    rotation=45,
    ha='right',
    rotation_mode='anchor',
    fontsize=9,
)
plt.ylabel("Mutual Information")
plt.title("Composer association strength - Mutual Information")
plt.tight_layout()
plt.show()


# --- Per‐composer MI heat-map using actual composer names ---
mi_matrix = np.zeros((len(X.columns), len(composer_classes)))
for j, comp_label in enumerate(composer_classes):
    # binarize y for this composer
    y_bin = (df_cleaned["composer"] == comp_label).astype(int)
    for i, col in enumerate(X.columns):
        mi_matrix[i, j] = mutual_info_score(
            pd.qcut(X[col],
                    q=min(10, len(X)//3),
                    duplicates="drop"),
            y_bin
        )

# Heatmap with Greens colormap
plt.figure(figsize=(8, 6))
sns.heatmap(
    mi_matrix,
    cmap='Greens',  # Changed to Greens colormap
    cbar_kws={"label": "Mutual Information"},
    xticklabels=composer_classes,
    yticklabels=X.columns
)
plt.xticks(
    rotation=45,
    ha='right',
    rotation_mode='anchor',
    fontsize=9,
)
plt.yticks(rotation=0, fontsize=9)
plt.title("Per-composer MI heat-map")
plt.tight_layout()
plt.show()

# %%
# Add this cell to your classification.ipynb notebook

import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# --- Correlation Analysis ---
# Calculate correlation matrix
corr_matrix = X.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    cmap='RdBu_r', 
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={"shrink": .8},
    annot_kws={'size': 8}
)

plt.title('Feature Correlation Matrix\n(Normalized Counterpoint Rule Counts)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()

# --- Summary statistics ---
# Get upper triangle of correlation matrix (excluding diagonal)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
correlations = upper_tri.stack().values

print(f"\nCorrelation Summary:")
print(f"  Mean absolute correlation: {np.mean(np.abs(correlations)):.3f}")
print(f"  Max correlation: {np.max(correlations):.3f}")
print(f"  Min correlation: {np.min(correlations):.3f}")
print(f"  # correlations > 0.7: {np.sum(np.abs(correlations) > 0.7)}")
print(f"  # correlations > 0.5: {np.sum(np.abs(correlations) > 0.5)}")

# --- Identify highly correlated pairs ---
high_corr_threshold = 0.
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > high_corr_threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_val
            ))

if high_corr_pairs:
    print(f"\nHighly correlated pairs (|r| > {high_corr_threshold}):")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1} ↔ {feat2}: r = {corr:.3f}")
else:
    print(f"\nNo highly correlated pairs found (|r| > {high_corr_threshold})")

# %%
# For all the highly correlated (>0.8) feature pairs, remove one of them, based on lowest MI
X = X.drop(columns=['ascending_leap_to_from_quarter, 35'])
y = y

# %%



corr_matrix = X.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    cmap='RdBu_r', 
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={"shrink": .8},
    annot_kws={'size': 8}
)

plt.title('Feature Correlation Matrix\n(Normalized Counterpoint Rule Counts)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()

# --- Summary statistics ---
# Get upper triangle of correlation matrix (excluding diagonal)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
correlations = upper_tri.stack().values

print(f"\nCorrelation Summary:")
print(f"  Mean absolute correlation: {np.mean(np.abs(correlations)):.3f}")
print(f"  Max correlation: {np.max(correlations):.3f}")
print(f"  Min correlation: {np.min(correlations):.3f}")
print(f"  # correlations > 0.7: {np.sum(np.abs(correlations) > 0.7)}")
print(f"  # correlations > 0.5: {np.sum(np.abs(correlations) > 0.5)}")

# --- Identify highly correlated pairs ---
high_corr_threshold = 0.7
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > high_corr_threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_val
            ))

if high_corr_pairs:
    print(f"\nHighly correlated pairs (|r| > {high_corr_threshold}):")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1} ↔ {feat2}: r = {corr:.3f}")
else:
    print(f"\nNo highly correlated pairs found (|r| > {high_corr_threshold})")


# %% [markdown]
# ## Create train/val datasets

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Train-validation split (stratified by composer) ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_SEED, 
    stratify=y
)

# %% [markdown]
# ## Classification baseline - logistic regression

# %% [markdown]
# ### Parameter tuning

# %%
# --- Build pipeline ---

# 1. Settings
class_weight = 'balanced'
n_splits = 5
parameter_value_tolerance = 0.01
cv = StratifiedKFold(
    n_splits=n_splits, 
    shuffle=True,
    random_state=RANDOM_SEED
)

# 2. Define pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        #penalty="l1",
        penalty='elasticnet', l1_ratio=0.5,
        solver="saga",
        multi_class="ovr",
        max_iter=5000,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        tol=parameter_value_tolerance,               
        warm_start=True,
        n_jobs=-1
    ))
])

# Prepare storage
C_values = np.logspace(-3, 0, 12)
f1_macro = []
balanced_acc = []
f1_train = []  # NEW: training scores
balanced_acc_train = []  # NEW: training scores
features_mean = []
features_std = []

for C in tqdm(C_values, desc="Hyperparameter search"):
    pipe.set_params(clf__C=C, clf__warm_start=True)

    # Get both train and validation scores
    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring={"f1": "f1_macro", "bal": "balanced_accuracy"},
        return_estimator=True,
        return_train_score=True,  # NEW: get training scores
        n_jobs=-1
    )

    # Validation scores (what we had before)
    f1_macro.append(scores["test_f1"].mean())
    balanced_acc.append(scores["test_bal"].mean())
    
    # NEW: Training scores
    f1_train.append(scores["train_f1"].mean())
    balanced_acc_train.append(scores["train_bal"].mean())

    # Count non-zero features in each fold’s model
    nonzero_counts = []
    for est in scores["estimator"]:
        coef = est.named_steps["clf"].coef_
        # count features with any class coefficient ≠ 0
        nonzero_counts.append(np.sum(np.any(coef != 0, axis=0)))

    # store mean and std
    features_mean.append(np.mean(nonzero_counts))
    features_std.append(np.std(nonzero_counts))

    



# %%
plt.errorbar(
    C_values, features_mean, yerr=features_std, 
    color='green', marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel('C (inverse regularization strength)')
plt.ylabel('Avg. # non-zero features')
plt.title('Sparsity path vs C')
plt.tight_layout()
plt.show()

# %%
# --- determine elbow points to plot ---
f1_elbow_idx      = 4
bal_acc_elbow_idx = 4
feature_elbow_idx = 4

# get the C and y-values at each elbow
C_feat, y_feat = C_values[feature_elbow_idx], features_mean[feature_elbow_idx]
C_f1,   y_f1   = C_values[f1_elbow_idx],     f1_macro[f1_elbow_idx]
C_bal,  y_bal  = C_values[bal_acc_elbow_idx], balanced_acc[bal_acc_elbow_idx]

# --- create figure ---
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xscale('log')
ax1.set_xlabel('Regularisation strength $C$')
ax1.set_ylabel('# included features', color='black')

# --- plot sparsity with error bars ---
ax1.errorbar(
    C_values, features_mean,
    yerr=features_std,
    marker='D', linestyle='-',
    label='Included features (mean ± std)',
    color='green'
)
ax1.set_ylim(0, max(features_mean) * 1.1)
ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

ax1.set_ylim(0, max(features_mean) * 1.1)
ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

# Secondary axis
ax2 = ax1.twinx()

# Plot f1 train and test (blue)
ax2.plot(C_values, f1_macro, 'o-', label='F1-macro (validation)', color='tab:blue')
ax2.plot(C_values, f1_train, 'o--', label='F1-macro (training)', color='tab:blue', alpha=0.7)

# Balanced Accuracy = (Sensitivity + Specificity) / 2.
# Plot balanced accuracy train and test (orange)
# ax2.plot(C_values, balanced_acc, 's-', label='Balanced accuracy (validation)', color='tab:orange')
# ax2.plot(C_values, balanced_acc_train, 's--', label='Balanced accuracy (training)', color='tab:orange', alpha=0.7)

ax2.set_ylabel('Score')
ax2.set_ylim(0, 1.05)

# Highlight elbow points
ax1.scatter([C_feat], [y_feat], s=200, facecolors='none', edgecolors='red', linewidths=2, zorder=5)

# Print and legend (unchanged)
print("Elbow analysis results:")
print(f"  Features elbow:     C={C_feat:.3g}, #feat≈{y_feat:.1f}±{features_std[feature_elbow_idx]:.1f}")
print(f"  F1-macro elbow:     C={C_f1:.3g}, score={y_f1:.3f}")
print(f"  Balanced-acc elbow: C={C_bal:.3g}, score={y_bal:.3f}")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Model performance & sparsity vs regularisation strength')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Train full model, run on test set

# %%
optimal_C = C_values[feature_elbow_idx]  # or choose based on your elbow analysis
pipe.set_params(clf__C=optimal_C)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_val)
val_f1 = f1_score(y_val, y_pred, average='macro')
val_bal_acc = balanced_accuracy_score(y_val, y_pred)

# --- Detailed classification report ---
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=composer_classes))

# %%
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_with_percentages(
        y_true, y_pred, class_names, title="Confusion Matrix",
        print_stats=True
        ):
    """
    Plot a confusion matrix heatmap with only diagonal percentage annotations,
    and a color scale fixed from 0% to 100%.
    """
    # Compute confusion matrix and row-wise percentages
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap with fixed color range
    sns.heatmap(
        cm_percent,
        annot=False,
        fmt='',
        cmap='Greens',
        vmin=0,            # minimum of colorbar
        vmax=100,          # maximum of colorbar
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Row-wise %"},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Overlay diagonal percentage annotations
    for i in range(cm.shape[0]):
        pct = cm_percent[i, i]
        ax.text(
            i + 0.5, i + 0.5,
            f"{pct:.1f}%",
            ha='center', va='center',
            color='white' if pct > 50 else 'black',
            fontsize=12
        )

    plt.tight_layout()

    # Print summary statistics
    if print_stats:
        total = cm.sum()
        acc = np.trace(cm) / total
        print(f"{title} Summary:")
        print(f"  Overall Accuracy: {acc:.3f}")
        print("  Per-class accuracy (diagonal %):")
        for i, name in enumerate(class_names):
            support = cm[i].sum()
            print(f"    {name:<10}: {cm_percent[i, i]:5.1f}%  ({support} samples)")

    return cm, cm_percent



# Create the confusion matrix
cm, cm_percent = plot_confusion_matrix_with_percentages(
    y_val, y_pred, composer_classes, 
    title=None
)

plt.show()

# cm_percent_df = pd.DataFrame(cm_percent,index=composer_classes,columns=composer_classes).round(1)
# display(cm_percent_df)



# --- Additional Analysis ---

# Most confused pairs
print(f"\nMost frequent misclassifications:")
for i in range(len(composer_classes)):
    for j in range(len(composer_classes)):
        if i != j and cm[i, j] > 0:
            misclass_rate = cm[i, j] / cm[i, :].sum() * 100
            if misclass_rate > 10:  # Show misclassifications > 10%
                print(f"  {composer_classes[i]} → {composer_classes[j]}: {cm[i, j]} pieces ({misclass_rate:.1f}%)")


# %%
coef = pipe.named_steps["clf"].coef_
active_mask = np.any(np.abs(coef) > parameter_value_tolerance, axis=0)

# pick an aggregation rule once and reuse it
agg = np.mean(np.abs(coef), axis=0)      # or np.max / np.sum

importance_df = (
    pd.DataFrame({'feature': X_train.columns,
                  'importance': agg,
                  'active': active_mask})
      .sort_values('importance', ascending=False)
      .reset_index(drop=True)
)
#importance_df 

print(f"Active features at C={optimal_C:.3g}: {active_mask.sum()}/{X_train.shape[1]}")

print("\nTop features by |coef|:")
print(importance_df.head(20).to_string(index=False))

# %%
from sklearn.inspection import permutation_importance

result = permutation_importance(
    pipe, X_val, y_val, n_repeats=30, random_state=0, n_jobs=-1
)
perm_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)
print(perm_df.head(10))


# %% [markdown]
# # Decision tree

# %%
import math

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

'''
Dont use the trained model above for feature selection bceause:
1. Data-leakage in evaluation
When you compute coefficients on the full training set and then decide which features to keep, you have already “peeked” at every sample.
2. Model-specific importance ≠ universal importance 
Logistic coefficients describe linear marginal effects conditioned on all other variables. Decision-trees exploit non-linear, interaction terms
3. Shrunken values are shrunk by design 
Your elastic-net intentionally drives many β → 0 to combat multicollinearity. The ranking therefore reflects regularisation artefacts as much as true relevance. Heavy-correlated blocks often keep one survivor with an inflated |β| and suppress its twins – a tree would happily keep either twin if it helps a split. 
'''

# parse logistic model (same hyper-params as earlier elbow)
lasso = LogisticRegression(
    penalty="l1", solver="saga", multi_class="ovr",
    C=optimal_C, max_iter=5000, class_weight="balanced",
    random_state=RANDOM_SEED, n_jobs=-1
)

# 2 meta-selector drops weak-importance features                       ⬇️
selector = SelectFromModel(lasso, threshold="median")  # keep top-50 %

# 3regularised decision tree (early-stopping + cost-complexity)
tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=RANDOM_SEED
)

pipe_dt = Pipeline([
    ("selector", selector),
    ("tree",     tree)
])



# --- dynamically compute max_depth as 10%–100% of feature count, rounded up
n_feat = X_train.shape[1]
depths = sorted({ math.ceil(n_feat * frac)
                  for frac in np.linspace(0.1, 1.0, 10) })
print(f"Grid-search will try tree__max_depth = {depths}")

param_grid = {
    "tree__max_depth":       depths,
    "tree__min_samples_leaf":[1, 2, 5, 10],
    "tree__ccp_alpha":       [0.0, 1e-3, 1e-2]      # post-pruning
}

gs_dt = GridSearchCV(
    pipe_dt, param_grid, cv=cv,
    scoring="f1_macro", n_jobs=-1, refit=True
)
gs_dt.fit(X_train, y_train)

print("Best tree params:", gs_dt.best_params_)
print(f"Val-F1: {gs_dt.best_score_:.3f}")

# 4. evaluate on hold-out validation set
y_pred_dt = gs_dt.predict(X_val)
print("\nDecision-tree report:")
print(classification_report(y_val, y_pred_dt, target_names=composer_classes))

# %%
# --- Build legend from the colours actually used by plot_tree ---------
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from sklearn.tree import _tree  

# Retrieve names of selected features for plotting
sel_feat = X_train.columns[gs_dt.best_estimator_.named_steps["selector"].get_support()]

# Visualise
plt.figure(figsize=(20, 12))

# Call plot_tree and store the returned list of artists
node_annotations = plot_tree(
    gs_dt.best_estimator_['tree'], # Adjusted for mock object
    feature_names=sel_feat,
    class_names=composer_classes,
    filled=True, 
    rounded=True,
    max_depth=3,
    impurity=False,
    proportion=False,
    fontsize=10
)

for ann in node_annotations:
    text = ann.get_text()
    
    # 1. Correctly split by the actual newline character '\n'
    lines = text.split('\n')
    
    # 2. Use a more robust check ('in') to find the line to remove
    new_lines = [line for line in lines if 'value =' not in line]
    
    # 3. Join the lines back with the newline character '\n'
    new_text = '\n'.join(new_lines)
    
    # Set the modified text back to the annotation
    ann.set_text(new_text)

tree      = gs_dt.best_estimator_.named_steps["tree"]
class_ids = tree.classes_  
                     # order used by plot_tree

tree_struct = tree.tree_
leaf_mask   = tree_struct.children_left == _tree.TREE_LEAF   # Boolean array

pure_hues = {}                                               # {label: RGB}
for ann, is_leaf in zip(node_annotations, leaf_mask):
    if not is_leaf:
        continue                                             # skip non-leaf

    # grab "class = …" line if present
    cls_line = next((l for l in ann.get_text().split('\n')
                     if 'class =' in l), None)
    if cls_line is None:
        continue                                             # defensive: no label

    class_name = cls_line.split('=', 1)[1].strip()

    # fully-saturated colour of this leaf
    rgb = ann.get_bbox_patch().get_facecolor()[:3]           # (r,g,b) 0-1
    pure_hues.setdefault(class_name, rgb)                    # keep first hit

# --- build legend ------------------------------------------------------
legend_patches = [
    mpatches.Patch(color=to_hex(rgb), label=cls)
    for cls, rgb in pure_hues.items()
]
plt.legend(handles=legend_patches,
           title="Pure-leaf hues",
           loc="upper left", frameon=True, edgecolor='k',
           fontsize='large', title_fontsize='x-large')


# %% [markdown]
# # SVM

# %%


# %%
###################3

# %%
from sklearn.svm             import LinearSVC
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import cross_validate
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from tqdm.auto import tqdm

# 0️⃣ pipeline skeleton
base_pipe = Pipeline([
    ("scale", StandardScaler(with_mean=False)),
    ("svc",   LinearSVC(
        penalty="l1", dual=False, class_weight="balanced",
        max_iter=5000, random_state=RANDOM_SEED, multi_class="ovr"
    ))
])

# 1️⃣ hyper-parameter sweep
C_grid = np.logspace(-3, 0, 8)     # 0.001 … 10
f1_val, f1_tr = [], []
nz_mean, nz_std = [], []

for C in tqdm(C_grid, desc="LinearSVC-L1 grid"):
    base_pipe.set_params(svc__C=C)
    scores = cross_validate(
        base_pipe, X_train, y_train,
        cv=cv, n_jobs=-1,
        scoring="f1_macro",
        return_train_score=True,
        return_estimator=True
    )

    f1_val.append(scores["test_score"].mean())
    f1_tr.append(scores["train_score"].mean())

    # sparsity — non-zero features
    n_active = [np.count_nonzero(est.named_steps["svc"].coef_.any(axis=0))
                for est in scores["estimator"]]
    nz_mean.append(np.mean(n_active))
    nz_std.append(np.std(n_active))


# %%


chosen_C = C_grid[5]                      # ← your manual choice
idx_sel  = np.where(C_grid == chosen_C)[0][0]   # index in the grid
y_feat   = nz_mean[idx_sel]                       # y-coord = sparsity curve
print(f"\nChosen C = {chosen_C:.4g}")

# 2️⃣ plot CV results BEFORE final fit
fig, ax1 = plt.subplots(figsize=(9,5))
ax1.set_xscale('log')
ax1.set_xlabel('Margin penalty $C$')
ax1.set_ylabel('# active features', color='black')

ax1.errorbar(C_grid, nz_mean, yerr=nz_std,
             marker='D', color='green', linestyle='-',
             label='Included features (mean ± std)')
ax1.grid(True, linestyle='--', linewidth=0.5)

ax2 = ax1.twinx()
ax2.plot(C_grid, f1_val, 'o-', label='F1-macro (validation)', color='tab:blue')
ax2.plot(C_grid, f1_tr,  'o--', label='F1-macro (training)',   color='tab:blue', alpha=0.6)
ax2.set_ylabel('F1-macro')
ax2.set_ylim(0, 1.05)

# --- draw a red hollow circle around the selected C -------------------
ax1.scatter([chosen_C], [y_feat],
            s=200, facecolors='none', edgecolors='red',
            linewidths=2, zorder=5)

# merge legends
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left')

plt.title('SVM-L1 — CV')
plt.tight_layout()
plt.show()

# %%

# Run on validation set
final_pipe = base_pipe.set_params(svc__C=chosen_C).fit(X_train, y_train)
y_pred = final_pipe.predict(X_val)


print("\nValidation report (Linear SVM-L1):")
print(classification_report(y_val, y_pred, target_names=composer_classes))

cm, _ = plot_confusion_matrix_with_percentages(
    y_val, y_pred, composer_classes, title="SVM (L1-linear)", print_stats=False
)
plt.show()

# -------- sparsity diagnostics ---------------------------------------
coef_abs = np.abs(final_pipe.named_steps["svc"].coef_)
feat_mask = coef_abs.any(axis=0)
print(f"\nSelected {feat_mask.sum()} / {feat_mask.size} features "
      f"({feat_mask.sum()/feat_mask.size:.1%}).")

for idx, name in enumerate(composer_classes):
    weights = pd.Series(final_pipe.named_steps["svc"].coef_[idx],
                        index=X_train.columns)
    display(weights.abs().sort_values(ascending=False).head(5)
            .rename(f"Top |coef| for {name}"))

# %%



