import pickle
import numpy as np
import pandas as pd

with open('results_checkpoint.pkl', 'rb') as f:
    ck = pickle.load(f)

all_results    = ck['all_results']
relieff_scores = ck['relieff_scores']

CLASSIFIERS = ['SVM', 'kNN', 'Decision Tree', 'Random Forest', 'MLP']

# ── Per-fold results (one row per dataset × fold × phase) ─────────────────────
rows = []
for ds, phases in all_results.items():
    for phase_key, phase_label in [('baseline', 'Phase1_Baseline'), ('relieff', 'Phase2_ReliefF')]:
        n_folds = len(phases[phase_key][CLASSIFIERS[0]]['acc'])
        for fi in range(n_folds):
            row = {'Dataset': ds, 'Phase': phase_label, 'Fold': f'Fold {fi+1}'}
            for clf in CLASSIFIERS:
                m = phases[phase_key][clf]
                acc_so_far = np.array(m['acc'][:fi+1])
                f1_so_far  = np.array(m['f1'][:fi+1])
                row[f'{clf}_Accuracy']     = round(m['acc'][fi], 6)
                row[f'{clf}_Accuracy_Std'] = round(acc_so_far.std(), 6)
                row[f'{clf}_F1']           = round(m['f1'][fi],  6)
                row[f'{clf}_F1_Std']       = round(f1_so_far.std(),  6)
                row[f'{clf}_Params']       = str(m['params'][fi])
            rows.append(row)

df_folds = pd.DataFrame(rows)
df_folds.to_csv('results_per_fold.csv', index=False)
print(f'Saved results_per_fold.csv  ({len(df_folds)} rows)')

# ── Summary (mean ± std per dataset × classifier × phase) ─────────────────────
rows = []
for ds, phases in all_results.items():
    for phase_key, phase_label in [('baseline', 'Phase1_Baseline'), ('relieff', 'Phase2_ReliefF')]:
        row = {'Dataset': ds, 'Phase': phase_label}
        for clf in CLASSIFIERS:
            accs = np.array(phases[phase_key][clf]['acc'])
            f1s  = np.array(phases[phase_key][clf]['f1'])
            row[f'{clf}_Acc_Mean'] = round(accs.mean(), 4)
            row[f'{clf}_Acc_Std']  = round(accs.std(),  4)
            row[f'{clf}_F1_Mean']  = round(f1s.mean(),  4)
            row[f'{clf}_F1_Std']   = round(f1s.std(),   4)
        rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv('results_summary.csv', index=False)
print(f'Saved results_summary.csv  ({len(df_summary)} rows)')

# ── ReliefF feature importances ───────────────────────────────────────────────
rows = []
for ds, scores in relieff_scores.items():
    for i, score in enumerate(scores):
        rows.append({'Dataset': ds, 'Feature': f'F{i+1}', 'ReliefF_Score': round(score, 6)})

df_relieff = pd.DataFrame(rows)
df_relieff.to_csv('results_relieff_scores.csv', index=False)
print(f'Saved results_relieff_scores.csv  ({len(df_relieff)} rows)')
