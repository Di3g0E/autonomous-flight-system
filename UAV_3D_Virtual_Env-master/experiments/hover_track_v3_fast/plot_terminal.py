import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import collections
from pathlib import Path
import os

data = """
    Ep  4  R=  8681.7  vis=100.0%  cent=0.276  frac=0.1639  [OK]
    Ep  5  R=  4973.9  vis=100.0%  cent=0.508  frac=0.1347  [OK]
    Ep  6  R=  5574.2  vis= 98.9%  cent=0.409  frac=0.0366  [OK]
    Ep  7  R= 10892.5  vis=100.0%  cent=0.132  frac=0.2408  [OK]
    Ep  8  R= 10580.6  vis=100.0%  cent=0.176  frac=0.2281  [OK]
    Ep  9  R=  6374.1  vis=100.0%  cent=0.413  frac=0.1140  [OK]
    Ep 10  R=  7518.0  vis= 99.9%  cent=0.370  frac=0.1648  [OK]
  -- MEDIUM (off=0.6m vel=0.25 ang=0.1) --
    Ep 11  R=  1726.4  vis= 77.6%  cent=0.667  frac=0.1048  [EARLY]
    Ep 12  R=  9842.4  vis= 98.5%  cent=0.221  frac=0.2172  [OK]
    Ep 13  R=  1884.8  vis= 52.7%  cent=0.623  frac=0.0686  [EARLY]
    Ep 14  R=  1804.3  vis= 53.6%  cent=0.678  frac=0.0486  [EARLY]
    Ep 15  R=  7260.8  vis=100.0%  cent=0.397  frac=0.2009  [OK]
    Ep 16  R=   995.6  vis= 91.7%  cent=0.649  frac=0.1079  [EARLY]
    Ep 17  R=   836.9  vis= 92.7%  cent=0.726  frac=0.0939  [EARLY]
    Ep 18  R=  4290.3  vis= 94.2%  cent=0.647  frac=0.1103  [OK]
    Ep 19  R=  6158.2  vis= 90.3%  cent=0.439  frac=0.1608  [OK]
    Ep 20  R=  1643.2  vis= 54.8%  cent=0.786  frac=0.0577  [OK]
  -- HARD (off=1.0m vel=0.35 ang=0.15) --
    Ep 21  R=  3128.2  vis= 85.1%  cent=0.699  frac=0.0587  [OK]
    Ep 22  R=  2271.8  vis= 68.2%  cent=0.732  frac=0.0683  [OK]
    Ep 23  R=  5810.3  vis= 87.9%  cent=0.477  frac=0.1402  [OK]
    Ep 24  R=  4682.0  vis= 87.8%  cent=0.526  frac=0.0750  [OK]
    Ep 25  R=  3878.7  vis= 74.0%  cent=0.464  frac=0.0380  [OK]
    Ep 26  R=  5432.1  vis= 84.0%  cent=0.419  frac=0.0541  [OK]
    Ep 27  R=  3987.9  vis= 79.7%  cent=0.511  frac=0.1874  [EARLY]
    Ep 28  R=    38.8  vis= 33.8%  cent=1.025  frac=0.0261  [EARLY]
    Ep 29  R=   -77.9  vis=  3.0%  cent=1.215  frac=0.0097  [OK]
    Ep 30  R=   381.2  vis= 46.1%  cent=0.804  frac=0.0706  [EARLY]

============================================================
  Evaluating: model_850000_steps.zip  (850,000 steps)
============================================================
  -- EASY (off=0.2m vel=0.1 ang=0.05) --
    Ep  1  R=  9003.1  vis=100.0%  cent=0.278  frac=0.2394  [OK]
    Ep  2  R=  3622.2  vis=100.0%  cent=0.440  frac=0.1090  [EARLY]
    Ep  3  R=  7570.1  vis=100.0%  cent=0.384  frac=0.2267  [OK]
    Ep  4  R=  2692.0  vis=100.0%  cent=0.367  frac=0.1104  [EARLY]
    Ep  5  R=  8235.4  vis=100.0%  cent=0.321  frac=0.1998  [OK]
    Ep  6  R=  9345.1  vis=100.0%  cent=0.253  frac=0.2425  [OK]
    Ep  7  R=  9837.3  vis=100.0%  cent=0.219  frac=0.2458  [OK]
    Ep  8  R=  9405.9  vis=100.0%  cent=0.246  frac=0.2623  [OK]
    Ep  9  R=  1826.3  vis=100.0%  cent=0.351  frac=0.1015  [EARLY]
    Ep 10  R=  9691.0  vis=100.0%  cent=0.237  frac=0.2585  [OK]
  -- MEDIUM (off=0.6m vel=0.25 ang=0.1) --
    Ep 11  R=  8665.9  vis=100.0%  cent=0.306  frac=0.1961  [OK]
    Ep 12  R=  9124.2  vis= 98.5%  cent=0.221  frac=0.1268  [OK]
    Ep 13  R=  7459.4  vis=100.0%  cent=0.377  frac=0.1947  [OK]
    Ep 14  R=  7407.5  vis=100.0%  cent=0.393  frac=0.2188  [OK]
    Ep 15  R= 10086.4  vis=100.0%  cent=0.211  frac=0.2677  [OK]
    Ep 16  R=     0.3  vis= 11.3%  cent=1.066  frac=0.0196  [EARLY]
    Ep 17  R=   862.8  vis= 95.6%  cent=0.569  frac=0.0532  [EARLY]
    Ep 18  R=  7502.5  vis=100.0%  cent=0.407  frac=0.2329  [OK]
    Ep 19  R=  8270.6  vis=100.0%  cent=0.324  frac=0.1907  [OK]
    Ep 20  R=  5425.8  vis= 99.2%  cent=0.485  frac=0.1428  [OK]
  -- HARD (off=1.0m vel=0.35 ang=0.15) --
    Ep 21  R=   569.4  vis= 24.5%  cent=0.796  frac=0.0284  [EARLY]
    Ep 22  R=  3323.9  vis= 93.5%  cent=0.780  frac=0.1160  [OK]
    Ep 23  R=   387.2  vis= 22.5%  cent=1.211  frac=0.0175  [OK]
    Ep 24  R=  7182.4  vis= 97.0%  cent=0.372  frac=0.1755  [OK]
    Ep 25  R=  4880.6  vis= 67.5%  cent=0.403  frac=0.1644  [OK]
    Ep 26  R=  -146.9  vis=  0.4%  cent=0.933  frac=0.0059  [OK]
    Ep 27  R=   450.9  vis= 46.1%  cent=0.930  frac=0.0997  [EARLY]
    Ep 28  R=  8635.8  vis= 87.0%  cent=0.217  frac=0.1729  [OK]
    Ep 29  R=  -140.1  vis=  1.1%  cent=1.203  frac=0.0122  [OK]
    Ep 30  R=    60.9  vis= 44.9%  cent=1.148  frac=0.0147  [EARLY]

============================================================
  Evaluating: model_900000_steps.zip  (900,000 steps)
============================================================
  -- EASY (off=0.2m vel=0.1 ang=0.05) --
    Ep  1  R=  9384.6  vis=100.0%  cent=0.237  frac=0.1921  [OK]
    Ep  2  R=  6185.5  vis=100.0%  cent=0.405  frac=0.1038  [OK]
    Ep  3  R=  9495.0  vis=100.0%  cent=0.253  frac=0.2354  [OK]
    Ep  4  R=  9374.0  vis=100.0%  cent=0.244  frac=0.1959  [OK]
    Ep  5  R=  9611.3  vis=100.0%  cent=0.240  frac=0.2172  [OK]
    Ep  6  R=  8149.0  vis=100.0%  cent=0.320  frac=0.1705  [OK]
    Ep  7  R=  7084.9  vis=100.0%  cent=0.385  frac=0.1935  [OK]
    Ep  8  R=  9153.2  vis=100.0%  cent=0.260  frac=0.2252  [OK]
    Ep  9  R=  9226.5  vis=100.0%  cent=0.254  frac=0.1957  [OK]
    Ep 10  R= 10515.1  vis=100.0%  cent=0.156  frac=0.1926  [OK]
  -- MEDIUM (off=0.6m vel=0.25 ang=0.1) --
    Ep 11  R=  8738.3  vis= 99.9%  cent=0.246  frac=0.1181  [OK]
    Ep 12  R= 10308.5  vis= 98.5%  cent=0.176  frac=0.2481  [OK]
    Ep 13  R=  5836.4  vis=100.0%  cent=0.464  frac=0.1465  [OK]
    Ep 14  R=  3410.3  vis= 92.8%  cent=0.413  frac=0.0992  [EARLY]
    Ep 15  R=  7475.7  vis=100.0%  cent=0.385  frac=0.1936  [OK]
    Ep 16  R=  7008.8  vis= 96.8%  cent=0.411  frac=0.1945  [OK]
    Ep 17  R=   326.7  vis= 81.1%  cent=0.976  frac=0.0347  [EARLY]
    Ep 18  R=  8801.1  vis=100.0%  cent=0.262  frac=0.1335  [OK]
    Ep 19  R= 10573.5  vis=100.0%  cent=0.169  frac=0.1873  [OK]
    Ep 20  R=   205.5  vis= 69.1%  cent=0.881  frac=0.0755  [EARLY]
  -- HARD (off=1.0m vel=0.35 ang=0.15) --
    Ep 21  R=    10.6  vis= 17.8%  cent=1.047  frac=0.0156  [EARLY]
    Ep 22  R=   580.3  vis= 64.0%  cent=0.645  frac=0.1132  [EARLY]
    Ep 23  R=    84.8  vis= 28.8%  cent=1.123  frac=0.0190  [EARLY]
    Ep 24  R=  5356.2  vis= 86.4%  cent=0.370  frac=0.0369  [OK]
    Ep 25  R=  4269.1  vis= 91.2%  cent=0.490  frac=0.0746  [EARLY]
    Ep 26  R=  -102.3  vis=  2.5%  cent=1.057  frac=0.0062  [EARLY]
    Ep 27  R=  -154.1  vis=  1.7%  cent=1.082  frac=0.0070  [EARLY]
    Ep 28  R=    20.7  vis= 20.5%  cent=0.984  frac=0.0475  [EARLY]
    Ep 29  R=  -121.9  vis=  0.5%  cent=1.102  frac=0.0117  [EARLY]
    Ep 30  R=  3975.5  vis= 96.5%  cent=0.553  frac=0.0667  [OK]

============================================================
  Evaluating: model_1500000_steps.zip  (1,500,000 steps)
============================================================
  -- EASY (off=0.2m vel=0.1 ang=0.05) --
    Ep  1  R=  8217.5  vis=100.0%  cent=0.305  frac=0.1919  [OK]
    Ep  2  R=  8841.9  vis=100.0%  cent=0.279  frac=0.2030  [OK]
    Ep  3  R=  9013.3  vis=100.0%  cent=0.269  frac=0.2253  [OK]
    Ep  4  R=  9209.7  vis=100.0%  cent=0.253  frac=0.1945  [OK]
    Ep  5  R=  1687.3  vis= 71.7%  cent=0.598  frac=0.1133  [EARLY]
    Ep  6  R=  7517.7  vis=100.0%  cent=0.284  frac=0.0568  [OK]
    Ep  7  R=  8839.9  vis=100.0%  cent=0.226  frac=0.1274  [OK]
    Ep  8  R=  9869.8  vis=100.0%  cent=0.221  frac=0.2451  [OK]
    Ep  9  R=  9358.5  vis=100.0%  cent=0.235  frac=0.1916  [OK]
    Ep 10  R=  7583.5  vis=100.0%  cent=0.328  frac=0.1449  [OK]
  -- MEDIUM (off=0.6m vel=0.25 ang=0.1) --
    Ep 11  R=  6129.7  vis=100.0%  cent=0.416  frac=0.0706  [OK]
    Ep 12  R=  5800.0  vis= 99.8%  cent=0.522  frac=0.1628  [OK]
    Ep 13  R=  3846.1  vis=100.0%  cent=0.682  frac=0.1076  [OK]
    Ep 14  R=   768.4  vis= 67.9%  cent=0.893  frac=0.0692  [EARLY]
    Ep 15  R=  2085.2  vis= 77.5%  cent=0.949  frac=0.0511  [OK]
    Ep 16  R=  8268.4  vis= 98.8%  cent=0.340  frac=0.2345  [OK]
    Ep 17  R=  7638.4  vis=100.0%  cent=0.339  frac=0.1306  [OK]
    Ep 18  R=  6003.1  vis=100.0%  cent=0.436  frac=0.0906  [OK]
    Ep 19  R=  4782.5  vis=100.0%  cent=0.559  frac=0.0831  [OK]
    Ep 20  R=   790.4  vis= 38.2%  cent=1.060  frac=0.0252  [EARLY]
  -- HARD (off=1.0m vel=0.35 ang=0.15) --
    Ep 21  R=  4071.0  vis= 95.5%  cent=0.566  frac=0.0446  [OK]
    Ep 22  R=   156.7  vis= 18.9%  cent=0.962  frac=0.0462  [EARLY]
"""

# Parsing logic
current_ckpt = "800k" # Assuming the first one starts at 800k since 850k appears later
current_tier = "easy"
data_dict = collections.defaultdict(lambda: collections.defaultdict(list))

for line in data.split('\n'):
    line = line.strip()
    if 'Evaluating: model_' in line:
        ckpt_str = re.search(r'model_(\d+)_steps', line).group(1)
        current_ckpt = f"{int(ckpt_str)//1000}k"
    elif '-- EASY' in line:
        current_tier = 'easy'
    elif '-- MEDIUM' in line:
        current_tier = 'medium'
    elif '-- HARD' in line:
        current_tier = 'hard'
    elif line.startswith('Ep '):
        match = re.search(r'Ep\s+\d+\s+R=\s*([\-\d\.]+)\s+vis=\s*([\d\.]+)%\s+cent=\s*([\d\.]+)\s+frac=\s*([\d\.]+)\s+\[(OK|EARLY)\]', line)
        if match:
            r = float(match.group(1))
            vis = float(match.group(2))
            cent = float(match.group(3))
            frac = float(match.group(4))
            status = match.group(5)
            data_dict[current_ckpt][current_tier].append({
                'r': r,
                'vis': vis,
                'cent': cent,
                'frac': frac,
                'surv': 1.0 if status == 'OK' else 0.0
            })
            data_dict[current_ckpt]['global'].append({
                'r': r,
                'vis': vis,
                'cent': cent,
                'frac': frac,
                'surv': 1.0 if status == 'OK' else 0.0
            })

steps = ['800k', '850k', '900k', '1500k']
out_dir = Path('experiments/hover_track_v3_fast')
out_dir.mkdir(parents=True, exist_ok=True)

# Generate tier plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
tier_colors = {'easy': '#4CAF50', 'medium': '#2196F3', 'hard': '#FF5722'}

for tier in ('easy', 'medium', 'hard'):
    surv = []
    rew = []
    for s in steps:
        if len(data_dict[s][tier]) > 0:
            surv.append(np.mean([e['surv'] for e in data_dict[s][tier]]) * 100)
            rew.append(np.mean([e['r'] for e in data_dict[s][tier]]))
        else:
            surv.append(np.nan)
            rew.append(np.nan)
            
    axes[0].plot(range(len(steps)), surv, marker='o', label=tier.capitalize(), color=tier_colors[tier], linewidth=2)
    axes[1].plot(range(len(steps)), rew, marker='o', label=tier.capitalize(), color=tier_colors[tier], linewidth=2)

for ax, title in zip(axes, ['Survival Rate (%)', 'Mean Reward']):
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
fig.suptitle('Checkpoint Comparison (Terminal Data) — Per Tier', fontsize=14)
fig.tight_layout()
fig.savefig(str(out_dir / 'terminal_checkpoint_tiers.png'), dpi=150)
plt.close(fig)

# Generate global plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
metrics = [
    ('r', 'Total Reward'),
    ('vis', 'Visibility (%)'),
    ('cent', 'Centering Distance'),
    ('frac', 'Target Fraction'),
    ('surv', 'Survival Rate (%)'),
]

for ax, (key, title) in zip(axes.flat[:5], metrics):
    means = []
    stds = []
    for s in steps:
        if len(data_dict[s]['global']) > 0:
            vals = [e[key]*100 if key == 'surv' else e[key] for e in data_dict[s]['global']]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            
    ax.errorbar(range(len(steps)), means, yerr=stds, marker='o', capsize=4, linewidth=2)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_title(title)
    ax.grid(alpha=0.3)

axes.flat[-1].set_visible(False)
fig.suptitle('Checkpoint Comparison (Terminal Data) — Global Metrics', fontsize=14)
fig.tight_layout()
fig.savefig(str(out_dir / 'terminal_checkpoint_global.png'), dpi=150)
plt.close(fig)
