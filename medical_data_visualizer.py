import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import 
df = pd.read_csv('medical_examination.csv')

# 2 Add 'overweight' 
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3 0 always good - 1 always bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4 Draw Cat
def draw_cat_plot():
    # 5 DataFrame for cat plot
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 Rename 'value' to 'count'
    df_cat = df_cat.rename(columns={'value': 'count'})

    # 7 Draw catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', hue='count', col='cardio', data=df_cat, kind='count').fig

    # 8 Save plot
    fig.savefig('catplot.png')
    return fig

# 10 Heat Map
def draw_heat_map():
    # 11 Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 correlation matrix
    corr = df_heat.corr()

    # 13 mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # 15 Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True, cbar_kws={'shrink': 0.5}, ax=ax)

    # 16 Save heatmap
    fig.savefig('heatmap.png')
    return fig
