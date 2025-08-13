# Bayesian-of-cachexia
# 本コードは論文「○○○（2025）」の解析手順を模擬データで再現したものです。
# 個人情報保護および倫理的配慮のため、実データは含まれていません。
# モデル構造・事前分布・解析手順は論文本文と同一です。
#
# 実行環境：
#   Google Colabを推奨します。
#   Colabではライブラリインストールと日本語フォント設定が自動的に適用されます。
#   ローカル環境で実行する場合は、必要なライブラリのインストールと
#   日本語フォント設定を適宜調整してください。

#注意：Google Colab用の解析コードです。その他の総合開発環境では作動しない可能性があります。
#注意: 実際のデータを使用する場合は、以下を df = pd.read_csv("使用者のデータ.csv") に置き換える必要があります。


!pip install pandas numpy scipy matplotlib seaborn pymc arviz japanize-matplotlib

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import japanize_matplotlib # Import for Japanese font support

# Configure matplotlib to use a font that supports Japanese characters
plt.rcParams['font.family'] = 'IPAexGothic'


# 再現性のための乱数シード設定
np.random.seed(42)

# === 1. データ読み込み===
#注意: 実際のデータを使用する場合は、以下を df = pd.read_csv("使用者のデータ.csv") に置き換える必要があります。

n = 49
data = {
    'Age': np.random.normal(84.5, 7.8, n),
    'Sex': ['M']*12 + ['F']*37,  # 男性12名、女性37名
    'BMI': np.random.normal(19.9, 4.7, n),
    'MMSE': np.random.normal(24.8, 4.8, n),
    'GripStrength': np.concatenate([np.random.normal(25.1, 6.3, 12), np.random.normal(15.5, 5.0, 37)]),
    'CRP': np.random.normal(1.0, 0.5, n),
    'FTSST': np.random.normal(15.5, 7.4, n),
    'Residence': [0]*36 + [1]*13,  # 0: 自宅, 1: 施設
    'RehabFreq': [1]*19 + [2]*26 + [3]*4  # 1: 週1回, 2: 週2回, 3: 週3回以上
}
df = pd.DataFrame(data)

# 前処理: カテゴリ変数を数値化

df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
df['RehabFreq_binary'] = np.where(df['RehabFreq'] == 1, 0, 1)  # 0: 週1回, 1: 週2回以上

# 論文の相関を反映したBI_gainの生成
df['BI_gain'] = (10.9
                 - 0.25*(df['Age']-84.5)/7.8  # τ=-0.25
                 - 0.21*(df['BMI']-19.9)/4.7  # τ=-0.21
                 - 0.22*(df['FTSST']-15.5)/7.4  # τ=-0.22
                 + 0.18*(df['MMSE']-24.8)/4.8  # τ=0.18
                 + 0.15*(df['GripStrength']-df['GripStrength'].mean())/df['GripStrength'].std()  # τ=0.15
                 - 0.19*(df['CRP']-1.0)/0.5  # τ=-0.19
                 + 0.62*5*df['RehabFreq_binary']  # d=0.62, スケール調整
                 + np.random.normal(0, 2, n))

# 欠損値チェック

print("欠損値チェック:\n", df.isna().sum())

# === 2. 基礎統計 ===

cont_vars = ['Age', 'BMI', 'MMSE', 'GripStrength', 'CRP', 'FTSST']
cat_vars = ['Sex', 'Residence', 'RehabFreq_binary']

print("\n=== 基礎統計 ===")
for var in cont_vars:
    stat, p = stats.shapiro(df[var].dropna())
    print(f"{var} Shapiro-Wilk p={p:.4f} {'(正規)' if p>0.05 else '(非正規)'}")

    # ヒストグラムとQ-Qプロットの可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[var], kde=True, ax=ax1)
    ax1.set_title(f"{var} ヒストグラム")
    stats.probplot(df[var].dropna(), dist="norm", plot=ax2)
    ax2.set_title(f"{var} Q-Qプロット")
    plt.tight_layout()
    plt.savefig(f"{var}_plots.pdf")  # 論文用に保存
    plt.close()

    if p > 0.05:
        print(f"{var}: {df[var].mean():.2f} ± {df[var].std():.2f}")
    else:
        print(f"{var}: {df[var].median():.2f} ({df[var].quantile(0.25):.2f}–{df[var].quantile(0.75):.2f})")

for var in cat_vars:
    counts = df[var].value_counts()
    percents = df[var].value_counts(normalize=True) * 100
    print(f"\n{var}:")
    for val, count in counts.items():
        print(f"  {val}: {count} ({percents[val]:.1f}%)")


# === 3. 事前分布設定のための探索的データ解析 ===
print("\n=== 探索的データ解析 ===")
priors = {
    'Age': {'mean': -0.25, 'sd': (0.40-0.10)/(2*1.96)},  # 論文より
    'BMI': {'mean': -0.21, 'sd': (0.35-0.08)/(2*1.96)},
    'MMSE': {'mean': 0.18, 'sd': (0.31-0.05)/(2*1.96)},
    'FTSST': {'mean': -0.22, 'sd': (0.36-0.09)/(2*1.96)},
    'GripStrength': {'mean': 0.15, 'sd': (0.28-0.02)/(2*1.96)},
    'CRP': {'mean': -0.19, 'sd': (0.33-0.06)/(2*1.96)},
    'Sex': {'mean': 0, 'sd': 0.5},  # CIが0を含む
    'Residence': {'mean': 0, 'sd': 0.5},  # CIが0を含む
    'RehabFreq_binary': {'mean': 0.62, 'sd': (0.96-0.25)/(2*1.96)}  # d=0.62
}

# KendallのτとブートストラップCI (Manual Bootstrap)
n_resamples = 1000
alpha = 0.05

for var in cont_vars:
    tau, p = stats.kendalltau(df[var], df['BI_gain'])

    # Manual bootstrap
    bootstrap_taus = []
    data_arrays = np.array([df[var].values, df['BI_gain'].values]).T # Transpose to have rows as samples
    n_samples = len(data_arrays)

    for _ in range(n_resamples):
        # Resample with replacement
        resample_indices = np.random.choice(n_samples, n_samples, replace=True)
        resampled_data = data_arrays[resample_indices]
        # Calculate tau on resampled data
        boot_tau, _ = stats.kendalltau(resampled_data[:, 0], resampled_data[:, 1])
        bootstrap_taus.append(boot_tau)

    # Calculate percentile confidence interval
    ci_low = np.percentile(bootstrap_taus, alpha/2 * 100)
    ci_high = np.percentile(bootstrap_taus, (1 - alpha/2) * 100)

    print(f"{var} Kendall τ={tau:.3f}, p={p:.4f}, 95%CI: [{ci_low:.3f}, {ci_high:.3f}]")


# カテゴリ変数のCohen's d (using scipy.stats.bootstrap as it worked)
from scipy.stats import bootstrap # Import bootstrap here

def cohens_d(g1, g2):
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1), np.std(g2)
    pooled_sd = np.sqrt(((len(g1)-1)*s1**2 + (len(g2)-1)*s2**2) / (len(g1) + len(g2) - 2))
    return (m1 - m2) / pooled_sd

for var in cat_vars:
    groups = df.groupby(var)['BI_gain']
    group_vals = list(groups.groups.keys())
    if len(group_vals) == 2:
        g1, g2 = groups.get_group(group_vals[0]), groups.get_group(group_vals[1])
        d = cohens_d(g1, g2)
        def d_func(idx):
            sample = df['BI_gain'].iloc[idx]
            sample_g1 = sample[df[var] == group_vals[0]]
            sample_g2 = sample[df[var] == group_vals[1]]
            # Handle potential empty groups in resamples
            if len(sample_g1) < 2 or len(sample_g2) < 2:
                 return np.nan  # Or handle as appropriate, e.g., return 0 or skip
            return cohens_d(sample_g1, sample_g2)

        # Filter out NaN results from the bootstrap if any occurred due to empty resampled groups
        res = bootstrap((np.arange(len(df)),), d_func, vectorized=False, n_resamples=1000)
        ci_low, ci_high = res.confidence_interval
        print(f"{var} Cohen's d={d:.2f}, 95%CI: [{ci_low:.2f}, {ci_high:.2f}]")


# 事前分布の出力
print("\n=== 事前分布 ===")
print(f"BI_gain: N(5.8, {5.33**2:.2f})")
for var, p in priors.items():
    print(f"{var}: N({p['mean']:.3f}, {p['sd']**2:.3f})")



# === 4. Bayesian階層モデリング ===
print("\n=== Bayesian階層モデリング ===")

# PyMCモデルの定義
coords = {
    "coeffs": ['Age', 'BMI', 'MMSE', 'GripStrength', 'CRP', 'FTSST', 'Sex', 'Residence', 'RehabFreq_binary']
}
with pm.Model(coords=coords) as hierarchical_model:
    # 事前分布 (探索的データ解析の結果に基づく)
    # 連続変数の回帰係数 (tau)
    age_beta = pm.Normal("age_beta", mu=priors['Age']['mean'], sigma=priors['Age']['sd'])
    bmi_beta = pm.Normal("bmi_beta", mu=priors['BMI']['mean'], sigma=priors['BMI']['sd'])
    mmse_beta = pm.Normal("mmse_beta", mu=priors['MMSE']['mean'], sigma=priors['MMSE']['sd'])
    grip_beta = pm.Normal("grip_beta", mu=priors['GripStrength']['mean'], sigma=priors['GripStrength']['sd'])
    crp_beta = pm.Normal("crp_beta", mu=priors['CRP']['mean'], sigma=priors['CRP']['sd'])
    ftsst_beta = pm.Normal("ftsst_beta", mu=priors['FTSST']['mean'], sigma=priors['FTSST']['sd'])

    # カテゴリ変数の差の平均 (Cohen's d に近い尺度)
    # RehabFreq_binary のみ事前分布を設定 (d=0.62)
    rehab_beta = pm.Normal("rehab_beta", mu=priors['RehabFreq_binary']['mean'], sigma=priors['RehabFreq_binary']['sd'])

    # SexとResidenceはCIに0を含むため、より広い事前分布
    sex_beta = pm.Normal("sex_beta", mu=priors['Sex']['mean'], sigma=priors['Sex']['sd'])
    residence_beta = pm.Normal("residence_beta", mu=priors['Residence']['mean'], sigma=priors['Residence']['sd'])


    # 切片
    intercept = pm.Normal("intercept", mu=df['BI_gain'].mean(), sigma=df['BI_gain'].std())

    # 標準偏差
    sigma = pm.HalfNormal("sigma", sigma=df['BI_gain'].std())

    # 線形モデル
    # 各説明変数を標準化 (回帰係数の解釈を容易にするため)
    age_scaled = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    bmi_scaled = (df['BMI'] - df['BMI'].mean()) / df['BMI'].std()
    mmse_scaled = (df['MMSE'] - df['MMSE'].mean()) / df['MMSE'].std()
    grip_scaled = (df['GripStrength'] - df['GripStrength'].mean()) / df['GripStrength'].std()
    crp_scaled = (df['CRP'] - df['CRP'].mean()) / df['CRP'].std()
    ftsst_scaled = (df['FTSST'] - df['FTSST'].mean()) / df['FTSST'].std()


    mu = intercept + age_beta * age_scaled + bmi_beta * bmi_scaled + mmse_beta * mmse_scaled + \
         grip_beta * grip_scaled + crp_beta * crp_scaled + ftsst_beta * ftsst_scaled + \
         sex_beta * df['Sex'] + residence_beta * df['Residence'] + rehab_beta * df['RehabFreq_binary']


    # 尤度
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=df['BI_gain'])

    # モデルのサンプリング
    print("モデルのサンプリングを開始...")
    idata = pm.sample(2000, tune=1000, cores=1, target_accept=0.95, return_inferencedata=True)
    print("サンプリング完了。")

"""Now that the model is sampled, we can analyze the results, visualize the posterior distributions, and check for convergence."""

# === 5. 結果の要約と可視化 (事後分布) ===
print("\n=== 結果の要約 ===")
print(az.summary(idata, fmt="wide"))

print("\n=== 事後分布の可視化 ===")
az.plot_trace(idata)
plt.tight_layout()
plt.savefig("trace_plots.pdf")
plt.show()

az.plot_forest(idata, var_names=["age_beta", "bmi_beta", "mmse_beta", "grip_beta", "crp_beta", "ftsst_beta", "sex_beta", "residence_beta", "rehab_beta", "intercept"], combined=True)
plt.tight_layout()
plt.savefig("forest_plot.pdf")
plt.show()

# 収束診断
rhat = az.rhat(idata)
print("\n=== Rhat (収束診断) ===")
print(rhat)

ess = az.ess(idata)
print("\n=== ESS (実効サンプルサイズ) ===")
print(ess)

print("\n=== 事後平均と論文の値の比較 ===")
# Use the original idata for summary of parameter posteriors
posterior_means = az.summary(idata, var_names=["age_beta", "bmi_beta", "mmse_beta", "grip_beta", "crp_beta", "ftsst_beta", "sex_beta", "residence_beta", "rehab_beta", "intercept"])["mean"]
print("事後平均:\n", posterior_means)
print("\n論文で報告されている相関 (τ) または効果量 (d):\n", priors) # Prior means are based on reported effects

# 可視化結果の保存 (事後分布)
print("\n事後分布のグラフをtrace_plots.pdf, forest_plot.pdfとして保存しました。")
