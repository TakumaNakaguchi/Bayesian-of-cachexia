# 本コードは論文「○○○（2025）」の解析手順を模擬データで再現したものです。
# 個人情報保護および倫理的配慮のため、実データは含まれていません。
# モデル構造・事前分布・解析手順は論文本文と同一です。
#
# 実行環境：
#   Google Colabを推奨します。
#   Colabではライブラリインストールと日本語フォント設定が自動的に適用されます。
#   ローカル環境で実行する場合は、必要なライブラリのインストールと
#   日本語フォント設定を適宜調整してください。

!pip install pandas numpy scipy matplotlib seaborn pymc arviz japanize-matplotlib

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import japanize_matplotlib  # Import for Japanese font support

# 日本語フォントの設定（フォールバック処理を追加）
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
except:
    plt.rcParams['font.family'] = 'sans-serif'
    print("警告: IPAexGothicフォントが利用できません。デフォルトフォントを使用します。")

# 再現性のための乱数シード設定
np.random.seed(42)

# === 1. データ読み込み ===
# 注意: 実際のデータを使用する場合は、以下を df = pd.read_csv("使用者のデータ.csv") に置き換える必要があります。

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
    'RehabFreq': [1]*19 + [2]*26 + [3]*4  # 1: 週1回, 2: 週2回以上, 3: 週3回
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
# 論文セクション「データ解析：基礎統計の算出」に基づく
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
    try:
        plt.savefig(f"{var}_plots.pdf")  # 論文用に保存
    except Exception as e:
        print(f"ファイル保存エラー ({var}_plots.pdf): {e}")
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
# 論文セクション「事前分布の設定」に基づく
print("\n=== 探索的データ解析 ===")
priors = {
    'Age': {'mean': -0.25, 'sd': (0.40-0.10)/(2*1.96)},
    'BMI': {'mean': -0.21, 'sd': (0.35-0.08)/(2*1.96)},
    'MMSE': {'mean': 0.18, 'sd': (0.31-0.05)/(2*1.96)},
    'FTSST': {'mean': -0.22, 'sd': (0.36-0.09)/(2*1.96)},
    'GripStrength': {'mean': 0.15, 'sd': (0.28-0.02)/(2*1.96)},
    'CRP': {'mean': -0.19, 'sd': (0.33-0.06)/(2*1.96)},
    'Sex': {'mean': 0, 'sd': 0.5},
    'Residence': {'mean': 0, 'sd': 0.5},
    'RehabFreq_binary': {'mean': 0.62, 'sd': (0.96-0.25)/(2*1.96)}  # d=0.62
}

# KendallのτとブートストラップCI
n_resamples = 1000
alpha = 0.05

for var in cont_vars:
    tau, p = stats.kendalltau(df[var], df['BI_gain'])
    bootstrap_taus = []
    data_arrays = np.array([df[var].values, df['BI_gain'].values]).T
    n_samples = len(data_arrays)
    for _ in range(n_resamples):
        resample_indices = np.random.choice(n_samples, n_samples, replace=True)
        resampled_data = data_arrays[resample_indices]
        boot_tau, _ = stats.kendalltau(resampled_data[:, 0], resampled_data[:, 1])
        if not np.isnan(boot_tau):  # NaNを除外
            bootstrap_taus.append(boot_tau)
    ci_low = np.percentile(bootstrap_taus, alpha/2 * 100)
    ci_high = np.percentile(bootstrap_taus, (1 - alpha/2) * 100)
    print(f"{var} Kendall τ={tau:.3f}, p={p:.4f}, 95%CI: [{ci_low:.3f}, {ci_high:.3f}]")

# カテゴリ変数のCohen's d
from scipy.stats import bootstrap

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
        def d_func(data, axis):
            sample_g1 = data[df[var] == group_vals[0]]
            sample_g2 = data[df[var] == group_vals[1]]
            if len(sample_g1) < 2 or len(sample_g2) < 2:
                return np.nan
            return cohens_d(sample_g1, sample_g2)
        res = bootstrap((df['BI_gain'].values,), d_func, vectorized=False, n_resamples=1000)
        ci_low, ci_high = res.confidence_interval
        print(f"{var} Cohen's d={d:.2f}, 95%CI: [{ci_low:.2f}, {ci_high:.2f}]")

# 事前分布の出力
print("\n=== 事前分布 ===")
print(f"BI_gain (SEM): N(5.8, {5.33**2:.2f})")
print(f"BI_gain (MDC): N(5.8, {16.3**2:.2f})")
for var, p in priors.items():
    print(f"{var}: N({p['mean']:.3f}, {p['sd']**2:.3f})")

# === 4. Bayesian階層モデリング (主要解析) ===
print("\n=== Bayesian階層モデリング (主要解析) ===")

# カテゴリ変数をPandasのCategorical型に変換
df['Residence_cat'] = pd.Categorical(df['Residence'])
df['RehabFreq_cat'] = pd.Categorical(df['RehabFreq_binary'])

# PyMCモデルの定義 (階層構造)
coords = {
    "coeffs": ['Age', 'BMI', 'MMSE', 'GripStrength', 'CRP', 'FTSST', 'Sex'],
    "residence_group": df['Residence_cat'].unique(),
    "rehabfreq_group": df['RehabFreq_cat'].unique()
}
with pm.Model(coords=coords) as hierarchical_model_main:
    intercept_residence = pm.Normal("intercept_residence", mu=df['BI_gain'].mean(), sigma=df['BI_gain'].std(), dims="residence_group")
    intercept_rehab = pm.Normal("intercept_rehab", mu=df['BI_gain'].mean(), sigma=df['BI_gain'].std(), dims="rehabfreq_group")
    age_beta = pm.Normal("age_beta", mu=priors['Age']['mean'], sigma=priors['Age']['sd'], dims="residence_group")
    bmi_beta = pm.Normal("bmi_beta", mu=priors['BMI']['mean'], sigma=priors['BMI']['sd'], dims="residence_group")
    mmse_beta = pm.Normal("mmse_beta", mu=priors['MMSE']['mean'], sigma=priors['MMSE']['sd'], dims="residence_group")
    grip_beta = pm.Normal("grip_beta", mu=priors['GripStrength']['mean'], sigma=priors['GripStrength']['sd'], dims="residence_group")
    crp_beta = pm.Normal("crp_beta", mu=priors['CRP']['mean'], sigma=priors['CRP']['sd'], dims="residence_group")
    ftsst_beta = pm.Normal("ftsst_beta", mu=priors['FTSST']['mean'], sigma=priors['FTSST']['sd'], dims="residence_group")
    sex_beta = pm.Normal("sex_beta", mu=priors['Sex']['mean'], sigma=priors['Sex']['sd'], dims="residence_group")
    sigma = pm.HalfNormal("sigma", sigma=df['BI_gain'].std())

    # 線形モデル
    age_scaled = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    bmi_scaled = (df['BMI'] - df['BMI'].mean()) / df['BMI'].std()
    mmse_scaled = (df['MMSE'] - df['MMSE'].mean()) / df['MMSE'].std()
    grip_scaled = (df['GripStrength'] - df['GripStrength'].mean()) / df['GripStrength'].std()
    crp_scaled = (df['CRP'] - df['CRP'].mean()) / df['CRP'].std()
    ftsst_scaled = (df['FTSST'] - df['FTSST'].mean()) / df['FTSST'].std()

    # 階層構造
    mu = (intercept_residence[df['Residence_cat'].cat.codes] +
          intercept_rehab[df['RehabFreq_cat'].cat.codes] +
          age_beta[df['Residence_cat'].cat.codes] * age_scaled +
          bmi_beta[df['Residence_cat'].cat.codes] * bmi_scaled +
          mmse_beta[df['Residence_cat'].cat.codes] * mmse_scaled +
          grip_beta[df['Residence_cat'].cat.codes] * grip_scaled +
          crp_beta[df['Residence_cat'].cat.codes] * crp_scaled +
          ftsst_beta[df['Residence_cat'].cat.codes] * ftsst_scaled +
          sex_beta[df['Residence_cat'].cat.codes] * df['Sex'])

    # 尤度
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=df['BI_gain'])

    # モデルのサンプリング（複数コアを使用）
    print("モデルのサンプリングを開始...")
    idata_main = pm.sample(2000, tune=500, cores=4, target_accept=0.95, return_inferencedata=True)
    print("サンプリング完了。")

# === 5. 感度解析 ===
print("\n=== Bayesian階層モデリング (感度解析) ===")
bi_gain_mdc_sigma = 16.3
with pm.Model(coords=coords) as hierarchical_model_mdc:
    intercept_residence = pm.Normal("intercept_residence", mu=df['BI_gain'].mean(), sigma=bi_gain_mdc_sigma, dims="residence_group")
    intercept_rehab = pm.Normal("intercept_rehab", mu=df['BI_gain'].mean(), sigma=bi_gain_mdc_sigma, dims="rehabfreq_group")
    age_beta = pm.Normal("age_beta", mu=priors['Age']['mean'], sigma=priors['Age']['sd'], dims="residence_group")
    bmi_beta = pm.Normal("bmi_beta", mu=priors['BMI']['mean'], sigma=priors['BMI']['sd'], dims="residence_group")
    mmse_beta = pm.Normal("mmse_beta", mu=priors['MMSE']['mean'], sigma=priors['MMSE']['sd'], dims="residence_group")
    grip_beta = pm.Normal("grip_beta", mu=priors['GripStrength']['mean'], sigma=priors['GripStrength']['sd'], dims="residence_group")
    crp_beta = pm.Normal("crp_beta", mu=priors['CRP']['mean'], sigma=priors['CRP']['sd'], dims="residence_group")
    ftsst_beta = pm.Normal("ftsst_beta", mu=priors['FTSST']['mean'], sigma=priors['FTSST']['sd'], dims="residence_group")
    sex_beta = pm.Normal("sex_beta", mu=priors['Sex']['mean'], sigma=priors['Sex']['sd'], dims="residence_group")
    sigma = pm.HalfNormal("sigma", sigma=df['BI_gain'].std())

    mu = (intercept_residence[df['Residence_cat'].cat.codes] +
          intercept_rehab[df['RehabFreq_cat'].cat.codes] +
          age_beta[df['Residence_cat'].cat.codes] * age_scaled +
          bmi_beta[df['Residence_cat'].cat.codes] * bmi_scaled +
          mmse_beta[df['Residence_cat'].cat.codes] * mmse_scaled +
          grip_beta[df['Residence_cat'].cat.codes] * grip_scaled +
          crp_beta[df['Residence_cat'].cat.codes] * crp_scaled +
          ftsst_beta[df['Residence_cat'].cat.codes] * ftsst_scaled +
          sex_beta[df['Residence_cat'].cat.codes] * df['Sex'])

    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=df['BI_gain'])

    print("感度解析のサンプリングを開始...")
    idata_mdc = pm.sample(2000, tune=500, cores=4, target_accept=0.95, return_inferencedata=True)
    print("サンプリング完了。")

# === 6. 結果の要約と可視化 (事後分布) ===
print("\n=== 主要解析の結果の要約 ===")
summary_main = az.summary(idata_main, fmt="wide")
print(summary_main)

print("\n=== 感度解析の結果の要約 ===")
print(az.summary(idata_mdc, fmt="wide"))

print("\n=== 主要解析の事後分布の可視化 ===")
az.plot_trace(idata_main, var_names=['age_beta', 'bmi_beta', 'mmse_beta', 'grip_beta', 'crp_beta', 'ftsst_beta', 'sex_beta', 'intercept_residence', 'intercept_rehab', 'sigma'])
plt.tight_layout()
try:
    plt.savefig("trace_plots_main.pdf")
except Exception as e:
    print(f"ファイル保存エラー (trace_plots_main.pdf): {e}")
plt.show()

az.plot_forest(idata_main, var_names=['age_beta', 'bmi_beta', 'mmse_beta', 'grip_beta', 'crp_beta', 'ftsst_beta', 'sex_beta', 'intercept_residence', 'intercept_rehab'], combined=True)
plt.title("主要解析の事後分布")
plt.tight_layout()
try:
    plt.savefig("forest_plot_main.pdf")
except Exception as e:
    print(f"ファイル保存エラー (forest_plot_main.pdf): {e}")
plt.show()

# 主要解析と感度解析の比較可視化
print("\n=== 主要解析と感度解析の比較 ===")
az.plot_forest([idata_main, idata_mdc], var_names=['age_beta', 'bmi_beta', 'mmse_beta', 'grip_beta', 'crp_beta', 'ftsst_beta', 'sex_beta'], combined=True)
plt.title("主要解析と感度解析の比較")
plt.tight_layout()
try:
    plt.savefig("forest_plot_comparison.pdf")
except Exception as e:
    print(f"ファイル保存エラー (forest_plot_comparison.pdf): {e}")
plt.show()

# 収束診断
rhat_main = az.rhat(idata_main)
print("\n=== Rhat (主要解析) ===")
# Rhat>1.1の変数のみ警告表示
rhat_issues = {var: val for var, val in rhat_main.to_dataframe().items() if any(v > 1.1 for v in val.values.flatten())}
if rhat_issues:
    print("警告: 以下の変数のRhatが1.1を超えています（収束に問題の可能性）:")
    print(rhat_issues)
else:
    print("全ての変数のRhatは1.1未満（収束良好）。")

ess_main = az.ess(idata_main)
print("\n=== ESS (主要解析) ===")
# ESSの要約（最小値と平均値）
print(f"ESS最小値: {ess_main.to_dataframe().min().min():.0f}")
print(f"ESS平均値: {ess_main.to_dataframe().mean().mean():.0f}")

# 効果判定基準
print("\n=== 効果判定 (論文セクション「効果判定基準」に基づく) ===")
effect_results = []
for var in ['age_beta', 'bmi_beta', 'mmse_beta', 'grip_beta', 'crp_beta', 'ftsst_beta', 'sex_beta']:
    posterior = idata_main.posterior[var].values.flatten()
    prob_positive = np.mean(posterior > 0)
    prob_negative = np.mean(posterior < 0)
    hdi = az.hdi(idata_main, var_names=[var])[var].values
    effect = ("改善効果あり" if hdi[0] > 0 else
              "悪化効果あり" if hdi[1] < 0 else
              "改善効果あり (90%基準)" if prob_positive >= 0.9 else
              "悪化効果あり (90%基準)" if prob_negative >= 0.9 else
              "効果不明")
    effect_results.append({
        'Variable': var,
        'P(β>0)': prob_positive,
        'P(β<0)': prob_negative,
        '95% CrI': f"[{hdi[0]:.2f}, {hdi[1]:.2f}]",
        'Effect': effect
    })
effect_summary = pd.DataFrame(effect_results)
try:
    effect_summary.to_csv("effect_summary.csv")
except Exception as e:
    print(f"ファイル保存エラー (effect_summary.csv): {e}")
print(effect_summary)

# 効果判定の可視化
print("\n=== 効果判定の可視化 ===")
plt.figure(figsize=(10, 6))
sns.barplot(data=effect_summary, x='P(β>0)', y='Variable', hue='Effect')
plt.title("効果判定：P(β>0)の分布")
plt.tight_layout()
try:
    plt.savefig("effect_summary_plot.pdf")
except Exception as e:
    print(f"ファイル保存エラー (effect_summary_plot.pdf): {e}")
plt.show()

# 主要解析と感度解析の比較
print("\n=== 主要解析と感度解析の事後平均の比較 (論文セクション「感度解析」に基づく) ===")
posterior_means_main = az.summary(idata_main, var_names=['age_beta', 'bmi_beta', 'mmse_beta', 'grip_beta', 'crp_beta', 'ftsst_beta', 'sex_beta'])["mean"]
posterior_means_mdc = az.summary(idata_mdc, var_names=['age_beta', 'bmi_beta', 'mmse_beta', 'grip_beta', 'crp_beta', 'ftsst_beta', 'sex_beta'])["mean"]
diff_means = posterior_means_main - posterior_means_mdc
print("主要解析の事後平均:\n", posterior_means_main)
print("\n感度解析の事後平均:\n", posterior_means_mdc)
print("\n事後平均の差 (主要解析 - 感度解析):\n", diff_means)
print("主要解析と感度解析の結果の方向性と大きさを比較し、頑健性を評価する。")

# 可視化結果の保存
print("\n事後分布のグラフをtrace_plots_main.pdf, forest_plot_main.pdf, forest_plot_comparison.pdf, effect_summary_plot.pdfとして保存しました。")
