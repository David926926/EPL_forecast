"""Assemble every result table into one markdown file: report/all_results.md.
All numbers are read from the files the pipeline writes; nothing is typed by hand.
Run after src.robustness: python -m src.make_output
"""
import numpy as np
import pandas as pd

from src.clustering import SEGMENTS, TABLES
from src.config import FEATURES_PATH, MARKET_PATH, MATCHES_PATH, PROCESSED, ROOT, SEASONS
from src.features import F3_COLS, GAP_COLS

OUT = ROOT / "report" / "all_results.md"
SEG = {"even": "势均力敌", "middle": "中间", "lopsided": "悬殊"}


def md(df, digits=4, index=False):
    return df.round(digits).to_markdown(index=index) + "\n"


def main():
    L = []
    w = L.append
    w("# 全部结果表\n")
    w("> 由 `python -m src.make_output` 自动生成，所有数字都直接读自流程输出的文件（`data/processed/`、`report/tables/`）。"
      "解读见 [report_en.md](report_en.md)。\n")
    w("> 记号：类型 = 势均力敌（even）/ 中间（middle）/ 悬殊（lopsided）。合并测试期 = 4 个滚动测试赛季 2012/13–2015/16。"
      "log loss、RPS、Brier 都是越低越好。\n")

    # ---------------- Step 1 recap
    m = pd.read_parquet(MATCHES_PATH)
    w("## Step 1 回顾：数据\n")
    w(f"- 比赛 {len(m)} 场，赛季 {SEASONS[0]}–{SEASONS[-1]}，每季 380 场；缺首发 ID 的比赛 {(m.n_missing_starters > 0).sum()} 场"
      f"（共缺 {m.n_missing_starters.sum()} 个 ID）。\n")
    w("- 各赛季结果分布：\n")
    w(md(pd.crosstab(m.season, m.y, normalize="index")[["H", "D", "A"]], 3, index=True))

    # ---------------- Step 2
    f = pd.read_parquet(FEATURES_PATH)
    model = f[f.is_warmup == 0]
    w("## Step 2：特征\n")
    w("F1（球员评分）= `gk_diff`、`def_diff`、`mid_diff`、`att_diff`；F2 = F1 + 队伍表现（`gd38_diff`、`sot38_diff`、"
      "`home_promoted`、`away_promoted`）；F3 = F2 + 近期与情境的 9 个变量。T = 只用队伍表现的 4 个变量（反向对照）。"
      "聚类用 3 个差距特征（绝对值）。下表为 2009/10–2015/16（2,660 场）的描述统计：\n")
    desc = model[F3_COLS + GAP_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    w(md(desc, 3, index=True))
    w("各特征与结果的关系（2009/10–2015/16）：按 `gd38_diff` 五等分的主胜率，以及按 |gd38_diff| 五等分的平局率：\n")
    q1 = model.groupby(pd.qcut(model.gd38_diff, 5), observed=True).y.apply(lambda s: (s == "H").mean()).rename("主胜率")
    q2 = model.groupby(pd.qcut(model.abs_gd38_diff, 5), observed=True).y.apply(lambda s: (s == "D").mean()).rename("平局率")
    w(md(q1.reset_index().rename(columns={"gd38_diff": "gd38_diff 区间"}), 3))
    w(md(q2.reset_index().rename(columns={"abs_gd38_diff": "|gd38_diff| 区间"}), 3))

    # ---------------- Step 3
    bs = pd.read_csv(PROCESSED / "baseline_scores.csv")
    mk = pd.read_parquet(MARKET_PATH)
    w("## Step 3：市场基线（全部比赛）\n")
    for metric in ["log_loss", "rps", "brier"]:
        w(f"**{metric}**\n")
        w(md(bs.pivot(index="season", columns="model", values=metric)[["uniform", "frequency", "market_B365", "market_consensus"]], 4, index=True))
    w("**各家公司平均抽水（overround − 1）**\n")
    w(md(mk.groupby("season")[[c for c in mk if c.startswith("margin_")]].mean(), 4, index=True))

    # ---------------- Step 4
    w("## Step 4：聚类与比赛类型\n")
    specs = pd.read_csv(TABLES / "cluster_fold_specs.csv")
    w("### 4.1 各折选出的模型、类型占比与稳定性\n")
    w(md(specs, 3))
    grid = pd.read_csv(TABLES / "cluster_model_selection.csv")
    g4 = grid[(grid.fold == 4) & (grid.method == "gmm")].pivot(index="k", columns="cov", values="bic")
    s4 = grid[(grid.fold == 4)].assign(name=lambda x: np.where(x.method == "gmm", "GMM-" + x["cov"].fillna(""), "KMeans")) \
        .pivot(index="k", columns="name", values="silhouette")
    w("### 4.2 第 4 折训练期的模型选择\n**BIC（GMM，越低越好）**\n")
    w(md(g4, 1, index=True))
    w("**轮廓系数**\n")
    w(md(s4, 3, index=True))
    w("### 4.3 第 4 折的各簇（原始单位的簇中心、训练期占比、全部赛季的结果分布）\n")
    w(md(pd.read_csv(TABLES / "cluster_outcomes_fold4.csv"), 3))
    w("### 4.4 各类型的结果分布（GMM、|gd38| 分位数规则、K-means；样本内与样本外）\n")
    w(md(pd.read_csv(TABLES / "cluster_segment_outcomes.csv"), 3))
    w("### 4.5 GMM 类型与对照定义的一致程度（合并测试期）\n")
    w(md(pd.read_csv(TABLES / "cluster_agreement.csv"), 3))
    w("### 4.6 势均力敌组 vs 市场概率差最小的 20%（合并测试期）\n")
    w(md(pd.read_csv(TABLES / "cluster_even_vs_market.csv"), 3))
    w("### 4.7 各类型的可预测性（合并测试期）\n")
    w(md(pd.read_csv(TABLES / "cluster_predictability.csv"), 4))
    w("### 4.8 各测试赛季、各类型的平局率与主胜率\n")
    w(md(pd.read_csv(TABLES / "cluster_test_season_outcomes.csv"), 3))
    w("### 4.9 第 4 折模型下各赛季的类型占比\n")
    w(md(pd.read_csv(TABLES / "cluster_segment_share_by_season.csv"), 3))

    # ---------------- Step 5
    w("## Step 5：按类型建模\n")
    sm = pd.read_csv(TABLES / "models_summary.csv").fillna({"layer": "-"})
    sc = pd.read_csv(TABLES / "models_scores.csv").fillna({"layer": "-"})
    bsp = pd.read_csv(TABLES / "models_bootstrap.csv")
    for seg in SEGMENTS:
        w(f"### 5.{SEGMENTS.index(seg) + 1} {SEG[seg]}组（{seg}）\n")
        t = sm[sm.segment == seg].sort_values("log_loss")[["model", "layer", "log_loss", "rps", "brier", "skill", "gap_closed", "n"]]
        w("**合并测试期**（skill = 1 − log loss ÷ 本类型频率基线；gap_closed = 从频率基线到市场走完的比例）\n")
        w(md(t, 4))
        w("**分赛季 log loss**\n")
        w(md(sc[(sc.segment == seg) & (sc.season != "pooled")].pivot_table(index=["model", "layer"], columns="season", values="log_loss"), 4, index=True))
        v = bsp[(bsp.segment == seg) & (bsp.kind == "vs_baseline")][["model", "contrast", "diff", "ci_lo", "ci_hi", "p_better"]]
        w("**和基线的 log loss 差值（负数 = 更好），95% bootstrap 区间**\n")
        w(md(v, 4))
    w("### 5.4 各层特征的增量（Δ log loss，负数 = 加这一层有帮助）\n")
    inc = bsp[bsp.kind == "layer_increment"][["segment", "model", "contrast", "diff", "ci_lo", "ci_hi", "p_better"]]
    inc = inc.assign(significant=np.where((inc.ci_hi < 0) | (inc.ci_lo > 0), "yes", ""))
    w(md(inc, 4))
    w("### 5.5 第 4 折 F1+F2+F3 逻辑回归系数（gd38/sot38 已对评分残差化，近期状态变量已对评分 + gd38/sot38 残差化）\n")
    cf = pd.read_csv(TABLES / "models_coefficients.csv")
    cf = cf.assign(significant=np.where((cf.ci_lo > 0) | (cf.ci_hi < 0), "yes", ""))
    w(md(cf[["segment", "contrast", "feature", "coef", "ci_lo", "ci_hi", "significant", "C", "n", "max_vif_raw", "max_vif_residualised"]], 3))
    w("### 5.6 调参结果（每折 × 类型 × 模型 × 特征层）\n")
    tun = pd.read_csv(TABLES / "models_tuning.csv").fillna({"layer": "-"})
    w(md(tun[["fold", "segment", "model", "layer", "params", "val_log_loss", "n_train", "n_val"]], 4))

    # ---------------- Step 6
    if (TABLES / "regimes_summary.csv").exists():
        w("## Step 6：分类型 vs 全局训练，以及两种模型间集成\n")
        w("segment（per-type）= 只用本类型的比赛训练；global = 用全部比赛训练；blend（混合集成）= 同一模型、同一特征组的"
          "分类型预测与全局预测各取一半；model_avg（多模型平均）= 6 个模型预测的等权平均。权重都事先固定，不调参。"
          "都在本类型的测试比赛上评估，调参用本类型的验证比赛。\n")
        s6 = pd.read_csv(TABLES / "regimes_summary.csv").fillna({"layer": "-"})
        b6 = pd.read_csv(TABLES / "regimes_bootstrap.csv").fillna({"layer": "-"})
        for seg in SEGMENTS:
            w(f"### 6.{SEGMENTS.index(seg) + 1} {SEG[seg]}组：log loss（合并测试期）\n")
            t = s6[s6.segment == seg].pivot_table(index=["model", "layer"], columns="regime", values="log_loss")
            w(md(t, 4, index=True))
            w("**技能分数**\n")
            w(md(s6[s6.segment == seg].pivot_table(index=["model", "layer"], columns="regime", values="skill"), 4, index=True))
        w("### 6.4 训练方式之间的配对比较（Δ log loss，负数 = 前者更好）\n")
        r = b6[b6.kind == "regime"].assign(significant=lambda x: np.where((x.ci_hi < 0) | (x.ci_lo > 0), "yes", ""))
        w(md(r[["segment", "model", "layer", "contrast", "diff", "ci_lo", "ci_hi", "p_better", "significant"]], 4))
        w("### 6.5 和市场、单变量实力基线的比较\n")
        v = b6[b6.kind.isin(["vs_market", "vs_strength"])].assign(significant=lambda x: np.where((x.ci_hi < 0) | (x.ci_lo > 0), "yes", ""))
        w(md(v[["segment", "model", "layer", "contrast", "diff", "ci_lo", "ci_hi", "significant"]], 4))
        w("### 6.6 各训练方式下的特征层增量\n")
        inc6 = b6[b6.kind == "layer_increment"].assign(significant=lambda x: np.where((x.ci_hi < 0) | (x.ci_lo > 0), "yes", ""))
        w(md(inc6[["segment", "model", "contrast", "diff", "ci_lo", "ci_hi", "significant"]], 4))
        w("### 6.7 调参结果（global）\n")
        t6 = pd.read_csv(TABLES / "regimes_tuning.csv").fillna({"layer": "-"})
        w(md(t6[["fold", "segment", "regime", "model", "layer", "params", "val_log_loss", "n_train"]], 4))
    # ---------------- Step 7
    if (TABLES / "robustness_overall.csv").exists():
        w("## Step 7：稳健性——分赛季、整块 bootstrap、校准\n")
        w("关键模型：frequency（本类型）、gd38-only（per-type）、logit F1 ratings（只用球员评分，per-type）、"
          "logit T（只用队伍表现，per-type / global）、market。\n")
        w("### 7.1 分赛季 log loss\n")
        st = pd.read_csv(TABLES / "robustness_season_scores.csv")
        w(md(st.pivot_table(index=["segment", "key"], columns="season", values="log_loss"), 4, index=True))
        w("### 7.2 全部测试比赛（1,520 场，三种类型的预测合并）\n")
        ov = pd.read_csv(TABLES / "robustness_overall.csv")
        for metric in ["log_loss", "rps", "brier"]:
            w(f"**{metric}**\n")
            w(md(ov.pivot_table(index="key", columns="season", values=metric), 4, index=True))
        w("### 7.3 按比赛重抽样 vs 按轮次整块重抽样的 95% 区间\n")
        w(md(pd.read_csv(TABLES / "robustness_bootstrap.csv"), 4))
        w("### 7.4 校准统计（cal_slope = 1 为理想；< 1 过度自信；> 1 不够自信）\n")
        w(md(pd.read_csv(TABLES / "robustness_calibration_stats.csv"), 3))
        w("### 7.5 校准曲线的分箱数据（每种类型 × 结果 × 模型，按预测概率五等分）\n")
        w(md(pd.read_csv(TABLES / "robustness_calibration_curves.csv"), 3))
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
