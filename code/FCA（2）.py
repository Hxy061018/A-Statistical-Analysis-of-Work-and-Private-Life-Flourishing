# =========================================================
# 简化版：FONCTION 与 3 个变量的卡方检验 + 对应分析 + 自动报告
# =========================================================

# 如首次运行，请先安装：
# pip install pandas numpy scipy matplotlib seaborn

import os
import re
import unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# =========================================================
# 1. 参数
# =========================================================
FILE_PATH = "Flourishing_Processed_Part1(1).csv"
OUTPUT_DIR = "FONCTION_CA_Results"
ALPHA = 0.01
LOW_FREQ_THRESHOLD = 5   # FONCTION_GROUP 低频类别合并阈值

TARGET_VARS = [
    "Flourishing_Work",
    "Flourishing_Private",
    "Emotional_Status"
]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 2. 工具函数
# =========================================================
def make_dir(path):
    os.makedirs(path, exist_ok=True)

def save_df(df_, path, index=True):
    df_.to_csv(path, encoding="utf-8-sig", index=index)

def write_text(text, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(text)

def clean_string_col(series):
    return series.where(series.isna(), series.astype(str).str.strip())

def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p < 0.0001:
        return "< 0.0001"
    return f"= {p:.4f}"

def join_items(items):
    items = [str(x) for x in items if pd.notna(x)]
    return "、".join(items) if len(items) > 0 else "无"

# =========================================================
# 3. FONCTION 标准化和归类
# =========================================================
def strip_accents(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text

def normalize_text(text):
    if pd.isna(text):
        return np.nan
    text = str(text).strip().lower()
    if text in ["", "nan", "none"]:
        return np.nan
    text = strip_accents(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text != "" else np.nan

def classify_fonction(text):
    """
    把 FONCTION 合并成几个较稳定的大类
    你后面如果想手调，只改这个函数即可
    """
    if pd.isna(text):
        return np.nan

    t = text

    # 助理/秘书/支持类
    if any(k in t for k in [
        "assistante", "assistant", "secretaire", "office manager",
        "agent administratif", "administratif", "support", "receptionniste"
    ]):
        return "assistant_support"

    # 负责人/经理/主管类
    if any(k in t for k in [
        "responsable", "manager", "chef de projet", "chef de projets",
        "chef de service", "chef de production", "manager de proximite"
    ]):
        return "manager_responsable"

    # 总监/高管/负责人/经营者
    if any(k in t for k in [
        "directeur", "directrice", "dirigeant", "dirigeante",
        "gerant", "gerante", "pdg", "president", "presidente",
        "cadre", "daf", "dg"
    ]):
        return "director_executive"

    # 咨询/教练/培训
    if any(k in t for k in [
        "consultant", "consultante", "coach", "formateur", "formatrice",
        "conseil", "therapeute", "psychotherapeute", "hypnotherapeute"
    ]):
        return "consulting_coaching_training"

    # 教育/研究
    if any(k in t for k in [
        "enseignant", "enseignante", "professeur", "doctorant",
        "doctorante", "ater", "recherche", "enseignant chercheur"
    ]):
        return "education_research"

    # 医疗/健康
    if any(k in t for k in [
        "medecin", "infirmier", "infirmiere", "psychologue",
        "assistante sociale", "secretaire medicale", "ergotherapeute",
        "kinesitherapeute", "sante", "medical"
    ]):
        return "health_medical"

    # 工程/技术/IT
    if any(k in t for k in [
        "ingenieur", "developpeur", "informatique", "technicien",
        "analyste", "r d", "developpement", "gestionnaire informatique"
    ]):
        return "engineering_it_technical"

    # 销售/市场/传播
    if any(k in t for k in [
        "commercial", "marketing", "communication", "vente", "ventes",
        "webmarketing", "relation client", "vrp", "technico commercial"
    ]):
        return "sales_marketing_communication"

    # 人力/法务/行政
    if any(k in t for k in [
        "rh", "ressources humaines", "drh", "juriste", "juridique",
        "paie", "recrutement", "administration"
    ]):
        return "hr_legal_admin"

    # 自由职业/创业
    if any(k in t for k in [
        "independant", "independante", "auto entrepreneur", "liberal", "entrepreneur"
    ]):
        return "self_employed"

    return "other"

def collapse_low_frequency(series, threshold=5, other_label="other_low_freq"):
    freq = series.value_counts(dropna=True)
    rare_levels = freq[freq < threshold].index
    return series.apply(lambda x: other_label if pd.notna(x) and x in rare_levels else x)

# =========================================================
# 4. CA 计算
# =========================================================
def correspondence_analysis(contingency_df):
    N = contingency_df.values.astype(float)
    n = N.sum()

    P = N / n
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    Dr_inv_sqrt = np.diag(1 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(c))

    S = Dr_inv_sqrt @ (P - np.outer(r, c)) @ Dc_inv_sqrt
    U, singular_values, VT = np.linalg.svd(S, full_matrices=False)

    max_dim = min(contingency_df.shape[0] - 1, contingency_df.shape[1] - 1)
    U = U[:, :max_dim]
    singular_values = singular_values[:max_dim]
    VT = VT[:max_dim, :]
    V = VT.T

    eigenvalues = singular_values ** 2
    total_inertia = eigenvalues.sum()
    variance_percent = eigenvalues / total_inertia * 100 if total_inertia > 0 else np.zeros_like(eigenvalues)
    cumulative_percent = np.cumsum(variance_percent)

    dim_names = [f"Dim{i+1}" for i in range(max_dim)]

    F = Dr_inv_sqrt @ U @ np.diag(singular_values)
    G = Dc_inv_sqrt @ V @ np.diag(singular_values)

    row_coord = pd.DataFrame(F, index=contingency_df.index, columns=dim_names)
    col_coord = pd.DataFrame(G, index=contingency_df.columns, columns=dim_names)

    row_contrib_arr = np.zeros_like(F)
    col_contrib_arr = np.zeros_like(G)

    for j in range(max_dim):
        if eigenvalues[j] > 0:
            row_contrib_arr[:, j] = (r * (F[:, j] ** 2) / eigenvalues[j]) * 100
            col_contrib_arr[:, j] = (c * (G[:, j] ** 2) / eigenvalues[j]) * 100

    row_contrib = pd.DataFrame(row_contrib_arr, index=contingency_df.index, columns=dim_names)
    col_contrib = pd.DataFrame(col_contrib_arr, index=contingency_df.columns, columns=dim_names)

    eig_df = pd.DataFrame({
        "Eigenvalue": eigenvalues,
        "Variance_percent": variance_percent,
        "Cumulative_percent": cumulative_percent
    }, index=dim_names)

    return {
        "eig": eig_df,
        "row_coord": row_coord,
        "col_coord": col_coord,
        "row_contrib": row_contrib,
        "col_contrib": col_contrib
    }

# =========================================================
# 5. 画图
# =========================================================
def plot_heatmap(contingency, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_scree(eig_df, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, len(eig_df) + 1)
    ax.bar(x, eig_df["Variance_percent"], color="steelblue")
    ax.plot(x, eig_df["Variance_percent"], marker="o", color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dim{i}" for i in x])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Explained inertia (%)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_ca_biplot(row_coord, col_coord, eig_df, title, save_path):
    if eig_df.shape[0] < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.scatter(row_coord["Dim1"], row_coord["Dim2"], color="blue", label="Rows", s=60)
    ax.scatter(col_coord["Dim1"], col_coord["Dim2"], color="red", marker="^", label="Columns", s=70)

    for idx, row in row_coord.iterrows():
        ax.text(row["Dim1"], row["Dim2"], str(idx), color="blue", fontsize=9)

    for idx, row in col_coord.iterrows():
        ax.text(row["Dim1"], row["Dim2"], str(idx), color="red", fontsize=9)

    ax.set_xlabel(f"Dim1 ({eig_df.iloc[0]['Variance_percent']:.2f}%)")
    ax.set_ylabel(f"Dim2 ({eig_df.iloc[1]['Variance_percent']:.2f}%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# =========================================================
# 6. 自动解释
# =========================================================
def top_contributors(contrib_df, dim, top_n=3):
    if dim not in contrib_df.columns:
        return []
    s = contrib_df[dim].sort_values(ascending=False).head(top_n)
    return [f"{idx}({val:.2f}%)" for idx, val in s.items()]

def generate_report_text(var2, contingency, chi2, p, dof, expected_df, ca_res=None):
    n_valid = int(contingency.values.sum())
    low_exp_count = int((expected_df.values < 5).sum())

    lines = []
    lines.append(f"【FONCTION_GROUP × {var2}】")
    lines.append(f"有效样本量：n = {n_valid}")
    lines.append(f"卡方独立性检验：χ²({dof}) = {chi2:.3f}, p {format_p_value(p)}")
    lines.append(f"FONCTION_GROUP 类别：{join_items(contingency.index.tolist())}")
    lines.append(f"{var2} 类别：{join_items(contingency.columns.tolist())}")
    lines.append(f"期望频数 < 5 的单元格数：{low_exp_count}")

    if p >= ALPHA:
        lines.append(f"由于 p ≥ {ALPHA}，因此在该显著性水平下，不需要继续做对应分析。")
        lines.append("结论：It is not necessary to do CA.")
        return "\n".join(lines)

    lines.append(f"由于 p < {ALPHA}，说明 FONCTION_GROUP 与 {var2} 存在显著关联，因此继续进行对应分析。")

    eig_df = ca_res["eig"]
    row_contrib = ca_res["row_contrib"]
    col_contrib = ca_res["col_contrib"]

    if eig_df.shape[0] >= 1:
        lines.append(f"第一维解释率：{eig_df.iloc[0]['Variance_percent']:.2f}%")
    if eig_df.shape[0] >= 2:
        lines.append(f"第二维解释率：{eig_df.iloc[1]['Variance_percent']:.2f}%")
        lines.append(f"前两维累计解释率：{eig_df.iloc[1]['Cumulative_percent']:.2f}%")

    row_top1 = top_contributors(row_contrib, "Dim1", top_n=3)
    col_top1 = top_contributors(col_contrib, "Dim1", top_n=3)
    lines.append(f"第一维上，FONCTION_GROUP 贡献最大的类别：{join_items(row_top1)}")
    lines.append(f"第一维上，{var2} 贡献最大的类别：{join_items(col_top1)}")

    if eig_df.shape[0] >= 2:
        row_top2 = top_contributors(row_contrib, "Dim2", top_n=3)
        col_top2 = top_contributors(col_contrib, "Dim2", top_n=3)
        lines.append(f"第二维上，FONCTION_GROUP 贡献最大的类别：{join_items(row_top2)}")
        lines.append(f"第二维上，{var2} 贡献最大的类别：{join_items(col_top2)}")

    lines.append("解释建议：优先结合前两维解释率、贡献率和双标图中类别的相对位置理解变量之间的对应关系。")

    return "\n".join(lines)

# =========================================================
# 7. 数据读取和清洗
# =========================================================
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    for col in ["SECTEUR", "FONCTION", "Flourishing_Work", "Flourishing_Private", "Emotional_Status"]:
        df[col] = clean_string_col(df[col])

    df["TAILLE"] = pd.to_numeric(df["TAILLE"], errors="coerce")

    # 按你之前的逻辑：删除 SECTEUR 无效 或 TAILLE == 0
    invalid_secteur = df["SECTEUR"].isna() | (df["SECTEUR"] == "") | (df["SECTEUR"] == "0")
    df_clean = df.loc[~(invalid_secteur | (df["TAILLE"] == 0))].copy()

    # 处理目标变量
    for col in TARGET_VARS:
        df_clean[col] = clean_string_col(df_clean[col])
        df_clean[col] = df_clean[col].replace(["", "nan", "None"], np.nan)

    # 标准化和归类 FONCTION
    df_clean["FONCTION_NORM"] = df_clean["FONCTION"].apply(normalize_text)
    df_clean["FONCTION_GROUP_RAW"] = df_clean["FONCTION_NORM"].apply(classify_fonction)
    df_clean["FONCTION_GROUP"] = collapse_low_frequency(
        df_clean["FONCTION_GROUP_RAW"],
        threshold=LOW_FREQ_THRESHOLD,
        other_label="other_low_freq"
    )

    return df, df_clean

# =========================================================
# 8. 单个分析
# =========================================================
def run_one_analysis(df, var2):
    pair_name = f"FONCTION_GROUP_vs_{var2}"
    pair_dir = os.path.join(OUTPUT_DIR, pair_name)
    make_dir(pair_dir)

    sub = df[["FONCTION_GROUP", var2]].dropna().copy()

    contingency = pd.crosstab(sub["FONCTION_GROUP"], sub[var2])

    # 保存列联表
    save_df(contingency, os.path.join(pair_dir, "contingency_table.csv"))
    plot_heatmap(
        contingency,
        f"FONCTION_GROUP vs {var2}",
        os.path.join(pair_dir, "01_heatmap.png")
    )

    # 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    save_df(expected_df, os.path.join(pair_dir, "expected_counts.csv"))

    summary = pd.DataFrame([{
        "pair": pair_name,
        "n_valid": int(contingency.values.sum()),
        "chi_square": chi2,
        "df": dof,
        "p_value": p,
        "decision": "Do CA" if p < ALPHA else "It is not necessary to do CA"
    }])
    save_df(summary, os.path.join(pair_dir, "summary.csv"), index=False)

    print(f"\n{pair_name}")
    print(summary)

    if p < ALPHA:
        ca_res = correspondence_analysis(contingency)

        save_df(ca_res["eig"], os.path.join(pair_dir, "eigenvalues.csv"))
        save_df(ca_res["row_coord"], os.path.join(pair_dir, "row_coordinates.csv"))
        save_df(ca_res["col_coord"], os.path.join(pair_dir, "column_coordinates.csv"))
        save_df(ca_res["row_contrib"], os.path.join(pair_dir, "row_contributions.csv"))
        save_df(ca_res["col_contrib"], os.path.join(pair_dir, "column_contributions.csv"))

        plot_scree(
            ca_res["eig"],
            f"Scree Plot: FONCTION_GROUP vs {var2}",
            os.path.join(pair_dir, "02_scree_plot.png")
        )

        if ca_res["eig"].shape[0] >= 2:
            plot_ca_biplot(
                ca_res["row_coord"],
                ca_res["col_coord"],
                ca_res["eig"],
                f"CA Biplot: FONCTION_GROUP vs {var2}",
                os.path.join(pair_dir, "03_ca_biplot.png")
            )

        report_text = generate_report_text(var2, contingency, chi2, p, dof, expected_df, ca_res=ca_res)
    else:
        report_text = generate_report_text(var2, contingency, chi2, p, dof, expected_df, ca_res=None)

    write_text(report_text, os.path.join(pair_dir, "report.txt"))

    return summary.iloc[0].to_dict(), report_text

# =========================================================
# 9. 主程序
# =========================================================
def main():
    make_dir(OUTPUT_DIR)

    df_raw, df_clean = load_and_prepare_data(FILE_PATH)

    print("原始数据行数：", len(df_raw))
    print("清洗后数据行数：", len(df_clean))

    # 保存清洗数据和 FONCTION 分类检查表
    save_df(df_clean, os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)

    mapping_check = df_clean[[
        "FONCTION", "FONCTION_NORM", "FONCTION_GROUP_RAW", "FONCTION_GROUP"
    ]].drop_duplicates()
    save_df(mapping_check, os.path.join(OUTPUT_DIR, "fonction_mapping_check.csv"), index=False)

    all_summary = []
    all_reports = []

    for var2 in TARGET_VARS:
        summary_row, report_text = run_one_analysis(df_clean, var2)
        all_summary.append(summary_row)
        all_reports.append(report_text)

    summary_df = pd.DataFrame(all_summary)
    save_df(summary_df, os.path.join(OUTPUT_DIR, "all_summary.csv"), index=False)

    combined_report = []
    combined_report.append("FONCTION_GROUP 与 3 个变量的对应分析报告汇总")
    combined_report.append("=" * 60)
    combined_report.append("")

    for txt in all_reports:
        combined_report.append(txt)
        combined_report.append("")
        combined_report.append("-" * 60)
        combined_report.append("")

    write_text("\n".join(combined_report), os.path.join(OUTPUT_DIR, "all_reports.txt"))

    print("\n分析完成。结果都在文件夹：", OUTPUT_DIR)

if __name__ == "__main__":
    main()
