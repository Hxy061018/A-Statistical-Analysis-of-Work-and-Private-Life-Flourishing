# =========================================================
# 自动化卡方检验 + 对应分析（CA）+ 结果解释
# 适用于：
# SEX, Flourishing_Work, Flourishing_Private, Emotional_Status
# =========================================================

# -----------------------------
# 如首次运行，可先安装依赖：
# pip install pandas numpy scipy matplotlib seaborn
# -----------------------------

import os
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# =========================================================
# 0. 全局参数
# =========================================================
FILE_PATH = "Flourishing_Processed_Part1(1).csv"
OUTPUT_ROOT = "CA_Auto_Report"
INTERPRET_DIR = os.path.join(OUTPUT_ROOT, "interpretations")
ALPHA = 0.01

ANALYSIS_VARS = [
    "SEX",
    "Flourishing_Work",
    "Flourishing_Private",
    "Emotional_Status"
]

# 过滤规则：
# "or"  = 删除 SECTEUR 无效 或 TAILLE==0 的行
# "and" = 只有 SECTEUR 无效 且 TAILLE==0 才删除
DELETE_RULE = "or"

# 图形参数
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 1. 基础工具函数
# =========================================================
def make_dir(path):
    os.makedirs(path, exist_ok=True)


def save_df(df_, path, index=True):
    df_.to_csv(path, encoding="utf-8-sig", index=index)


def write_text(text, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(text)


def clean_string_col(series):
    """
    只对非缺失值去空格，避免把 NaN 变成字符串 'nan'
    """
    return series.where(series.isna(), series.astype(str).str.strip())


def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p < 0.0001:
        return "< 0.0001"
    return f"= {p:.4f}"


def join_items(items):
    items = [str(x) for x in items if pd.notna(x)]
    if len(items) == 0:
        return "无"
    return "、".join(items)


# =========================================================
# 2. 可视化函数
# =========================================================
def plot_contingency_heatmap(contingency, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ca_biplot_2d(row_coord, col_coord, eig_df, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    if row_coord is not None and not row_coord.empty:
        ax.scatter(row_coord["Dim1"], row_coord["Dim2"], color="blue", label="Rows", s=60)
        for idx, row in row_coord.iterrows():
            ax.text(row["Dim1"], row["Dim2"], str(idx), color="blue", fontsize=9)

    if col_coord is not None and not col_coord.empty:
        ax.scatter(col_coord["Dim1"], col_coord["Dim2"], color="red", marker="^", label="Columns", s=70)
        for idx, row in col_coord.iterrows():
            ax.text(row["Dim1"], row["Dim2"], str(idx), color="red", fontsize=9)

    xlab = f"Dim1 ({eig_df.iloc[0]['Variance_percent']:.2f}%)"
    ylab = f"Dim2 ({eig_df.iloc[1]['Variance_percent']:.2f}%)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_row_map_2d(row_coord, eig_df, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.scatter(row_coord["Dim1"], row_coord["Dim2"], color="blue", label="Rows", s=60)
    for idx, row in row_coord.iterrows():
        ax.text(row["Dim1"], row["Dim2"], str(idx), color="blue", fontsize=9)

    xlab = f"Dim1 ({eig_df.iloc[0]['Variance_percent']:.2f}%)"
    ylab = f"Dim2 ({eig_df.iloc[1]['Variance_percent']:.2f}%)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_col_map_2d(col_coord, eig_df, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.scatter(col_coord["Dim1"], col_coord["Dim2"], color="red", marker="^", label="Columns", s=70)
    for idx, row in col_coord.iterrows():
        ax.text(row["Dim1"], row["Dim2"], str(idx), color="red", fontsize=9)

    xlab = f"Dim1 ({eig_df.iloc[0]['Variance_percent']:.2f}%)"
    ylab = f"Dim2 ({eig_df.iloc[1]['Variance_percent']:.2f}%)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ca_map_1d(row_coord, col_coord, eig_df, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    if row_coord is not None and not row_coord.empty:
        ax.scatter(row_coord["Dim1"], np.ones(len(row_coord)), color="blue", label="Rows", s=60)
        for idx, row in row_coord.iterrows():
            ax.text(row["Dim1"], 1.03, str(idx), color="blue", fontsize=9, ha="center")

    if col_coord is not None and not col_coord.empty:
        ax.scatter(col_coord["Dim1"], np.zeros(len(col_coord)), color="red", marker="^", label="Columns", s=70)
        for idx, row in col_coord.iterrows():
            ax.text(row["Dim1"], 0.03, str(idx), color="red", fontsize=9, ha="center")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Columns", "Rows"])
    ax.set_xlabel(f"Dim1 ({eig_df.iloc[0]['Variance_percent']:.2f}%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_contribution(contrib_df, dim_col, title, save_path, color="steelblue"):
    if dim_col not in contrib_df.columns:
        return

    temp = contrib_df[[dim_col]].sort_values(by=dim_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    temp[dim_col].plot(kind="bar", ax=ax, color=color)
    ax.set_ylabel("Contribution (%)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. 对应分析 CA：手动实现
# =========================================================
def correspondence_analysis(contingency_df):
    """
    输入：列联表 DataFrame
    输出：CA结果字典
    """
    N = contingency_df.values.astype(float)
    n = N.sum()

    if n <= 0:
        raise ValueError("Contingency table total count must be > 0.")

    P = N / n
    r = P.sum(axis=1)  # row masses
    c = P.sum(axis=0)  # column masses

    if np.any(r == 0) or np.any(c == 0):
        raise ValueError("Zero row mass or zero column mass found; CA cannot be computed.")

    Dr_inv_sqrt = np.diag(1 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(c))

    # 标准化残差矩阵
    S = Dr_inv_sqrt @ (P - np.outer(r, c)) @ Dc_inv_sqrt

    # SVD
    U, singular_values, VT = np.linalg.svd(S, full_matrices=False)

    max_dim = min(contingency_df.shape[0] - 1, contingency_df.shape[1] - 1)
    if max_dim < 1:
        raise ValueError("CA requires at least one dimension.")

    U = U[:, :max_dim]
    singular_values = singular_values[:max_dim]
    VT = VT[:max_dim, :]
    V = VT.T

    eigenvalues = singular_values ** 2
    total_inertia = eigenvalues.sum()

    if total_inertia > 0:
        variance_percent = eigenvalues / total_inertia * 100
    else:
        variance_percent = np.zeros_like(eigenvalues)

    cumulative_percent = np.cumsum(variance_percent)
    dim_names = [f"Dim{i+1}" for i in range(max_dim)]

    # 主坐标
    F = Dr_inv_sqrt @ U @ np.diag(singular_values)  # row principal coordinates
    G = Dc_inv_sqrt @ V @ np.diag(singular_values)  # col principal coordinates

    row_coord = pd.DataFrame(F, index=contingency_df.index, columns=dim_names)
    col_coord = pd.DataFrame(G, index=contingency_df.columns, columns=dim_names)

    # 贡献率
    row_contrib_arr = np.zeros_like(F)
    col_contrib_arr = np.zeros_like(G)

    for j in range(max_dim):
        if eigenvalues[j] > 0:
            row_contrib_arr[:, j] = (r * (F[:, j] ** 2) / eigenvalues[j]) * 100
            col_contrib_arr[:, j] = (c * (G[:, j] ** 2) / eigenvalues[j]) * 100

    row_contrib = pd.DataFrame(row_contrib_arr, index=contingency_df.index, columns=dim_names)
    col_contrib = pd.DataFrame(col_contrib_arr, index=contingency_df.columns, columns=dim_names)

    # cos2
    row_dist2 = np.sum(F ** 2, axis=1)
    col_dist2 = np.sum(G ** 2, axis=1)

    row_cos2_arr = np.zeros_like(F)
    col_cos2_arr = np.zeros_like(G)

    for i in range(F.shape[0]):
        if row_dist2[i] > 0:
            row_cos2_arr[i, :] = (F[i, :] ** 2) / row_dist2[i]

    for i in range(G.shape[0]):
        if col_dist2[i] > 0:
            col_cos2_arr[i, :] = (G[i, :] ** 2) / col_dist2[i]

    row_cos2 = pd.DataFrame(row_cos2_arr, index=contingency_df.index, columns=dim_names)
    col_cos2 = pd.DataFrame(col_cos2_arr, index=contingency_df.columns, columns=dim_names)

    eig_df = pd.DataFrame({
        "Eigenvalue": eigenvalues,
        "Variance_percent": variance_percent,
        "Cumulative_percent": cumulative_percent
    }, index=dim_names)

    row_mass = pd.DataFrame({"Mass": r}, index=contingency_df.index)
    col_mass = pd.DataFrame({"Mass": c}, index=contingency_df.columns)

    return {
        "eig": eig_df,
        "row_coord": row_coord,
        "col_coord": col_coord,
        "row_contrib": row_contrib,
        "col_contrib": col_contrib,
        "row_cos2": row_cos2,
        "col_cos2": col_cos2,
        "row_mass": row_mass,
        "col_mass": col_mass
    }


# =========================================================
# 4. 解释文本辅助函数
# =========================================================
def top_contributors(contrib_df, dim, top_n=3):
    if dim not in contrib_df.columns:
        return []
    s = contrib_df[dim].sort_values(ascending=False).head(top_n)
    return [f"{idx}({val:.2f}%)" for idx, val in s.items()]


def top_sign_groups(coord_df, dim, top_n=3):
    """
    返回某一维上，正向与负向绝对值较大的类别
    """
    if dim not in coord_df.columns:
        return [], []

    s = coord_df[dim]

    pos = s[s > 0].sort_values(ascending=False).head(top_n).index.tolist()
    neg = s[s < 0].sort_values(ascending=True).head(top_n).index.tolist()  # 更负的在前

    return pos, neg


def closest_row_col_pairs(row_coord, col_coord, dims, top_n=3):
    """
    在前若干维坐标空间中找出相对接近的行-列类别组合
    仅作探索性参考
    """
    pairs = []
    for r in row_coord.index:
        for c in col_coord.index:
            rv = row_coord.loc[r, dims].to_numpy(dtype=float)
            cv = col_coord.loc[c, dims].to_numpy(dtype=float)
            d = np.linalg.norm(rv - cv)
            pairs.append((r, c, d))

    pairs = sorted(pairs, key=lambda x: x[2])[:top_n]
    return [f"{r} ↔ {c}(距离={d:.3f})" for r, c, d in pairs]


def generate_interpretation_not_enough(var1, var2):
    lines = []
    lines.append(f"【{var1} × {var2}】")
    lines.append("该变量组合在清洗后的数据中有效类别数不足，无法进行有效的卡方独立性检验或对应分析。")
    lines.append("因此本组结果不作进一步统计解释。")
    return "\n".join(lines)


def generate_interpretation(
    var1,
    var2,
    contingency,
    chi2,
    p,
    dof,
    expected_df,
    alpha=0.01,
    ca_res=None
):
    n_valid = int(contingency.values.sum())
    row_cats = list(contingency.index)
    col_cats = list(contingency.columns)

    low_exp_count = int((expected_df.values < 5).sum())
    low_exp_pct = (low_exp_count / expected_df.size) * 100 if expected_df.size > 0 else 0

    lines = []
    lines.append(f"【{var1} × {var2}】")
    lines.append(
        f"本组分析的有效样本量为 n = {n_valid}。"
        f"{var1} 共包含 {len(row_cats)} 个类别（{join_items(row_cats)}），"
        f"{var2} 共包含 {len(col_cats)} 个类别（{join_items(col_cats)}）。"
    )
    lines.append(
        f"列联表维度为 {contingency.shape[0]} × {contingency.shape[1]}。"
        f"卡方独立性检验结果为：χ²({dof}) = {chi2:.3f}，p {format_p_value(p)}。"
    )

    if low_exp_count > 0:
        lines.append(
            f"其中，期望频数小于 5 的单元格共有 {low_exp_count} 个，"
            f"占全部单元格的 {low_exp_pct:.1f}%。因此在解释结果时需要适度谨慎。"
        )

    if p >= alpha:
        lines.append(
            f"由于 p 值不小于 {alpha}，在该显著性水平下不能拒绝独立性假设，"
            f"说明 {var1} 与 {var2} 之间缺乏足够的统计证据表明二者存在显著关联。"
        )
        lines.append("因此：It is not necessary to do CA。")
        return "\n".join(lines)

    # 进入 CA 解释
    lines.append(
        f"由于 p 值小于 {alpha}，因此拒绝独立性假设，说明 {var1} 与 {var2} 之间存在统计学上的关联，"
        f"有必要继续进行对应分析（CA）。"
    )

    eig_df = ca_res["eig"]
    row_coord = ca_res["row_coord"]
    col_coord = ca_res["col_coord"]
    row_contrib = ca_res["row_contrib"]
    col_contrib = ca_res["col_contrib"]

    n_dim = eig_df.shape[0]

    if n_dim == 1:
        dim1 = eig_df.iloc[0]["Variance_percent"]
        lines.append(
            f"对应分析仅提取出 1 个有效维度。"
            f"第一维解释了 {dim1:.2f}% 的总惯量，说明主要结构差异基本集中在第一维上。"
        )
    else:
        dim1 = eig_df.iloc[0]["Variance_percent"]
        dim2 = eig_df.iloc[1]["Variance_percent"]
        cum2 = eig_df.iloc[1]["Cumulative_percent"]
        lines.append(
            f"对应分析结果显示，第一维解释了 {dim1:.2f}% 的总惯量，"
            f"第二维解释了 {dim2:.2f}% 的总惯量，前两维累计解释了 {cum2:.2f}% 的总惯量。"
        )
        lines.append(
            "因此，前两维已经可以较好地概括该变量组合的主要对应结构。"
        )

    # 第一维贡献
    row_dim1_top = top_contributors(row_contrib, "Dim1", top_n=3)
    col_dim1_top = top_contributors(col_contrib, "Dim1", top_n=3)

    lines.append(
        f"在第一维上，行类别中贡献较大的类别主要包括：{join_items(row_dim1_top)}；"
        f"列类别中贡献较大的类别主要包括：{join_items(col_dim1_top)}。"
    )

    # 第二维贡献
    if n_dim >= 2:
        row_dim2_top = top_contributors(row_contrib, "Dim2", top_n=3)
        col_dim2_top = top_contributors(col_contrib, "Dim2", top_n=3)
        lines.append(
            f"在第二维上，行类别中贡献较大的类别主要包括：{join_items(row_dim2_top)}；"
            f"列类别中贡献较大的类别主要包括：{join_items(col_dim2_top)}。"
        )

    # 第一维方向
    row_pos1, row_neg1 = top_sign_groups(row_coord, "Dim1", top_n=3)
    col_pos1, col_neg1 = top_sign_groups(col_coord, "Dim1", top_n=3)

    lines.append(
        f"从第一维坐标方向看，{var1} 中偏正向的主要类别有：{join_items(row_pos1)}，"
        f"偏负向的主要类别有：{join_items(row_neg1)}；"
        f"{var2} 中偏正向的主要类别有：{join_items(col_pos1)}，"
        f"偏负向的主要类别有：{join_items(col_neg1)}。"
    )

    # 第二维方向
    if n_dim >= 2:
        row_pos2, row_neg2 = top_sign_groups(row_coord, "Dim2", top_n=3)
        col_pos2, col_neg2 = top_sign_groups(col_coord, "Dim2", top_n=3)

        lines.append(
            f"从第二维坐标方向看，{var1} 中偏正向的主要类别有：{join_items(row_pos2)}，"
            f"偏负向的主要类别有：{join_items(row_neg2)}；"
            f"{var2} 中偏正向的主要类别有：{join_items(col_pos2)}，"
            f"偏负向的主要类别有：{join_items(col_neg2)}。"
        )

    # 相对接近的组合
    if n_dim >= 2:
        dims_for_distance = ["Dim1", "Dim2"]
    else:
        dims_for_distance = ["Dim1"]

    near_pairs = closest_row_col_pairs(row_coord, col_coord, dims_for_distance, top_n=3)
    lines.append(
        f"在低维表示图中相对接近的行-列类别组合（仅作探索性参考）包括：{join_items(near_pairs)}。"
    )

    lines.append(
        "总体而言，可优先结合前两维的解释率、各类别对维度的贡献率以及在图中的相对位置来理解变量之间的对应关系。"
        "一般来说，若某一行类别与某一列类别在图上方向相近且距离较近，则说明二者可能存在更强的对应关系；"
        "反之，若分布方向相反或距离较远，则说明其对应关系较弱。"
    )

    return "\n".join(lines)


# =========================================================
# 5. 单组分析函数
# =========================================================
def run_pair_analysis(data, var1, var2, alpha=0.01, out_root="CA_Auto_Report"):
    pair_name = f"{var1}_vs_{var2}"
    pair_dir = os.path.join(out_root, pair_name)
    make_dir(pair_dir)

    sub = data[[var1, var2]].dropna().copy()

    # 类别不足
    if sub[var1].nunique() < 2 or sub[var2].nunique() < 2:
        interpretation = generate_interpretation_not_enough(var1, var2)
        write_text(interpretation, os.path.join(pair_dir, "interpretation.txt"))

        summary_row = {
            "pair": pair_name,
            "var1": var1,
            "var2": var2,
            "n_valid": len(sub),
            "chi_square": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "decision": "Not enough levels for chi-square / CA"
        }
        save_df(pd.DataFrame([summary_row]), os.path.join(pair_dir, "summary.csv"), index=False)

        return {
            "pair_name": pair_name,
            "summary": summary_row,
            "interpretation": interpretation
        }

    contingency = pd.crosstab(sub[var1], sub[var2])

    # 再次保护
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        interpretation = generate_interpretation_not_enough(var1, var2)
        write_text(interpretation, os.path.join(pair_dir, "interpretation.txt"))

        summary_row = {
            "pair": pair_name,
            "var1": var1,
            "var2": var2,
            "n_valid": int(contingency.values.sum()),
            "chi_square": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "decision": "Not enough levels for chi-square / CA"
        }
        save_df(pd.DataFrame([summary_row]), os.path.join(pair_dir, "summary.csv"), index=False)

        return {
            "pair_name": pair_name,
            "summary": summary_row,
            "interpretation": interpretation
        }

    # 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    residuals_df = (contingency - expected_df) / np.sqrt(expected_df)

    decision = "Do CA" if p < alpha else "It is not necessary to do CA"

    # 保存基础结果
    save_df(contingency, os.path.join(pair_dir, "contingency_table.csv"))
    save_df(expected_df, os.path.join(pair_dir, "expected_counts.csv"))
    save_df(residuals_df, os.path.join(pair_dir, "standardized_residuals.csv"))

    plot_contingency_heatmap(
        contingency,
        f"Contingency Table: {var1} vs {var2}",
        os.path.join(pair_dir, "01_contingency_heatmap.png")
    )

    summary_row = {
        "pair": pair_name,
        "var1": var1,
        "var2": var2,
        "n_valid": int(contingency.values.sum()),
        "chi_square": chi2,
        "df": dof,
        "p_value": p,
        "decision": decision
    }
    save_df(pd.DataFrame([summary_row]), os.path.join(pair_dir, "summary.csv"), index=False)

    print("\n====================================================")
    print(pair_name)
    print(f"n = {summary_row['n_valid']}")
    print(f"Chi-square = {chi2:.4f}")
    print(f"df = {dof}")
    print(f"p-value = {p:.6g}")
    print(f"Decision = {decision}")

    # 不做 CA
    if p >= alpha:
        interpretation = generate_interpretation(
            var1=var1,
            var2=var2,
            contingency=contingency,
            chi2=chi2,
            p=p,
            dof=dof,
            expected_df=expected_df,
            alpha=alpha,
            ca_res=None
        )
        write_text(interpretation, os.path.join(pair_dir, "interpretation.txt"))

        return {
            "pair_name": pair_name,
            "summary": summary_row,
            "interpretation": interpretation
        }

    # 做 CA
    ca_res = correspondence_analysis(contingency)

    eig_df = ca_res["eig"]
    row_mass = ca_res["row_mass"]
    col_mass = ca_res["col_mass"]
    row_coord = ca_res["row_coord"]
    col_coord = ca_res["col_coord"]
    row_contrib = ca_res["row_contrib"]
    col_contrib = ca_res["col_contrib"]
    row_cos2 = ca_res["row_cos2"]
    col_cos2 = ca_res["col_cos2"]

    # 保存 CA 结果
    save_df(eig_df, os.path.join(pair_dir, "eigenvalues.csv"))
    save_df(row_mass, os.path.join(pair_dir, "row_masses.csv"))
    save_df(col_mass, os.path.join(pair_dir, "column_masses.csv"))
    save_df(row_coord, os.path.join(pair_dir, "row_coordinates.csv"))
    save_df(col_coord, os.path.join(pair_dir, "column_coordinates.csv"))
    save_df(row_contrib, os.path.join(pair_dir, "row_contributions.csv"))
    save_df(col_contrib, os.path.join(pair_dir, "column_contributions.csv"))
    save_df(row_cos2, os.path.join(pair_dir, "row_cos2.csv"))
    save_df(col_cos2, os.path.join(pair_dir, "column_cos2.csv"))

    # 作图
    plot_scree(
        eig_df,
        f"Scree Plot: {var1} vs {var2}",
        os.path.join(pair_dir, "02_scree_plot.png")
    )

    n_dim = eig_df.shape[0]
    if n_dim >= 2:
        plot_ca_biplot_2d(
            row_coord, col_coord, eig_df,
            f"CA Biplot: {var1} vs {var2}",
            os.path.join(pair_dir, "03_ca_biplot.png")
        )

        plot_row_map_2d(
            row_coord, eig_df,
            f"Row Map: {var1} vs {var2}",
            os.path.join(pair_dir, "04_row_map.png")
        )

        plot_col_map_2d(
            col_coord, eig_df,
            f"Column Map: {var1} vs {var2}",
            os.path.join(pair_dir, "05_column_map.png")
        )

        plot_contribution(
            row_contrib, "Dim1",
            f"Row Contribution to Dim1: {var1} vs {var2}",
            os.path.join(pair_dir, "06_row_contrib_dim1.png"),
            color="royalblue"
        )

        plot_contribution(
            col_contrib, "Dim1",
            f"Column Contribution to Dim1: {var1} vs {var2}",
            os.path.join(pair_dir, "07_col_contrib_dim1.png"),
            color="firebrick"
        )

        plot_contribution(
            row_contrib, "Dim2",
            f"Row Contribution to Dim2: {var1} vs {var2}",
            os.path.join(pair_dir, "08_row_contrib_dim2.png"),
            color="royalblue"
        )

        plot_contribution(
            col_contrib, "Dim2",
            f"Column Contribution to Dim2: {var1} vs {var2}",
            os.path.join(pair_dir, "09_col_contrib_dim2.png"),
            color="firebrick"
        )

    elif n_dim == 1:
        plot_ca_map_1d(
            row_coord, col_coord, eig_df,
            f"1D CA Map: {var1} vs {var2}",
            os.path.join(pair_dir, "03_ca_1d_map.png")
        )

        plot_contribution(
            row_contrib, "Dim1",
            f"Row Contribution to Dim1: {var1} vs {var2}",
            os.path.join(pair_dir, "04_row_contrib_dim1.png"),
            color="royalblue"
        )

        plot_contribution(
            col_contrib, "Dim1",
            f"Column Contribution to Dim1: {var1} vs {var2}",
            os.path.join(pair_dir, "05_col_contrib_dim1.png"),
            color="firebrick"
        )

    # 自动生成解释
    interpretation = generate_interpretation(
        var1=var1,
        var2=var2,
        contingency=contingency,
        chi2=chi2,
        p=p,
        dof=dof,
        expected_df=expected_df,
        alpha=alpha,
        ca_res=ca_res
    )
    write_text(interpretation, os.path.join(pair_dir, "interpretation.txt"))

    return {
        "pair_name": pair_name,
        "summary": summary_row,
        "interpretation": interpretation
    }


# =========================================================
# 6. 数据清洗
# =========================================================
def load_and_clean_data(file_path, delete_rule="or"):
    df = pd.read_csv(file_path)

    # 去空格
    for col in ["SECTEUR", "SEX", "Flourishing_Work", "Flourishing_Private", "Emotional_Status"]:
        df[col] = clean_string_col(df[col])

    # TAILLE 转数值
    df["TAILLE"] = pd.to_numeric(df["TAILLE"], errors="coerce")

    # 定义无效 SECTEUR
    invalid_secteur = df["SECTEUR"].isna() | (df["SECTEUR"] == "") | (df["SECTEUR"] == "0")

    if delete_rule == "or":
        filter_condition = ~(invalid_secteur | (df["TAILLE"] == 0))
    elif delete_rule == "and":
        filter_condition = ~(invalid_secteur & (df["TAILLE"] == 0))
    else:
        raise ValueError("delete_rule must be either 'or' or 'and'.")

    df_clean = df.loc[filter_condition].copy()

    # SEX 映射
    df_clean["SEX"] = clean_string_col(df_clean["SEX"])
    df_clean["SEX"] = df_clean["SEX"].replace({
        "0": "woman",
        "1": "man"
    })
    df_clean.loc[~df_clean["SEX"].isin(["woman", "man"]), "SEX"] = np.nan

    # 分析变量清洗
    for col in ANALYSIS_VARS:
        df_clean[col] = clean_string_col(df_clean[col])
        df_clean[col] = df_clean[col].replace(["", "nan", "None"], np.nan)

    return df, df_clean


# =========================================================
# 7. 主程序
# =========================================================
def main():
    make_dir(OUTPUT_ROOT)
    make_dir(INTERPRET_DIR)

    # 读取并清洗
    df_raw, df_clean = load_and_clean_data(FILE_PATH, delete_rule=DELETE_RULE)

    print("原始数据行数：", len(df_raw))
    print("清洗后数据行数：", len(df_clean))
    print("将分析的变量：", ANALYSIS_VARS)
    print("两两组合总数：", len(list(combinations(ANALYSIS_VARS, 2))))

    # 保存清洗后的数据
    save_df(df_clean, os.path.join(OUTPUT_ROOT, "cleaned_data.csv"), index=False)

    all_summaries = []
    all_texts = []

    for var1, var2 in combinations(ANALYSIS_VARS, 2):
        result = run_pair_analysis(
            data=df_clean,
            var1=var1,
            var2=var2,
            alpha=ALPHA,
            out_root=OUTPUT_ROOT
        )

        all_summaries.append(result["summary"])

        # 保存一份集中解释文件
        central_text_path = os.path.join(INTERPRET_DIR, f"{result['pair_name']}_interpretation.txt")
        write_text(result["interpretation"], central_text_path)

        all_texts.append(result["interpretation"])

    # 汇总表
    summary_df = pd.DataFrame(all_summaries)
    save_df(summary_df, os.path.join(OUTPUT_ROOT, "all_pairs_chi_square_summary.csv"), index=False)

    # 总解释报告
    combined_report = []
    combined_report.append("四个分类变量两两组合的自动结果解释汇总")
    combined_report.append("=" * 60)
    combined_report.append("")

    for txt in all_texts:
        combined_report.append(txt)
        combined_report.append("")
        combined_report.append("-" * 60)
        combined_report.append("")

    combined_report_text = "\n".join(combined_report)

    write_text(combined_report_text, os.path.join(OUTPUT_ROOT, "all_interpretations.txt"))
    write_text(combined_report_text, os.path.join(INTERPRET_DIR, "all_interpretations.txt"))

    # README
    readme = []
    readme.append("输出文件说明")
    readme.append("=" * 40)
    readme.append("1. cleaned_data.csv：清洗后的数据")
    readme.append("2. all_pairs_chi_square_summary.csv：所有变量组合的卡方检验汇总")
    readme.append("3. all_interpretations.txt：所有变量组合的结果解释总报告")
    readme.append("4. interpretations/：集中存放每一组的解释文本")
    readme.append("5. 每个变量组合对应一个子文件夹，内含：")
    readme.append("   - summary.csv")
    readme.append("   - contingency_table.csv")
    readme.append("   - expected_counts.csv")
    readme.append("   - standardized_residuals.csv")
    readme.append("   - interpretation.txt")
    readme.append("   - 若 p < 0.01，还包括 CA 的坐标、贡献率、cos2 和图形")
    write_text("\n".join(readme), os.path.join(OUTPUT_ROOT, "README.txt"))

    print("\n================ 分析完成 ================\n")
    print("输出文件夹：", OUTPUT_ROOT)
    print("解释汇总文件：", os.path.join(OUTPUT_ROOT, "all_interpretations.txt"))
    print("每组单独解释文件夹：", INTERPRET_DIR)
    print("\n卡方检验汇总：")
    print(summary_df)


# =========================================================
# 8. 运行入口
# =========================================================
if __name__ == "__main__":
    main()
