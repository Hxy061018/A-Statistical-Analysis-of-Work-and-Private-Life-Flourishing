# 解码心理健康：工作与私人生活繁荣度的统计分析

(Decoding Mental Health: A Statistical Analysis of Work and Private Life Flourishing)

## 📋 项目简介

本项目基于一份包含多维心理测量尺度的问卷数据集，通过严格的数据清洗与特征工程，运用多元统计分析方法（PCA、FCA、MCA、LDA及聚类分析），深入探究了工作与私人生活场景下的心理健康状态。研究旨在揭示情绪健康、社会福祉与心理福祉的结构特征，并识别不同人口统计学群体在心理繁荣度上的分化规律，为企业员工关怀与科学的心理健康评估提供数据驱动的实证依据。

## 🎯 研究目标

- 验证心理健康三维框架（情绪、社会、心理福祉）在职场与私人场景下的稳健性。
- 探索工作繁荣度、私人生活繁荣度与情绪状态之间的深层映射关系。
- 降维并识别不同人口统计学特征（如年龄、婚姻、学历、企业规模）下的典型心理繁荣度群体画像。
- 评估核心心理因子在区分个体情绪状态与家庭客观状况方面的判别效力。

## 📊 数据集

- **数据来源**：调查问卷原始数据 (`Flourishing_N317_v2.csv`)
- **样本规模**：原始样本 317 份，经过严格的缺失值清理（50%有效性阈值）与 KNN 填补后，最终保留 **234 份** 有效高质量样本。
- **关键变量**：
  - **人口统计学**：年龄 (AGE)、性别 (SEX)、家庭状况 (SITUFAM)、学历 (NIVEAUETUDE)、企业规模 (TAILLE) 等。
  - **心理健康量表 (MHC-SF)**：工作环境心理健康 (MHCB)、私人环境心理健康 (MHCC)。
  - **情绪量表**：积极情绪 (PE) 与消极情绪 (NE)，并由此衍生情绪状态分类 (Positivity Ratio)。
  - **性格优势**：性格优势特征 (Force) 及使用频率 (FQ)。
  - **心流状态**：心流与工作表现 (Flux)。

## 🔧 技术栈

- **编程语言**：Python
- **数据处理与科学计算**：Pandas, NumPy, SciPy
- **机器学习与统计分析**：Scikit-learn (KNNImputer, PCA, KMeans, LinearDiscriminantAnalysis), fanalysis (MCA)
- **数据可视化**：Matplotlib, Seaborn
- **核心算法**：主成分分析 (PCA)、因子对应分析 (FCA)、多重对应分析 (MCA)、线性判别分析 (LDA)、K-Means 聚类

## 📁 项目结构

```
A-Statistical-Analysis-of-Work-and-Private-Life-Flourishing/
├── data/                 # 数据文件夹
│   ├── raw/              # 原始问卷数据 (Flourishing_N317_v2.csv)
│   └── processed/        # 清洗与特征工程后的数据
├── notebooks/            # Jupyter笔记本 (探索性分析)
├── src/                  # 源代码
│   ├── data_processing.py # 缺失值处理、KNN填补、特征衍生(动态比例阈值法)
│   ├── analysis.py        # PCA降维、FCA/MCA对应分析、LDA判别分析
│   └── visualization.py   # 热力图、碎石图、散点图及混淆矩阵绘制
├── results/              # 分析结果和可视化图表输出
├── README.md             # 项目说明文档
└── requirements.txt      # 项目依赖
```

## 🚀 快速开始

### 环境配置

```
# 克隆项目
git clone [https://github.com/Hxy061018/A-Statistical-Analysis-of-Work-and-Private-Life-Flourishing.git](https://github.com/Hxy061018/A-Statistical-Analysis-of-Work-and-Private-Life-Flourishing.git)
cd A-Statistical-Analysis-of-Work-and-Private-Life-Flourishing

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行分析

```
# 1. 运行数据清洗与预处理
python src/data_processing.py

# 2. 运行统计分析模型 (PCA, FCA, MCA, LDA)
python src/analysis.py

# 3. 生成可视化结果
python src/visualization.py
```

## 📈 主要发现

- **心理健康结构的稳健性**：PCA 提取的维度与经典的心理健康三维模型高度吻合，明确区分了“职业积极功能因子”与“职业积极情绪因子”。
- **工作与生活的镜像一致性**：FCA 表明工作繁荣度与私人生活繁荣度具有极显著的跨领域镜像关联，两者均与积极情绪健康直接挂钩。
- **群体画像显著分化**：MCA 与 K-Means 聚类识别出三大典型群体：
  1. 高抑郁风险群（离异/高知/女性主导）。
  2. 适度繁荣的稳定职场群（中青年/已婚/育儿期）。
  3. 全面繁荣的成熟优势群（小微企业/年长员工）。
- **心理状态与外部标签解耦**：LDA 模型能以 88.98% 的高准确率通过心理因子预测情绪状态，但在预测家庭客观状况时失效，证实了个体主观心理特质与客观人口标签存在解耦现象。

## 👥 贡献者 (哈尔滨工业大学 管理学院)

- **Haoran Cheng**: 主成分分析 (PCA)
- **Chenting Lin**: 判别分析 (DA)
- **Xiaoyu Hu**: 数据清理与特征工程 (Data Cleaning)
- **Haoyu Yang**: 因子对应分析 (FCA)
- **Jingyi Hou**: 多重对应分析与聚类 (MCA & Clustering)

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有任何问题或建议，欢迎通过以下方式联系：

- GitHub Issues：[填写您的项目issue页面链接]
- Email：[填写您的邮箱]

**最后更新**：2026-04-21
