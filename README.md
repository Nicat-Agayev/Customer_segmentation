# Customer Personality Analysis (Clustering)

An unsupervised machine-learning project that segments a company's customers into groups based on their demographics and purchasing behaviour, using **K-Means clustering**. The goal is to help the business understand its different customer types and target each one more effectively.

**Dataset:** [Customer Personality Analysis (Kaggle)](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

**[View the notebook on nbviewer](https://nbviewer.org/github/Nicat-Agayev/Customer_segmentation/blob/main/notebooks/customer_segmentation.ipynb)** — if GitHub's preview shows an error, this link always renders the notebook in full.

## Steps

1. **Data cleaning** — dropped rows with missing `Income` (24 of 2240) and removed constant columns (`Z_CostContact`, `Z_Revenue`).
2. **Feature engineering** — built interpretable features: `Age`, `Spent` (total spending), `Customer_For` (days enrolled), `Children`, `Living_With`, `Is_Parent`, and simplified `Education`.
3. **Outlier handling** — capped unrealistic `Age` and `Income` values.
4. **EDA** — explored income, spending, education and family relationships, plus a correlation matrix.
5. **Preprocessing** — label-encoded the categorical features and standardized everything with `StandardScaler` (essential for a distance-based algorithm like K-Means).
6. **Choosing k** — used the **Elbow method** and **Silhouette score** to select the number of clusters.
7. **Modelling** — fit K-Means on the scaled data; used **PCA only for 2-D visualization**.
8. **Cluster analysis** — profiled each segment.

## Results: three customer segments

| Segment | Income | Spending | Family | Share |
|---------|:------:|:--------:|--------|:-----:|
| Affluent, high spenders | ~76,000 | ~1,350 | mostly no children | ~24% |
| Mid-income parents | ~60,000 | ~800 | mostly 1 child | ~31% |
| Budget-conscious parents | ~34,000 | ~90 | mostly with children | ~45% |

Income, spending and family situation are the features that separate the segments most strongly. These groups let the company tailor its strategy — premium products for the affluent segment, value offers for the budget-conscious one.

## Repository structure

```
customer-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── marketing_campaign.csv
└── notebooks/
    └── customer_segmentation.ipynb
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/customer_segmentation.ipynb
```

Run the cells top to bottom. With `random_state=42` set throughout, the results are reproducible.

## Tools

Python · NumPy · pandas · scikit-learn · Matplotlib · seaborn
