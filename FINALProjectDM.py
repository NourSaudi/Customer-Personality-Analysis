# Auto-generated from FINALProjectDM.ipynb

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


df = pd.read_csv(r"E:\ANU Material\Semester 5\Data Mining\ProjectDM\marketing_campaign.csv", sep="\t")

print("\n===== First 5 Rows =====")
print(df.head())
print("\n===== Last 5 Rows =====")
print(df.tail())

print("\n===== Dataset Info =====")
print(df.info())

print("\n===== Descriptive Statistics =====")
print(df.describe(include='all'))


# %% [code]

# -------------------------
# 2. Dataset shape
# -------------------------
print("\n===== Dataset Shape =====")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n===== Checking Missing Values Before Dropping =====")
missing_per_column = df.isnull().sum()
total_missing = missing_per_column.sum()
print(f"\nTotal missing values in dataset: {total_missing}")

missing_percentage = (missing_per_column / df.shape[0]) * 100
missing_df = pd.DataFrame({
    "Missing Values": missing_per_column,
    "Percentage (%)": missing_percentage.round(2)
})
print("\nMissing values per column with percentage:")
print(missing_df)


# %% [code]

# -------------------------
# 3. Dropping rows with any null value (as in original)
# -------------------------
print("\n===== Dropping Rows with Null Values =====")
df = df.dropna()
missing_after_drop = df.isnull().sum().sum()
print(f"Total missing values after dropping rows: {missing_after_drop}")

print("\n===== Checking Dataset Shape After Dropping =====")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# %% [code]

# -------------------------
# 4. Data types and categorical value checks
# -------------------------
print("===== Data Types and Format Consistency =====\n")
print("Data types of each column:")
print(df.dtypes)

print("\nChecking categorical columns consistency (unique values):")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f"\nColumn '{col}' - {len(unique_values)} unique values:")
    print(unique_values[:10], "...")  # first 10 only

print("\nChecking numeric columns consistency:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
    print(f"Column '{col}' - non-numeric values count: {non_numeric}")


# %% [code]

# -------------------------
# 5. Create Age column from Year_Birth safely
# -------------------------
df.loc[:, 'Age'] = 2025 - df['Year_Birth']
print("\nAdded 'Age' column (2025 - Year_Birth).")
print(df[['Year_Birth', 'Age']].head())


# %% [code]
# -------------------------
# 7. Categorical Cleaning
# -------------------------
# Simplify Marital_Status
df['Marital_Status'] = df['Marital_Status'].replace({
    'Married': 'Partner',
    'Together': 'Partner',
    'Absurd': 'Alone',
    'Widow': 'Alone',
    'YOLO': 'Alone',
    'Divorced': 'Alone',
    'Single': 'Alone'
})

print("\n===== Marital Status Distribution after Cleaning =====")
print(df['Marital_Status'].value_counts())


# %% [code]
# -------------------------
# 8. Outlier Handling
# -------------------------
# Age < 100
df = df[df['Age'] < 100]
# Income < 600000
df = df[df['Income'] < 600000]

print("\n===== Shape after removing outliers =====")
print(df.shape)


# %% [code]

# -------------------------
# 6. Exploratory visualizations for demographic features
# -------------------------
demographic_cols = ['Age', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'Income']

sns.set(style="whitegrid")

for col in demographic_cols:
    plt.figure(figsize=(12,5))
    plt.suptitle(f"Histogram & Boxplot for {col}", fontsize=16)
    
    if col in df.select_dtypes(include=['int64', 'float64']).columns:
        plt.subplot(1,2,1)
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f"{col} Histogram")
        plt.xlabel(col)
        plt.ylabel("Count")
        
        plt.subplot(1,2,2)
        sns.boxplot(x=df[col])
        plt.title(f"{col} Boxplot")
    else:
        plt.subplot(1,2,1)
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"{col} Count Plot")
        
        plt.subplot(1,2,2)
        sns.boxplot(x=pd.factorize(df[col])[0])
        plt.title(f"{col} Boxplot (Categorical)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Explanation printed after each figure
    print(f"\n[Explanation] عرضنا التوزيع والـboxplot لـ'{col}' لنتعرف على التشتت، القيم المتطرفة (outliers)، وطبيعة التوزيع — مفيد لاختيار طرق المعالجة أو التكتل لاحقًا.")


# %% [code]

# -------------------------
# 7. Spending pattern stacked bar charts
# -------------------------
spending_cols = [
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]

category_cols = ['Education', 'Marital_Status', 'Kidhome']

for category in category_cols:
    grouped = df.groupby(category)[spending_cols].sum()
    ax = grouped.plot(
        kind='bar', stacked=True, figsize=(12,6), colormap='tab20'
    )
    plt.title(f"Spending Pattern by {category} (Stacked Bar Chart)")
    plt.ylabel("Total Spending")
    plt.xlabel(category)
    plt.xticks(rotation=45)
    plt.legend(title="Products", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    print(f"\n[Explanation] المخططات المتكدسة تبين كيف يتوزع الإنفاق على فئات المنتجات داخل كل فئة من '{category}' — يساعد في فهم أي مجموعات تنفق أكثر وأين نستهدف الحملات.")


# %% [code]

# -------------------------
# 8. Correlation heatmap of key numeric attributes
# -------------------------
# ensure Age exists (we already created it)
df.loc[:, 'Age'] = 2025 - df['Year_Birth']

key_numeric_cols = [
    'Age', 'Income', 'Kidhome', 'Teenhome',
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]

corr_matrix = df[key_numeric_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Key Attributes", fontsize=16)
plt.tight_layout()
plt.show()

# Explanation
print("\n[Explanation] خريطة الارتباط تساعدنا نعرف العلاقات بين الخصائص (مثل هل الدخل مرتبط بكمية الإنفاق؟) قبل اختيار عدد الكلاسترز أو الميزات المهمة.")


# %% [code]
# -------------------------
# Product spending comparison (English labels + explanations)
# -------------------------
import matplotlib.pyplot as plt
import seaborn as sns

products = [
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]

# Total spending per product category
totals = df[products].sum().sort_values(ascending=False)

sns.set(style='whitegrid', font_scale=1.05)
fig, axes = plt.subplots(1, 1, figsize=(14,6), constrained_layout=True)
axes = [axes]

# Bar chart for totals (horizontal for readability)
sns.barplot(x=totals.values, y=totals.index, palette='viridis', ax=axes[0])
axes[0].set_title('Total Spending by Product')
axes[0].set_xlabel('Total Spending')
axes[0].set_ylabel('Product')



plt.show()


# Short English explanation to help readers interpret the charts
print("\nExplanation:")
print("- Left chart: total spending per product category (useful to identify high-revenue product groups).")
print("- Use totals to prioritize revenue-generating categories and means to target high-engagement customers.")


# %% [code]
# -------------------------
# 10. Handling Missing Values
# -------------------------

print("\nCheck missing per column (after previous operations):")
print(df.isnull().sum())

# Simple imputation (numeric mean, categorical mode)
#df.fillna(df.mean(numeric_only=True), inplace=True)
#df.fillna(df.mode().iloc[0], inplace=True)


# %% [code]


# %% [code]

# -------------------------
# 11. Creating new features
# -------------------------

df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)

df["Children"] = df["Kidhome"] + df["Teenhome"]

df["Premium_Products"] = df["MntWines"] + df["MntMeatProducts"]

df["Age"] = 2025 - df["Year_Birth"]


df["Is_Parent"] = (df["Children"] > 0).astype(int)


df["Total_Accepted_Campaigns"] = (
    df["AcceptedCmp1"] +
    df["AcceptedCmp2"] +
    df["AcceptedCmp3"] +
    df["AcceptedCmp4"] +
    df["AcceptedCmp5"] +
    df["Response"]
)

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# Create Tenure (Days since customer joined)
# Using the max date in dataset as reference
max_date = df['Dt_Customer'].max()
df['Tenure'] = (max_date - df['Dt_Customer']).dt.days

# %% [code]

# -------------------------
# 12. ColumnTransformer with StandardScaler & OneHotEncoder
# -------------------------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

print("\nNumeric columns to scale:", numeric_cols)
print("Categorical columns to encode:", categorical_cols)

preprocessor = ColumnTransformer(
    transformers= [
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
   
)

# Fit-transform to get X ready for clustering/ML
X = preprocessor.fit_transform(df)

print("\nPreprocessing completed. 'X' is ready for clustering/ML algorithms.")
print(f"Resulting matrix shape: {X.shape}")


# %% [code]
# =========================================================
# Phase D: Modeling - 2. K-Medoids Clustering 
#  Using only numeric features for clustering
# =========================================================

# 1. Install if needed (run once)
# !pip install pyclustering

# %% [code]
# 2. Imports
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------- 
# IMPORTANT FIX: Select ONLY numeric features for clustering
# (Exclude categorical to avoid distance distortion from One-Hot Encoding)
# Using YOUR features from code (e.g., Total_Spending, Children, etc.)
# -------------------------
numeric_features = [
    'Age', 'Income', 'Total_Spending', 'Children', 
    'Total_Accepted_Campaigns', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth', 'Premium_Products',
    'Is_Parent', 'Tenure'
]

# Extract numeric data from YOUR df
X_numeric = df[numeric_features]
categorical_features = ['Education', 'Marital_Status']

n_numeric = len(numeric_features)
X_for_clustering = X[:, :n_numeric]  # First n_numeric columns are the scaled numerics

print(f"Shape for clustering (numeric only, already scaled): {X_for_clustering.shape}")
print("→ No duplicate scaling! We use the scaled numerics directly from ColumnTransformer.")

print(f"Clustering using ONLY numeric features: {len(numeric_features)} features")
print(f"Final matrix shape for clustering: {X_numeric.shape}")
print("(Categorical features kept in df for profiling and insights only)")

# %% [code]
# -------------------------
# 5. Find optimal k using Silhouette Score
# -------------------------
k_range = range(2, 10)
silhouette_scores = {}

print("\nComputing Silhouette Scores for different k...")
for k in k_range:
    np.random.seed(42)
    initial_medoids = np.random.choice(len(X_for_clustering), k, replace=False).tolist()
    
    kmedoids_instance = kmedoids(X_for_clustering.tolist(), initial_medoids, data_type='points')
    kmedoids_instance.process()
    
    clusters = kmedoids_instance.get_clusters()
    
    labels = np.full(len(X_for_clustering), -1, dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    
    sil = silhouette_score(X_for_clustering, labels)
    silhouette_scores[k] = sil
    print(f"k = {k} → Silhouette Score = {sil:.4f}")

optimal_k = max(silhouette_scores, key=silhouette_scores.get)
best_sil = silhouette_scores[optimal_k]

print("\n===== Optimal K Result =====")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Best Silhouette Score: {best_sil:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', color='teal')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# %% [code]
# ------------------------- 
# 4. Final K-Medoids with optimal_k
# -------------------------
np.random.seed(42)
initial_medoids_final = np.random.choice(len(X_for_clustering), optimal_k, replace=False).tolist()

kmedoids_final = kmedoids(X_for_clustering.tolist(), initial_medoids_final, data_type='points')
kmedoids_final.process()

final_clusters = kmedoids_final.get_clusters()
final_labels = np.full(len(X_for_clustering), -1, dtype=int)
for cluster_id, cluster in enumerate(final_clusters):
    for idx in cluster:
        final_labels[idx] = cluster_id

df['Cluster_KMedoids'] = final_labels

# ------------------------- 
# 5. Cluster profiles (including categorical for richer insights)
# -------------------------
# Numeric profiles
cluster_profiles_numeric = df.groupby('Cluster_KMedoids')[numeric_features].mean().round(2)

# Categorical profiles (most common value)
cluster_profiles_cat = df.groupby('Cluster_KMedoids')[categorical_features].agg(
    lambda x: x.mode()[0] if not x.mode().empty else 'None'
)

print("\n===== Cluster Profiles - Numeric Means =====")
print(cluster_profiles_numeric)

print("\n===== Cluster Profiles - Most Common Categorical Values =====")
print(cluster_profiles_cat)

print("\n===== Cluster Sizes =====")
cluster_sizes = df['Cluster_KMedoids'].value_counts().sort_index()
for cluster in cluster_sizes.index:
    size = cluster_sizes[cluster]
    perc = (size / len(df) * 100).round(2)
    print(f"Cluster {cluster}: {size} customers ({perc}%)")

# %% [code]
# ------------------------- 
# 6. Business Insights (updated for accuracy)
# -------------------------
print("\n===== Business Insights & Marketing Recommendations =====")
overall_mean_income = df['Income'].mean()
overall_mean_spending = df['Total_Spending'].mean()

for cluster in sorted(df['Cluster_KMedoids'].unique()):
    profile_num = cluster_profiles_numeric.loc[cluster]
    profile_cat = cluster_profiles_cat.loc[cluster]
    size = cluster_sizes[cluster]
    perc = (size / len(df) * 100).round(2)
    
    print(f"\nCluster {cluster} ({size} customers, {perc}% of total):")
    print(f" - Most common Education: {profile_cat['Education']}")
    print(f" - Most common Marital Status: {profile_cat['Marital_Status']}")
    
    if (profile_num['Income'] > overall_mean_income and
        profile_num['Total_Spending'] > overall_mean_spending and
        profile_num['Children'] < 1):
        print(" → High-Value Premium Customers")
        print("   Recommendation: Luxury campaigns focused on wines and premium meats.")
    
    elif profile_num['Children'] >= 1:
        print(" → Family-Oriented Customers")
        print("   Recommendation: Family bundles, fruits, sweets, and fish promotions.")
    
    elif profile_num['Income'] < overall_mean_income:
        print(" → Budget-Conscious Customers")
        print("   Recommendation: Deals, discounts, and web-based promotions.")
    
    else:
        print(" → Balanced/General Customers")
        print("   Recommendation: Broad campaigns or A/B testing.")


# %% [code]


# ------------------------- 
# 7. Robustness Explanation
# -------------------------
print("\n===== Advantage Utilization: Robustness to Outliers =====")
print("By using only scaled numeric features, distances are meaningful, and K-Medoids' medoid selection provides superior robustness to outliers compared to centroid-based methods.")



# %% [code]

# 8. PCA Visualization
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_for_clustering)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=final_labels, cmap='viridis', alpha=0.7, s=50)
plt.title('K-Medoids Clusters (PCA 2D Projection - Numeric Features)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# 
# > "K-Medoids clustering revealed two distinct and actionable customer segments:  
# > 1. High-Value Premium Customers (34.4%): Characterized by high income, high spending on premium products, and low number of children. Recommended for luxury and VIP campaigns.  
# > 2. Family-Oriented Budget Customers (65.6%): Larger segment with average income, presence of children, and price-sensitive behavior. Recommended for family deals and promotional offers.  
# > This segmentation enables precise targeted marketing, potentially increasing campaign response rates and ROI."
# 


# %% [code]
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# =========================================================
# PHASE D.3: FUZZY LOGIC IMPLEMENTATION
# =========================================================
# We use 'Income' and 'Total_Spending' as the primary indicators of customer personality
income_range = np.arange(df['Income'].min(), df['Income'].max(), 1)
spending_range = np.arange(df['Total_Spending'].min(), df['Total_Spending'].max(), 1)
segment_range = np.arange(0, 101, 1) # Output: 0-100 score of "Segment Value"

# 2. Define Antecedents (Inputs) and Consequents (Output)
income = ctrl.Antecedent(income_range, 'income')
spending = ctrl.Antecedent(spending_range, 'spending')
segment_score = ctrl.Consequent(segment_range, 'segment_score')

# 3. Define Membership Functions (The "Personality" Logic)
# Concept: We define what "Low/Mid/High" means for this specific business
income['low'] = fuzz.trapmf(income.universe, [0, 0, 20000, 45000])
income['mid'] = fuzz.trimf(income.universe, [35000, 55000, 75000])
income['high'] = fuzz.trapmf(income.universe, [60000, 90000, df['Income'].max(), df['Income'].max()])

spending['low'] = fuzz.trapmf(spending.universe, [0, 0, 200, 600])
spending['mid'] = fuzz.trimf(spending.universe, [400, 1000, 1500])
spending['high'] = fuzz.trapmf(spending.universe, [1200, 1800, df['Total_Spending'].max(), df['Total_Spending'].max()])

# Consequent: Segment categories
segment_score['budget'] = fuzz.trimf(segment_score.universe, [0, 0, 50])
segment_score['standard'] = fuzz.trimf(segment_score.universe, [30, 50, 70])
segment_score['premium'] = fuzz.trimf(segment_score.universe, [50, 100, 100])

# 4. Define Fuzzy Rules (The "Business Intelligence" Concept)
# This is where we interpret the customer personality
rule1 = ctrl.Rule(income['high'] & spending['high'], segment_score['premium'])
rule2 = ctrl.Rule(income['mid'] | spending['mid'], segment_score['standard'])
rule3 = ctrl.Rule(income['low'] & spending['low'], segment_score['budget'])
rule4 = ctrl.Rule(income['high'] & spending['low'], segment_score['standard']) # High earners who save

# 5. Build the Control System
segment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
segment_simulator = ctrl.ControlSystemSimulation(segment_ctrl)

# 6. Apply to ALL customers (Full Integration with teammates' work)
# We loop through the dataframe to calculate a fuzzy score for every customer
fuzzy_scores = []
for index, row in df.iterrows():
    segment_simulator.input['income'] = row['Income']
    segment_simulator.input['spending'] = row['Total_Spending']
    try:
        segment_simulator.compute()
        fuzzy_scores.append(segment_simulator.output['segment_score'])
    except:
        fuzzy_scores.append(50) # Default for edge cases

df['Fuzzy_Segment_Score'] = fuzzy_scores

# 7. Final Classification based on Fuzzy Score
def classify_segment(score):
    if score < 40: return 'Budget Conscious'
    elif score < 65: return 'Standard Customer'
    else: return 'Premium/Platinum'

df['Fuzzy_Personality_Label'] = df['Fuzzy_Segment_Score'].apply(classify_segment)

# =========================================================
# VISUALIZATION & RESULTS
# =========================================================

# View the Membership Logic (Concept from Page 3 of Lab)
income.view()
plt.title("Fuzzy Logic Concept: Income Membership") # Set title while graph is "open"
plt.show() # Now show the final result

# Visualize the final segments created by Fuzzy Logic
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Income', y='Total_Spending', hue='Fuzzy_Personality_Label', palette='viridis')
plt.title("Final Customer Personality Groups (Fuzzy Logic FIS)")
plt.show()

print("\n===== Fuzzy Logic Implementation Summary =====")
print(df['Fuzzy_Personality_Label'].value_counts())
print("\nSample Output (First 5 Customers):")
print(df[['Income', 'Total_Spending', 'Fuzzy_Segment_Score', 'Fuzzy_Personality_Label']].head())

# %% [code]
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

Z_single   = linkage(X_for_clustering, method='single',   metric='euclidean')
Z_complete = linkage(X_for_clustering, method='complete', metric='euclidean')
Z_average  = linkage(X_for_clustering, method='average',  metric='euclidean')
Z_ward     = linkage(X_for_clustering, method='ward',     metric='euclidean')

# نفس الـ Dendrograms
plt.figure(figsize=(15, 10))
plt.subplot(2,2,1); dendrogram(Z_single); plt.title('Single Linkage')
plt.subplot(2,2,2); dendrogram(Z_complete); plt.title('Complete Linkage')
plt.subplot(2,2,3); dendrogram(Z_average); plt.title('Average Linkage')
plt.subplot(2,2,4); dendrogram(Z_ward); plt.axhline(y=50, color='r', linestyle='--'); plt.title('Ward Linkage (Chosen)')
plt.tight_layout()
plt.show()

# Agglomerative مع X_for_clustering
H1 = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = H1.fit_predict(X_for_clustering)  # ← التغيير الوحيد
df['Cluster_Hierarchical'] = cluster_labels

# نفس الـ Visualization & Profiling
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Income', y='Total_Spending', hue='Cluster_Hierarchical', palette='Set1', s=60)
plt.title("Customer Segments: Hierarchical Clustering")
plt.show()

print("\n===== Hierarchical Results =====")
print("Cluster Distribution:", df['Cluster_Hierarchical'].value_counts().sort_index())
print("Profile Summary:\n", df.groupby('Cluster_Hierarchical')[['Income', 'Total_Spending', 'Age', 'Children']].mean().round(2))


# %% [markdown]
# # Project Overview
# 
# ## Dataset Information
# - **Download Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personalityanalysis)
# - **Data Content**: Comprehensive customer data including demographics, income levels, spending patterns, marketing campaign responses, children information, and recency metrics.
# 
# ## Primary Objective
# Perform customer segmentation to enable targeted marketing strategies and personalized campaign approaches.
# 
# ## Selected Analytical Methods
# The project will implement and compare the following approaches:
# 1. **Hierarchical Clustering**: For dendrogram-based segmentation.
# 2. **K-Medoids Clustering**: Robust centroid-based clustering.
# 3. **Fuzzy Logic Clustering**: For probabilistic segment membership.
# 4. **Genetic Algorithm**: For optimal cluster count determination.
# 
# ---
# 
# # Project Lifecycle Framework
# 
# ## Phase A: Business Understanding
# ### Strategic Business Objectives
# 1. **Customer Segmentation**: Identify natural groupings within the customer base based on behavioral and demographic characteristics.
# 2. **Targeted Marketing**: Determine which customer segments represent the most valuable targets for specific marketing campaigns.
# 3. **Response Prediction**: Develop insights that can help predict customer responsiveness to various marketing initiatives.
# 
# ## Phase B: Data Understanding
# ### Data Exploration Procedures
# 1. **Initial Data Assessment**
#    - Load and examine dataset structure and dimensions.
#    - Display preliminary data samples.
#    - Generate comprehensive dataset information.
#    - Calculate descriptive statistics.
# 2. **Data Quality Assessment**
#    - Identify and quantify missing values across all attributes.
#    - Assess data types and format consistency.
# 3. **Exploratory Visualization**
#    - Distribution analysis of key demographic features (Age, Income).
#    - Spending pattern visualization across different customer categories.
#    - Correlation heatmap generation to identify relationships between key attributes.
# 
# ## Phase C: Data Preparation
# ### Data Preprocessing Steps
# 1. **Missing Value Treatment**
#    - Implement appropriate strategies for handling missing data.
#    - Assess impact of missing values on analytical outcomes.
# 2. **Feature Engineering**
#    - Create `Total_Spending` feature: Summation of all product category expenditures.
#    - Create `Children` feature: Combined count from `Kidhome` and `Teenhome` variables.
#    - Develop additional derived features relevant to customer behavior analysis.
# 3. **Data Transformation**
#    - Standardize numerical features to ensure equal weighting in clustering algorithms.
#    - Apply appropriate scaling techniques (StandardScaler, MinMaxScaler) based on algorithm requirements.
#    - Encode categorical variables where necessary.
# 
# ## Phase D: Modeling Implementation
# ### 1. Hierarchical Clustering
# - **Methodology**: Agglomerative clustering approach.
# - **Visualization**: Dendrogram construction to illustrate cluster hierarchy.
# - **Cluster Determination**: Analysis of dendrogram structure to identify optimal cluster count.
# - **Output**: Hierarchical customer segments with clear relationship structure.
# 
# ### 2. K-Medoids Clustering
# - **Implementation**: Using sklearn library.
# - **Comparative Analysis**: Contrast with traditional K-means results.
# - **Advantage Utilization**: Leverage medoid-based robustness to outliers.
# - **Validation**: Silhouette analysis and cluster quality metrics.
# 
# ### 3. Fuzzy Logic Clustering
# - **Approach**: Fuzzy Logic implementation.
# - **Membership Probability**: Assign probabilistic segment affiliations.
# - **Flexibility**: Allow customers to belong to multiple segments with varying degrees.
# - **Interpretation**: Analyze overlapping segment characteristics.
# 
# ### 4. Genetic Algorithm Optimization
# - **Purpose**: Automatically determine optimal number of clusters (k).
# - **Methodology**: Evolutionary algorithm searching for optimal cluster configuration.
# - **Comparison**: Validate results against traditional Elbow method findings.
# - **Fitness Function**: Optimization based on cluster cohesion and separation metrics.

