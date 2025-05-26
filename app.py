# app.py

import streamlit as st
import hdf5plugin
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scanpy.external as sce
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Single-cell RNA-seq Analysis (Pancreas Data)")

# Step 1: Load data
st.header("1. Loading Data")
adata = sc.read('./pancreas_data.h5ad')
st.write("Original AnnData object:")
st.write(adata)

# Step 2: Show batch counts
st.subheader("Batch Distribution")
st.write(adata.obs["batch"].value_counts())

# Step 3: Preprocessing
st.header("2. Preprocessing")
with st.spinner("Running filtering and normalization..."):
    sc.pp.filter_cells(adata, min_genes=600)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# Step 4: Plot UMAP
st.header("3. UMAP Visualization Before Batch Correction")

ax1 = sc.pl.umap(adata, color=['celltype'], legend_fontsize=10, show=False)
st.pyplot(ax1.figure)

ax2 = sc.pl.umap(adata, color=['batch'], legend_fontsize=10, show=False)
st.pyplot(ax2.figure)

# Step 5: Harmony Integration
st.header("4. Batch Correction using Harmony")
sce.pp.harmony_integrate(adata, 'batch')
sc.pp.neighbors(adata)
sc.tl.umap(adata)

st.subheader("UMAP After Harmony Correction")

ax3 = sc.pl.umap(adata, color=['batch'], legend_fontsize=10, show=False)
st.pyplot(ax3.figure)

ax4 = sc.pl.umap(adata, color=['celltype'], legend_fontsize=10, show=False)
st.pyplot(ax4.figure)

# Step 6: DEG Analysis
st.header("5. Differential Expression Analysis (Case vs Control)")

sc.tl.rank_genes_groups(
    adata,
    groupby='disease',
    method='wilcoxon',
    groups=['case'],
    reference='control',
    use_raw=False
)

deg_result = adata.uns["rank_genes_groups"]

degs_df = pd.DataFrame({
    "genes": deg_result["names"]["case"],
    "pvals": deg_result["pvals"]["case"],
    "pvals_adj": deg_result["pvals_adj"]["case"],
    "logfoldchanges": deg_result["logfoldchanges"]["case"],
})
degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])

# Differential expression labeling
degs_df["diffexpressed"] = "NS"
degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"

top_downregulated = degs_df[degs_df["diffexpressed"] == "DOWN"]
top_downregulated = top_downregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, True]).head(20)

top_upregulated = degs_df[degs_df["diffexpressed"] == "UP"]
top_upregulated = top_upregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, False]).head(81)

top_genes_combined = pd.concat([top_downregulated["genes"], top_upregulated["genes"]])
df_annotated = degs_df[degs_df["genes"].isin(top_genes_combined)]

# Step 7: Volcano Plot
st.subheader("Volcano Plot of DEGs")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=degs_df,
    x="logfoldchanges",
    y="neg_log10_pval",
    hue="diffexpressed",
    palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
    alpha=0.7,
    edgecolor=None,
    ax=ax
)
ax.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
ax.axvline(x=-1, color='gray', linestyle='dashed')
ax.axvline(x=1, color='gray', linestyle='dashed')
ax.set_xlim(-11, 11)
ax.set_ylim(25, 175)
ax.set_xlabel("log2 Fold Change", fontsize=14)
ax.set_ylabel("-log10 p-value", fontsize=14)
ax.set_title("Volcano of DEGs (Disease vs Control)", fontsize=16)
ax.legend(title="Expression", loc="upper right")
st.pyplot(fig)
