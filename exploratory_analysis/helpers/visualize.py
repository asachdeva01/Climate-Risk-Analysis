"""Visualization helpers for EDA.

Each function produces one chart or chart group. Plots render inline in the notebook.
Numeric and categorical helpers are separated.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Target & predictor distributions
# ---------------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame, target: str):
    """Histogram + KDE of the response variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target].dropna(), bins=30, kde=True, ax=ax, color='steelblue')
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame, columns: list, ncols: int = 3):
    """Grid of histograms for numeric predictors."""
    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(df[col].dropna(), bins=25, kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(col)
    for j in range(len(columns), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Numeric predictor-vs-target
# ---------------------------------------------------------------------------

def plot_scatter_vs_target(df: pd.DataFrame, predictors: list, target: str, ncols: int = 3):
    """Grid of scatter plots: each numeric predictor vs. the response variable."""
    nrows = -(-len(predictors) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(predictors):
        axes[i].scatter(df[col], df[target], alpha=0.3, s=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target)
        axes[i].set_title(f"{col} vs {target}")
    for j in range(len(predictors), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Heatmap"):
    """Heatmap of a correlation matrix with annotations."""
    fig, ax = plt.subplots(figsize=(min(1.2 * len(corr), 14), min(len(corr), 12)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_bar(corr_series: pd.Series, title: str):
    """Horizontal bar chart of predictor-to-target correlations, sorted by |r|."""
    fig, ax = plt.subplots(figsize=(9, max(3, len(corr_series) * 0.35)))
    ordered = corr_series.reindex(corr_series.abs().sort_values(ascending=True).index)
    colors = ['tab:red' if v < 0 else 'tab:blue' for v in ordered.values]
    ax.barh(ordered.index, ordered.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Categorical predictor-vs-target
# ---------------------------------------------------------------------------

def plot_categorical_vs_target(df: pd.DataFrame, categoricals: list,
                               target: str, ncols: int = 3):
    """Grid of boxplots: target distribution across each category level."""
    nrows = -(-len(categoricals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(categoricals):
        order = df.groupby(col)[target].median().sort_values().index
        sns.boxplot(x=col, y=target, data=df, order=order, ax=axes[i])
        axes[i].set_title(f"{target} by {col}")
        axes[i].tick_params(axis='x', rotation=20)
    for j in range(len(categoricals), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Outlier assessment
# ---------------------------------------------------------------------------

def plot_boxplots(df: pd.DataFrame, columns: list, ncols: int = 3):
    """Grid of boxplots for outlier assessment of numeric columns."""
    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color='steelblue')
        axes[i].set_title(col)
    for j in range(len(columns), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
