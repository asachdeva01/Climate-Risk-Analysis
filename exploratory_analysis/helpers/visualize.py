"""Visualization helpers for EDA.

Each function produces one chart or chart group. The notebook calls these
so plotting logic stays out of narrative cells. Figures render inline in
the notebook — we do not save them separately.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, target: str = 'climate_risk_index'):
    """Histogram + KDE of the response variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target].dropna(), bins=25, kde=True, ax=ax, color='steelblue')
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    plt.tight_layout()
    plt.show()


def plot_predictor_distributions(df: pd.DataFrame, predictors: list, ncols: int = 3):
    """Grid of histograms for predictor variables."""
    nrows = -(-len(predictors) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(predictors):
        sns.histplot(df[col].dropna(), bins=25, kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(col)
    for j in range(len(predictors), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_scatter_vs_target(df: pd.DataFrame, predictors: list,
                           target: str = 'climate_risk_index', ncols: int = 3):
    """Grid of scatter plots: each predictor vs. the response variable."""
    nrows = -(-len(predictors) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(predictors):
        axes[i].scatter(df[col], df[target], alpha=0.4, s=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target)
        axes[i].set_title(f"{col} vs {target}")
    for j in range(len(predictors), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Heatmap"):
    """Heatmap of a correlation matrix with annotations."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, columns: list, ncols: int = 3):
    """Grid of boxplots for outlier assessment."""
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
