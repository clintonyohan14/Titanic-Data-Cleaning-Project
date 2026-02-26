"""
Project: Titanic Data Cleaning & Preprocessing
Author: Clinton
Objective: Clean raw Titanic dataset and prepare it for ML modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# LOAD DATASET
# --------------------------

df = pd.read_csv("train.csv")

print("="*50)
print("INITIAL DATA INSPECTION")
print("="*50)
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())
print("="*50)

# --------------------------
# DATA CLEANING
# --------------------------

# Handle missing Age (fill with median)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Remove unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Encode Gender (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encode Embarked column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------
# VISUALIZATION
# --------------------------

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Distribution")
plt.show()

# --------------------------
# SAVE CLEAN DATASET
# --------------------------

df.to_csv("clean_titanic.csv", index=False)

print("="*50)
print("FINAL DATA SUMMARY")
print("="*50)
print("Final Dataset Shape:", df.shape)
print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nProject Completed Successfully!")
print("="*50)