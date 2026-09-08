Ce projet vise à concevoir un modèle de classification prédictive pour évaluer le risque de défaut de paiement (crédit scoring) sur un jeu de données financiers. L'objectif est d'identifier les profils à risque en s'appuyant sur des méthodes d'apprentissage supervisé, depuis l'exploration des données brutes jusqu'à l'évaluation des performances du modèle.

Stack Technique :

- Langages : Python, R (pour les tests statistiques préliminaires).

- Bibliothèques : Pandas, NumPy (Manipulation), Matplotlib, Seaborn (Datavisualisation), Scikit-learn (Machine Learning).

- Outils : Jupyter Notebook, Git.

Méthodologie & Déroulé du projet :

1) Exploration & Nettoyage (EDA) : Traitement des valeurs manquantes (imputation), détection des outliers (Z-score) et analyse des corrélations.

2) Feature Engineering : Création de nouvelles variables (ex: ratio endettement/revenu), encodage des variables catégorielles (One-Hot) et normalisation des données quantitatives.

3) Modélisation Mathématique : Entraînement de plusieurs algorithmes de classification (Régression Logistique, Random Forest, Gradient Boosting).

4) Évaluation : Optimisation des hyperparamètres (GridSearchCV) et analyse des métriques (Matrice de confusion, Courbe ROC, AUC, F1-Score) pour privilégier le rappel (minimiser les faux négatifs).

Aperçu du Code (Python / Pandas / Scikit-Learn) :

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Chargement et préparation des données
df = pd.read_csv('financial_data.csv')
df.fillna(df.median(), inplace=True) # Imputation basique

# Feature Engineering
X = df.drop('Default_Risk', axis=1)
y = df['Default_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modélisation
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Prédiction et Évaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
