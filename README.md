# 📈 Prédiction du Risque de Défaut de Paiement (Crédit Scoring)

## 🎯 Objectif du Projet
Ce projet a pour objectif de concevoir un modèle de classification prédictive pour évaluer le risque de défaut de paiement sur un jeu de données financiers. En s'appuyant sur des méthodes d'apprentissage supervisé, ce projet couvre l'intégralité du cycle de vie de la donnée : depuis l'exploration et le nettoyage des données brutes jusqu'à l'évaluation fine des performances du modèle. 

Ce travail s'inscrit dans le cadre de mon Master 2 Data Science (spécialisation Santé, Assurance, Finance) à l'Université Paris-Saclay, avec pour ambition de répondre à des enjeux métiers concrets (réduction de l'attrition, optimisation de la tarification, maîtrise du risque).

---

## 🛠️ Stack Technique & Méthodologie

**1. Environnement & Langages**
*   **Python :** Manipulation, analyse et modélisation algorithmique.
*   **R :** Tests statistiques préliminaires (analyses multivariées, corrélations).
*   **SQL :** Extraction et requêtage ciblé des données financières.

**2. Bibliothèques Principales**
*   *Manipulation & Calcul :* `Pandas`, `NumPy`
*   *DataViz :* `Matplotlib`, `Seaborn`
*   *Machine Learning :* `Scikit-Learn`

**3. Déroulé du Projet**
*   **EDA (Exploratory Data Analysis) :** Traitement des valeurs manquantes (imputation), détection des outliers (Z-score) et analyse des corrélations.
*   **Feature Engineering :** Création de nouvelles variables expertes (ex: ratio d'endettement), encodage des variables catégorielles (One-Hot Encoding) et normalisation des données.
*   **Modélisation Mathématique :** Entraînement de plusieurs algorithmes de classification (Régression Logistique, Random Forest, Gradient Boosting).
*   **Évaluation & Optimisation :** Recherche des hyperparamètres optimaux via `GridSearchCV` et analyse des métriques (Matrice de confusion, Courbe ROC, AUC, F1-Score) pour minimiser les faux négatifs (privilégier le rappel).

---

## 💻 Aperçu du Code (Python)

Voici un extrait de la pipeline de modélisation utilisant un algorithme `RandomForestClassifier` :

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Chargement et préparation des données
df = pd.read_csv('financial_data.csv')

# Séparation des features (X) et de la cible (y)
X = df.drop('Default_Risk', axis=1)
y = df['Default_Risk']

# Imputation basique et Standardisation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 2. Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Modélisation via Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Prédiction et Évaluation
predictions = model.predict(X_test)
prob_predictions = model.predict_proba(X_test)[:, 1]

print("--- Rapport de Classification ---")
print(classification_report(y_test, predictions))
print(f"Score AUC-ROC : {roc_auc_score(y_test, prob_predictions):.4f}")
