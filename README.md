# 🏦 Analyse des risques bancaires (Banque de France)

## 🎯 Objectif du projet
Ce projet d'analyse exploratoire de données (EDA) s'inscrit dans le cadre de mes travaux en Data Science et Credit Scoring. L'objectif est d'étudier l'évolution macro-prudentielle des risques du secteur bancaire français à partir des séries temporelles de la Banque de France (Webstat). Cette approche macro-économique sert de contexte et de variable explicative pour enrichir de futurs modèles de prédiction de défaut de paiement.

---

## 🛠️ Outils et langages informatiques
* **Python :** Manipulation et nettoyage des données (`Pandas`, `NumPy`), datavisualisation (`Matplotlib`).
* **Business Intelligence :** `Power BI` et `Power Query` pour la modélisation et la conception de tableaux de bord interactifs.
* **Environnement :** Jupyter Notebook, Git / GitHub.

---

## 📊 Méthodologie utilisée
1. **Data Engineering & Nettoyage :** Chargement d'un fichier CSV brut issu de l'Open Data institutionnel (séparateur `;`, gestion de l'encodage `utf-8-sig` pour neutraliser le BOM).
2. **Traitement temporel :** Conversion stricte de la colonne `time_period_start` au format datetime et typage numérique de la métrique `obs_value`.
3. **Agrégation (Data Wrangling) :** Regroupement des données par période et calcul de la moyenne (`groupby().mean()`) pour transformer un volume de près de 3 000 lignes hétérogènes en un indicateur macro-économique global et lisible couvrant la période 2006-2024.
4. **Restitution :** Génération d'une courbe d'évolution sous Python et création d'un rapport interactif sous Power BI.

---

## 🚀 Comment lancer le script
1. Clonez ce repository sur votre machine.
2. Assurez-vous d'avoir installé les bibliothèques nécessaires en exécutant la commande suivante dans votre terminal :
   ```bash
   pip install pandas numpy matplotlib
3. Placez le fichier source donnees_banque_france.csv dans le même dossier que le code du projet.
4. Ouvrez Jupyter Notebook, chargez le script d'analyse et exécutez les cellules une par une pour relancer le pipeline de nettoyage complet.
5. Visualisez le résultat généré et récupérez l'image de la courbe sauvegardée automatiquement dans le répertoire.

💡 À quelle question est-ce que cela répond ?

Pourquoi combiner des données macro-économiques avec du Credit Scoring ?

Le comportement de remboursement d'un emprunteur ne dépend pas uniquement de son profil individuel (micro), mais aussi du climat économique global (macro). Injecter cet indicateur dans un modèle de Machine Learning permet d'ajuster le niveau de risque en période de crise et d'éviter le vieillissement des prédictions (concept drift).

Comment avez-vous surmonté la complexité du fichier brut de la Banque de France ?

Par la mise en place d'un pipeline rigoureux sous Pandas : gestion des spécificités d'encodage, suppression des lignes corrompues, conversion de types et agrégation par moyenne pour lisser la tendance générale du marché sans complexité algorithmique superflue.

Quel est l'apport de la partie Power BI ?

Elle démontre une double compétence en combinant le code scientifique (Python) et la restitution orientée Business Intelligence, indispensable pour communiquer efficacement les résultats chiffrés aux équipes métiers.
