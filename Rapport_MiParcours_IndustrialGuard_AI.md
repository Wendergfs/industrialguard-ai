# Rapport d’avancement de mi-parcours — IndustrialGuard AI

> **Cycle :** Ingénieur en IA & Data Science — 3ème année
> **Module :** Deep Learning 2 / Deep Learning Avancé
> **Enseignant :** Haythem Ghazouani
> **Année universitaire :** 2026–2027
> **Date de rédaction :** 17 novembre 2026

## 1. Fiche d’identité et état global

| Champ | Informations du groupe |
| :--- | :--- |
| **Intitulé du sujet** | IndustrialGuard AI — détection et localisation d’anomalies visuelles sur des pièces industrielles avec PaDiM, PatchCore et une interface EfficientAD en cours d’évaluation. |
| **Track métier** | Inspection visuelle industrielle / contrôle qualité par IA. Cette spécialité n’est pas explicitement proposée dans les quatre tracks du gabarit. |
| **Identifiant du groupe** | IndustrialGuard AI — numéro de groupe à confirmer. |
| **Dépôt Git** | [github.com/Wendergfs/industrialguard-ai](https://github.com/Wendergfs/industrialguard-ai) |
| **Workspace MLflow** | Serveur local : [http://127.0.0.1:5000](http://127.0.0.1:5000). Les exécutions et artefacts restent locaux. |
| **Avancement global** | **Environ 55 % — estimation provisoire à valider par l’équipe.** Le pipeline de données, les modèles de détection, l’évaluation, le suivi MLOps et la CI sont implémentés. Les benchmarks contrôlés de Phase 7 et la validation indépendante restent à réaliser. L’API et l’interface web du gabarit ne font pas partie du périmètre actuellement visible dans le dépôt. |

### Membres et rôles

1. **Gafsi Mohamed Aziz** —
2. **Rayen Ammar** —
3. **Selim Ben Fraj** —

Les responsabilités individuelles ne sont pas attribuées dans les éléments du projet consultés. La matrice RACI ci-dessous est donc à compléter collectivement.

## 2. Travail réalisé et éléments vérifiables

### 2.1 Données et versioning

- **Jeux de données :** MPDD, benchmark principal de composants mécaniques, et MVTec AD, benchmark industriel complémentaire.
- **Audit MPDD :** 1 346 images répertoriées — 1 064 normales et 282 défectueuses; 282 masques de référence appariés; aucune annotation manquante dans le rapport d’audit. Six catégories, images de 1 024 × 1 024.
- **Audit MVTec AD :** 5 354 images — 4 096 normales et 1 258 défectueuses; 1 258 masques appariés; aucune annotation manquante dans le rapport d’audit. Quinze catégories, résolutions de 700 × 700 à 1 024 × 1 024.
- **Préparation des données :** lecture des archives sans extraction globale, découverte des images et masques, vérification des fichiers et inventaires CSV/JSON; pipeline PyTorch avec gestion des splits, transformations et cache contrôlé.
- **DVC :** métadonnées DVC présentes pour `configs/`, les rapports d’audit, le protocole expérimental et les informations de reproductibilité. Un remote Google Drive est configuré. Les archives sources et les checkpoints ne sont pas suivis par DVC dans ce dépôt.
- **Limite actuelle :** les chemins des archives sont fournis par variables d’environnement; ils ne sont pas enregistrés dans la configuration partagée. Le rapport ne publie donc pas de lien d’archive que le dépôt ne documente pas.

### 2.2 Modèles et évaluation

- **Modèles présents :** PaDiM, PatchCore et une implémentation/architecture EfficientAD. Les outils comprennent l’entraînement ou ajustement, le chargement de checkpoints, l’inférence, les seuils et les métriques de détection/localisation.
- **Nature des méthodes :** PaDiM et PatchCore sont des méthodes fondées sur les caractéristiques et les données normales; elles n’utilisent pas la fonction de perte d’une segmentation supervisée classique. Aucune courbe d’entraînement par époque n’est revendiquée.
- **Backbone des résultats disponibles :** ResNet18 avec `pretrained=false` pour les smoke runs cités. Ces résultats limités ne constituent pas une mesure représentative des performances finales.
- **Protocole de seuil :** seuils sélectionnés sur les scores normaux de validation, puis métriques calculées sur le split de test inchangé. Les résultats ci-dessous proviennent néanmoins de runs de démonstration limités à **4 images de test par catégorie (1 normale, 3 défectueuses)**.

#### Scores disponibles — smoke runs limités, split de test

| Modèle / catégorie | AUROC image | AUROC pixel | AUPRO pixel | Dice segmentation |
| :--- | :---: | :---: | :---: | :---: |
| PaDiM — MPDD `bracket_black` | 1,000 | 0,639 | 0,339 | 0,000 |
| PaDiM — MVTec AD `screw` | 0,000 | 0,361 | 0,240 | 0,000 |
| PatchCore — MPDD `bracket_black` | 1,000 | 0,520 | 0,011 | 0,000 |
| PatchCore — MVTec AD `screw` | 1,000 | 0,459 | 0,073 | 0,000 |

Ces mesures sont reportées pour décrire les artefacts présents, pas comme résultats finaux : l’échantillon de test est minuscule, le backbone est aléatoire et les Dice de segmentation sont nuls. Les AUROC image de 1,0 ne doivent pas être interprétées comme une performance généralisable. Aucun objectif chiffré final n’est fixé avant les benchmarks contrôlés.

### 2.3 Traçabilité MLOps

- Intégration MLflow locale et configuration de tracking présentes dans `src/industrialguard/mlops/`; base et artefacts locaux sous `runs/mlflow/`.
- La base contient actuellement **un run**, dans l’expérience `IndustrialGuard-Phase4-Benchmark`, avec la métrique `synthetic_score = 0,75`. Il s’agit d’une trace synthétique de vérification technique, pas d’un entraînement modèle.
- Les résultats PaDiM/PatchCore et leurs checkpoints sont présents séparément dans `reports/model_development/results/` et `checkpoints/`. Ils ne correspondent pas à des runs MLflow réels enregistrés dans la base actuelle.
- La gestion de reproductibilité, les graines, la rédaction de métadonnées et la protection contre l’écrasement des tentatives sont implémentées. Les artefacts de courbes d’apprentissage ne sont pas applicables aux ajustements PaDiM/PatchCore disponibles.

### 2.4 Services applicatifs

- **API FastAPI :** aucune API ni route `/health` ou `/predict` n’a été trouvée dans le dépôt consulté.
- **Interface utilisateur :** aucune application React/Angular ni interface de glisser-déposer n’a été trouvée.
- Ces éléments sont donc **non réalisés / hors du périmètre actuel**, et non présentés comme opérationnels.

### 2.5 Qualité logicielle et CI

- Workflow GitHub Actions présent : tests CPU, Ruff, Mypy, vérification des dépendances et contrôle informatif des notebooks.
- Vérifications locales disponibles : `pytest` (40 réussis, 1 ignoré), `ruff check` (réussi), `ruff format --check` (44 fichiers déjà formatés) et `mypy src/` (27 fichiers sans erreur).
- Le workflow CI ne comporte pas de service API ou frontend. Aucun Dockerfile ni `docker-compose.yml` n’a été identifié dans le périmètre audité.

## 3. Avancement par composant

Les pourcentages ci-dessous sont des estimations de complétude des travaux du composant, pas des scores de qualité du modèle.

| Composant | Avancement estimé | Éléments prêts | Prochaine action |
| :--- | :---: | :--- | :--- |
| **Données & DVC** | 80 % | Audits MPDD/MVTec, inventaires, vérification de masques, cache et métadonnées DVC. | Rendre les archives disponibles dans l’environnement contrôlé et exécuter le préflight Phase 7. |
| **Modèles & évaluation** | 60 % | PaDiM/PatchCore, interface EfficientAD, checkpoints, métriques, seuils et petits smoke runs. | Exécuter les benchmarks contrôlés avec backbone vérifié et validation indépendante. |
| **Tracking MLOps** | 45 % | Tracking MLflow local et code de reproductibilité; une trace synthétique dans la base. | Enregistrer de vrais runs benchmark et leurs configurations, métriques et artefacts représentatifs. |
| **API** | Non démarré / à décider | Aucun service API détecté. | Confirmer si l’API appartient au périmètre; si oui, spécifier les contrats avant implémentation. |
| **Frontend** | Non démarré / à décider | Aucun frontend détecté. | Confirmer le besoin d’interface après stabilisation de l’inférence. |
| **Docker & CI/CD** | 50 % | CI automatisée pour tests, lint, formatage, types et dépendances. | Ajouter une conteneurisation uniquement si elle est requise pour la livraison. |

## 4. Difficultés, limites et réponses prévues

| Verrou constaté | Impact | Réponse prévue |
| :--- | :--- | :--- |
| Les résultats disponibles sont des smoke runs de 4 images de test par catégorie et utilisent un backbone ResNet18 non pré-entraîné. | Les métriques sont instables et ne permettent pas de comparer sérieusement les modèles ni d’affirmer une performance finale. | Les étiqueter comme smoke runs; lancer le protocole contrôlé avec poids vérifiés, échantillon complet et paramètres consignés. |
| Les benchmarks de Phase 7 exigent un manifeste de validation indépendant et des archives accessibles par variables d’environnement. | Le tuning contrôlé ne peut pas être validé avec les seuls artefacts actuellement disponibles. | Fournir et vérifier les entrées requises, contrôler l’absence de fuite depuis le test officiel, puis exécuter le notebook de préflight. |
| La base MLflow actuelle contient une seule trace synthétique, pas les runs des résultats de modèle publiés dans les rapports. | La traçabilité MLOps des expériences réelles n’est pas encore démontrée dans l’interface MLflow. | Enregistrer les prochaines exécutions benchmark dans MLflow et lier chaque run à ses métriques et artefacts. |
| Le gabarit attend une API, un frontend et Docker, absents du dépôt et non décrits dans le périmètre actuel. | La portée de livraison applicative n’est pas définie. | Faire valider le périmètre par l’équipe/l’encadrant avant d’ajouter ces composants. |

## 5. Travail restant

1. **Validation expérimentale :** réunir les archives et le manifeste indépendant requis; valider leurs catégories, annotations et séparation avec le test officiel.
2. **Benchmarks contrôlés :** reproduire PaDiM avec poids ResNet18 vérifiés; comparer les modèles autorisés selon le protocole Phase 7, avec plusieurs catégories et paramètres fixés.
3. **Suivi des résultats :** enregistrer les runs réels dans MLflow; conserver métriques, configurations, seuils, checkpoints et exemples visuels. Distinguer clairement images d’entrée, masques terrain et cartes prédites.
4. **Analyse :** examiner les faibles scores de segmentation des smoke runs; rapporter les métriques par catégorie et leurs limites au lieu de généraliser les scores d’un échantillon de quatre images.
5. **Livraison :** confirmer si une API, une interface web et Docker sont attendus; implémenter seulement les éléments approuvés, puis répéter la démonstration.

### Planning indicatif sur quatre semaines

Ce planning est une proposition relative; les dates et responsables restent à valider par les trois membres.

| Travail | Semaine +1 | Semaine +2 | Semaine +3 | Semaine +4 | Responsable |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Préflight, manifeste indépendant et accès aux données | ● |  |  |  | Toute l’équipe |
| Benchmarks contrôlés PaDiM/PatchCore | ● | ● |  |  |  |
| Benchmarks additionnels et analyse par catégorie |  | ● | ● |  | Toute l’équipe |
| Runs MLflow réels et artefacts visuels |  | ● | ● |  | Toute l’équipe |
| Décision sur API, frontend et conteneurisation | ● |  |  |  | Toute l’équipe |
| Rapport final et répétition de démonstration |  |  | ● | ● | Toute l’équipe |

### Matrice RACI à compléter

Les noms sont inscrits, mais aucune attribution de responsabilité individuelle n’a été fournie. Les cellules ne préjugent donc pas des rôles.

| Tâche | Gafsi Mohamed Aziz | Rayen Ammar | Selim Ben Fraj |
| :--- | :---: | :---: | :---: |
| Pipeline de données & DVC | |  |  |
| Modèles, benchmarks & évaluation |  | |  |
| Suivi MLflow & reproductibilité |  |  | |
| API / interface (si retenues) |  |  |  |
| CI, livraison & démonstration |  | |  |

## 6. Annexes et preuves disponibles

1. **Audit des datasets :** `reports/dataset_audit/` contient les résumés JSON, inventaires CSV, rapport de validation et graphiques de catégories/résolutions.
2. **Exemples d’images et masques de référence :** [planche d’audit MPDD/MVTec](reports/dataset_audit/image_mask_overlay_examples.png). Elle montre images normales, images défectueuses, masques de vérité terrain et superpositions. **Elle ne contient pas de carte d’anomalie prédite par PaDiM ou PatchCore.**
3. **Résultats et checkpoints :** `reports/model_development/results/` et `checkpoints/` contiennent les JSON de résultats et checkpoints. Le dossier d’essai PaDiM `runs/benchmark/.../attempt-001` et ses cartes de prédiction n’étaient pas présents au moment de la rédaction.
4. **MLflow :** serveur local sur le port 5000; la base observée ne contient qu’un run synthétique. Aucune courbe de perte ni capture d’écran de run d’entraînement réel n’est disponible.
5. **API / interface :** aucune capture d’écran n’est ajoutée, car aucun service ou frontend n’a été trouvé dans le dépôt.

---
