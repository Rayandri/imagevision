# Compression d'Images pour Moteur de Recherche

Projet EPITA - Compression Avancée d'Images

Objectif : Analyser l'impact de différentes méthodes de compression sur un moteur de recherche d'images basé sur la similarité visuelle.

## Critères d'évaluation

- 2 méthodes classiques (PNG sans perte, JPEG avec perte)
- 2 méthodes modernes (Haar, DCT)
- Mesures : ratio de compression, PSNR, SSIM, temps
- Impact sur la recherche d'images (précision, vitesse)
- Comparaison objective et code modulaire

## Méthodes de compression

- PNG (sans perte)
- JPEG Q85 (avec perte)
- Haar Wavelets
- DCT (Discrete Cosine Transform)

## Structure du projet

libs_vision/
    __init__.py
    compression_engine.py
    search_engine.py
    analysis_tools.py
Compression_Analysis_Complete.ipynb  (notebook principal)
data/ (dataset Kaggle)
legacy/ (ancien code)

## Utilisation

1. Lancer le notebook principal :

    jupyter notebook Compression_Analysis_Complete.ipynb

2. Exécuter toutes les cellules pour générer l'analyse complète et le rapport.

Ou bien, en Python :

    from libs_vision import CompressionImpactAnalyzer
    analyzer = CompressionImpactAnalyzer(data_path="data")
    results = analyzer.run_complete_analysis(max_images=20, num_queries=10)
    report = analyzer.generate_summary_report(results)
    print(report)

## Métriques évaluées

- Ratio de compression
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Temps de compression/décompression
- Précision du moteur de recherche (avant/après compression)
- Perte de précision
- Vitesse de recherche

## Méthodologie

- Indexation des features CNN (11520 dimensions)
- Compression/décompression des images
- Extraction de features sur images reconstituées
- Évaluation de la recherche par similarité cosinus
- Comparaison quantitative de chaque méthode

## Résultats principaux

**Meilleure méthode identifiée : PNG_Lossless**
- Perte de précision minimale : 0.400 (précision maintenue : 0.500)
- Compression sans perte avec ratio modéré (0.27x)
- Temps de traitement acceptable (0.077s)

**Conclusions de l'analyse :**
- Les moteurs de recherche par similarité sont robustes à la compression modérée
- Trade-off favorable entre stockage et qualité de recherche
- Les méthodes classiques restent compétitives face aux approches modernes
- PNG sans perte optimal pour préserver la qualité de recherche

## Résultats détaillés par méthode

### PNG_Lossless
- **Ratio compression :** 0.27x
- **PSNR :** ∞ dB (sans perte)
- **SSIM :** 1.000
- **Temps compression :** 0.077s
- **Précision recherche :** 0.500 (perte : 0.400)

### JPEG_Q85  
- **Ratio compression :** 1.45x
- **PSNR :** 39.8 dB
- **SSIM :** 0.998
- **Temps compression :** 0.001s
- **Précision recherche :** 0.400 (perte : 0.500)

### DCT_Q50
- **Ratio compression :** 0.05x
- **PSNR :** 31.6 dB
- **SSIM :** 0.989  
- **Temps compression :** 4.825s
- **Précision recherche :** 0.500 (perte : 0.400)

### Haar_T5.0
- **Ratio compression :** 0.03x
- **PSNR :** 11.2 dB
- **SSIM :** 0.297
- **Temps compression :** 0.068s
- **Précision recherche :** 0.300 (perte : 0.600)

*Baseline précision : 0.900 | Échantillon : 15 images | 10 requêtes*

## Fichiers générés

- compression_results_YYYYMMDD_HHMMSS.csv : métriques de compression
- search_impact_YYYYMMDD_HHMMSS.csv : impact sur la recherche
- rapport_final_compression.txt : synthèse et conclusions

## Dépendances

    pip install pandas numpy pillow pathlib

Aucune dépendance lourde. Code compatible Linux/WSL2.

## Détails techniques

- Dataset : Kaggle Image Search Engine (~3000 images, features CNN)
- Recherche : similarité cosinus
- 4 méthodes de compression conformes à l'énoncé

## Structure du code

- 3 modules principaux, responsabilités séparées
- Numpy/Pandas/Pillow uniquement
- Docstrings pour les fonctions complexes
- Processus reproductible

## Exécution rapide

    jupyter notebook Compression_Analysis_Complete.ipynb
    # Exécuter toutes les cellules
