# Street View House Number (SVHN)
Ce problème consiste à appliquer des algorithmes de machine learning sur un set de données (images) de numéros de portes.

Le but de l'exercice va être aussi de trouver les meilleures paramétrisations pour chaque algorithme.
Ci-dessous vous pouvez voir les meilleurs résultats que j'ai réussi à trouver.

Vous retrouverez dans le dossier images les résultats des exécutions de mon programme avec les différents algorithmes.

Plus d'informations sur le projet [ici](http://ufldl.stanford.edu/housenumbers/)


# Algorithmes testés, résultats et temps d'exécutions

| Algorithme                             | Précision | Temps d'exécution |
|----------------------------------------|-----------|-------------------|
| Classifieur bayésien gaussien          | 17.42%    | 1.09 seconde      |
| Classifieur bayésien gaussien avec PCA | 30.72%    | 9.23 secondes     |
| Classifieur K-neighbors                | 70.54%    | 17.01 secondes    |
| Support Vector Machine                 | 85.64%    | 50.72 secondes    |
| Decision tree                          | 44.36%    | 12.18 secondes    |
| Multi Layer Perceptron                 | 89.44%    | 47.31 secondes    |

## Installation

```bash
pip3 install -r requirements/requirements.txt
```

## Usage
```bash
python3 main.py
```

## Mode disponibles

- Le mode 'RUN'

Celui-ci va lancer une seule itération de l'algoritme choisi avec la meilleure configuration.  
Aussi, à la fin de l'exécution la matrice de confusion sera affichée.

- Le mode 'COMPARAISON' 

Il va lancer 10 itérations afin de centrer la meilleur configuration à peu près vers la 5ème itération  
Attention au temps d'exécution qui va approcher les 500 secondes (svm) dans le pire des cas !
A la fin des 10 itérations un graphique du pourcentage en fonction du paramètre sera affiché.
