# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from PyInquirer import style_from_dict, Token, prompt
from machine_learning_comparator import MachineLearningComparator

style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})

print(
    """
    The street view house number machine learning problem
    
    Vous pouvez grâce à ce programme tester plusieurs algorithmes
    de machine learning sur les données des numéros de maisons

    * Le mode RUN va lancer l'algorithme choisi avec la meilleur configuration trouvée
    
    * Le mode COMPARAISON va lancer une comparaison pour le même algorithme en changeant ses paramètres 
    Ce mode est disponible pour tous les algorithmes sauf gaussian et exécute 10 itérations à la suite
    
    
    Temps d'exécution par itération et par algorithme (à multiplier par 10 pour le mode comparaison) :
    Gaussian :                  1.09 seconde
    Gaussian PCA :              9.23 secondes
    K-neighbors :               17.01 secondes
    Support Vector Machine :    70.66 secondes
    Decision tree :             11.06 secondes
    Multi Layer Perceptron :    46.01 secondes

    """
    )

questions = [
    {
        'type': 'list',
        'name': 'mode',
        'message': 'Quel mode voulez vous tester ?',
        'choices': ['RUN', 'COMPARAISON'],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'list',
        'name': 'algorithm',
        'message': 'Pour quel algorithme ?',
        'choices': ['Gaussian',
                    'Gaussian PCA',
                    'K neighbors',
                    'Support Vector Machine',
                    'Decision tree',
                    'Multi Layer Perceptron'],
        'filter': lambda val: val.lower()
    }
]

machine_learning_comparator = MachineLearningComparator()

run_algorithm = {
    "gaussian": machine_learning_comparator.run_gaussian,
    "gaussian_pca": machine_learning_comparator.run_gaussian_pca,
    "k_neighbors": machine_learning_comparator.run_k_neighbors,
    "support_vector_machine": machine_learning_comparator.run_svc,
    "decision_tree": machine_learning_comparator.run_decision_tree,
    "multi_layer_perceptron": machine_learning_comparator.run_mlp
}

comp_algorithm = {
    "gaussian_pca": machine_learning_comparator.comp_gaussian_pca,
    "k_neighbors": machine_learning_comparator.comp_k_neighbors,
    "svc": machine_learning_comparator.comp_svc,
    "decision_tree": machine_learning_comparator.comp_decision_tree,
    "multi_layer_perceptron": machine_learning_comparator.comp_mlp
}

answers = prompt(questions, style=style)
mode = answers["mode"].lower()
algorithm = answers["algorithm"].lower().replace(" ", "_")


if mode == "run":
    try:
        run_algorithm[algorithm]()
    except KeyError:
        print("L'algorithme choisi ne correspond pas à la liste des algorithmes possibles :")
        [print(key) for key, value in run_algorithm.items()]
else:
    try:
        comp_algorithm[algorithm]()
    except KeyError:
        print("L'algorithme choisi ne correspond pas à la liste des algorithmes possibles :")
        [print(key) for key, value in comp_algorithm.items()]
