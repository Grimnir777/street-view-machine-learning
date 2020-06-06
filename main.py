# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError


style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})

print(
    """
    The street view house number machine learning problem
    
    Vous pouvez grâce à ce programme tester plusieurs algorithmes
    de machine learning sur les données des numéros de maisons

    Le mode RUN va lancer l'algorithme choisi avec la meilleur configuration trouvée
    Le mode COMPARAISON va lancer une comparaison pour le même algorithme en changeant ses paramètres (disponible pour tous les algorithmes sauf gaussian)
            

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
        'name': 'algo',
        'message': 'POur quel algorithme ?',
        'choices': ['Gaussian', 'Gaussian PCA','Support Vector Machine','K-neighbors','Decision tree','Multi Layer Perceptron'],
        'filter': lambda val: val.lower()
    }
]
answers = prompt(questions, style=style)
# if(answers.mode == 'comparaison' and answers[algo] == 'gaussian'):
#     print("L'algorithme gaussian ne peut pas être lancé en mode comparaison")
# else:
#     print('bite')
pprint(answers)