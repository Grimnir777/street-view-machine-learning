import os
import numpy as np
import time
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def print_usage():
    print(
        """
        The street view house number machine learning problem
        
        Vous pouvez grâce à ce programme tester plusieurs algorithmes
        de machine learning sur les données des numéros de maisons

        Usage:
            main.py <mode> <algo_name>
        Options:
            <mode> : mode à utiliser pour le programme parmi : run | comp
            
            Run va lancer l'algorithme choisi avec la meilleur configuration trouvée
            Comp va lancer une comparaison pour le même algorithme en changeant (disponible pour tous les algorithmes sauf gaussian)
            
            <algo_name> : Choix d'algorithme parmi les suivants : gaussian | gaussian_pca | svc | k_neighbors | decision_tree | mlp

            -h --help : Affiche cette aide
        """
    )


def load_data(img_file_name, labels_file_name):
    """
    Récupère des données (format .npy)
    retourne deux arrays X et Y ainsi que leurs tailles respectives
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(dir_path + "\\" + img_file_name)
    y = np.load(dir_path + "\\" + labels_file_name)

    size_x = X.shape[0]
    size_y = y.shape[0]

    if size_x != size_y:
        print(img_file_name,size_x)
        print(labels_file_name,size_y)
        raise ValueError('Set error : image and labels numbers do not match')
    return (X,y,size_x,size_y)

def feature_scaling(X_trn, X_dev):
    """
    Initialise un StandardScaler par rapport au training set
    Puis applique la transformation sur le training set et au test set
    """
    scaler = StandardScaler().fit(X_trn)

    X_trn = scaler.transform(X_trn)
    X_dev = scaler.transform(X_dev)
    return (X_trn, X_dev)

def print_accuracy(algo_name, prediction, y_dev):
    print("Pourcentage de prédiction pour l'algorithme : {}".format(algo_name))
    print("{:2.2f} %".format(accuracy_score(prediction, y_dev) * 100))

def gaussian_without_pca(X_trn, y_trn, X_dev):
    """
    Premier algorithme de prédiction : Classifieur bayésien gaussien
    """
    classifier = GaussianNB().fit(X_trn, y_trn)
    GaussianNB(priors=None)
    return(classifier.predict(X_dev))

def apply_pca(X_trn, X_dev, n_components):
    """
    Applique une analyse en composantes principales (APC ; PCA en anglais)
    """
    pca = IncrementalPCA(n_components= n_components).fit(X_trn)

    X_trn_pca = pca.transform(X_trn)
    X_dev_pca = pca.transform(X_dev)
    return (X_trn_pca, X_dev_pca)

def gaussian_with_pca(X_trn, y_trn, X_dev, n_components):
    """
    Deuxième algorithme de prédiction :
    Classifieur bayésien gaussien avec analyse en composantes principales (APC)
    """
    X_trn_pca, X_dev_pca = apply_pca(X_trn, X_dev, n_components)

    classifier = GaussianNB().fit(X_trn_pca, y_trn)
    GaussianNB(priors=None)

    return(classifier.predict(X_dev_pca))

def svc_with_pca(X_trn, y_trn, X_dev, n_components, gamma):
    """
    Troisième algorithme de prédiction :
    Support Vector Machine
    (l'utilisation de PCA est obligatoire sinon le temps de calcul est trop long)
    """
    X_trn_pca, X_dev_pca = apply_pca(X_trn, X_dev, n_components)
    classifier = svm.SVC(C=1).fit(X_trn_pca, y_trn)
    return classifier.predict(X_dev_pca)

def k_neighbors(X_trn, y_trn, X_dev, n_components, n_neighbors):
    """
    Quatrième algorithme de prédiction :
    K-neighbors
    (l'utilisation de PCA est obligatoire sinon le temps de calcul est trop long)
    """
    X_trn_pca, X_dev_pca = apply_pca(X_trn, X_dev, n_components)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_trn_pca, y_trn)
    return clf.predict(X_dev_pca)

def decision_tree(X_trn, y_trn, X_dev, n_components, max_depth):
    """
    Cinquième algorithme de prédiction :
    Decision Tree
    """
    X_trn_pca, X_dev_pca = apply_pca(X_trn, X_dev, n_components)
    clf = DecisionTreeClassifier(max_depth = max_depth, random_state = 0).fit(X_trn_pca, y_trn)
    return clf.predict(X_dev_pca)


def multi_layer_perceptron(X_trn, y_trn, X_dev, n_components, max_iter):
    """
    Sixième algorithme de prédiction :
    Multi-layer Perceptron classifier
    """
    X_trn_pca, X_dev_pca = apply_pca(X_trn, X_dev, n_components)
    clf = MLPClassifier(alpha=1, max_iter=max_iter).fit(X_trn_pca, y_trn)
    return clf.predict(X_dev_pca)

def print_confusion_matrix(algo_name, prediction, true_data):
    """
    Affiche la matrice de confusion
    """
    cm = confusion_matrix(true_data, prediction)
    sn.set(font_scale=1.4)
    sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(algo_name)
    plt.show()

def run_gaussian():
    start_time = time.time()
    result_dev = gaussian_without_pca(X_trn, y_trn, X_dev)
    print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
    print_accuracy('Gaussien sans PCA', result_dev, y_dev)
    print_confusion_matrix('Gaussien sans PCA',result_dev, y_dev)


## RUNS

def run_gaussian_pca():
    start_time = time.time()
    result_dev = gaussian_with_pca(X_trn, y_trn, X_dev, 50)
    print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
    print_accuracy('Gaussien avec PCA', result_dev, y_dev)
    print_confusion_matrix("Gaussien avec PCA avec nb_components = 50", result_dev, y_dev)

def run_svc():
    # Algo 3 - Support Vector Machine
    # Best value so far : 70.66 seconde(s) 80.50 % with n_components = 50 and gamma = 0.01
    start_time = time.time()
    result_dev = svc_with_pca(X_trn,y_trn, X_dev, 50, 2)
    print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
    print_accuracy("Support Vector Machine avec PCA",result_dev,y_dev)
    print_confusion_matrix("Support Vector Machine avec gamma = 0.01",result_dev, y_dev)

def run_k_neighbors():
    ## Algo 4 - K-neighbors Algo
    # Best value so far : 71% with  n_components = 50 and neighbors = 20 17.01 seconde(s)
    start_time = time.time()
    result_dev = k_neighbors(X_trn,y_trn,X_dev,50,20)
    print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
    print_accuracy("K-neighbors",result_dev,y_dev)
    print_confusion_matrix("K-neighbors avec neighbors = 20",result_dev, y_dev)

def run_decision_tree():
    ## Algo 5 - Decision tree
    # Best value so far : 37.58 % 11.06 seconde(s) n_components 50 ; max_depth 10
    start_time = time.time()
    result_dev = decision_tree(X_trn,y_trn,X_dev,50,10)
    print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
    print_accuracy("Decision tree",result_dev,y_dev)
    print_confusion_matrix("Decision tree avec max_depth = 10 ",result_dev, y_dev)

def run_mlp():
    ## Algo 6 - MLP
    # Best value so far : 46.01 seconde(s) 89.88 % max_iter 1000
    start_time = time.time()
    result_dev = multi_layer_perceptron(X_trn,y_trn,X_dev,50,1000)
    print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
    print_accuracy("Multi Layer Perceptron",result_dev,y_dev)
    print_confusion_matrix("Multi Layer Perceptron avec max_iter = 1000",result_dev, y_dev)

## COMP

def comp_gaussian_pca():
    nb_iteration = 10
    incremental = 10
    n_components = incremental

    indexes = []
    values = []
    for i in range(nb_iteration):
        result_dev = gaussian_with_pca(X_trn, y_trn, X_dev, int(n_components))
        print_accuracy('Gaussien avec PCA',result_dev, y_dev)

        indexes.append(n_components)
        values.append(accuracy_score(result_dev, y_dev))
        n_components += incremental

    plt.plot(indexes, values) 
    plt.show()

def comp_svc():
    nb_iteration = 10
    incremental = 0.02
    gamma = incremental

    indexes = []
    values = []
    for i in range(nb_iteration):
        result_dev = svc_with_pca(X_trn,y_trn, X_dev, 50, gamma)
        print_accuracy('SVC', result_dev, y_dev)
        indexes.append(gamma)
        values.append(accuracy_score(result_dev, y_dev))
        gamma += incremental
    plt.plot(indexes, values)
    plt.xlabel('Gamma')
    plt.ylabel('Précision')
    plt.title('Support Vector Machine')
    plt.show()

def comp_k_neighbors():
    nb_iteration = 10
    incremental = 4
    n_neighbors = incremental

    indexes = []
    values = []
    for i in range(nb_iteration):
        result_dev = k_neighbors(X_trn,y_trn, X_dev, 50, n_neighbors)
        print_accuracy('K-neighbors', result_dev, y_dev)
        indexes.append(n_neighbors)
        values.append(accuracy_score(result_dev, y_dev))
        n_neighbors += incremental
    plt.plot(indexes, values)
    plt.xlabel('Nombre de voisins')
    plt.ylabel('Précision')
    plt.title('K-neighbors')
    plt.show()

def comp_decision_tree():
    nb_iteration = 10
    incremental = 2
    max_depth = incremental

    indexes = []
    values = []
    for i in range(nb_iteration):
        result_dev = decision_tree(X_trn,y_trn, X_dev, 50, max_depth)
        print_accuracy('Decision Tree', result_dev, y_dev)
        indexes.append(max_depth)
        values.append(accuracy_score(result_dev, y_dev))
        max_depth += incremental
    plt.plot(indexes, values)
    plt.xlabel('Profondeur maximale')
    plt.ylabel('Précision')
    plt.title('Decision Tree')
    plt.show()


def comp_mlp():
    nb_iteration = 10
    incremental = 200
    max_iter = incremental

    indexes = []
    values = []
    for i in range(nb_iteration):
        result_dev = multi_layer_perceptron(X_trn,y_trn,X_dev, 50, max_iter)
        print_accuracy('Multi Layer Perceptron', result_dev, y_dev)
        indexes.append(max_iter)
        values.append(accuracy_score(result_dev, y_dev))
        max_iter += incremental
    plt.plot(indexes, values)
    plt.xlabel('Itération maximale')
    plt.ylabel('Précision')
    plt.title('Multi Layer Perceptron')
    plt.show()


## Chargement des données
print('Chargement des données ...')
start_time = time.time()
(X_trn, y_trn, size_trn_x, size_trn_y) = load_data("trn_img.npy","trn_lbl.npy")
(X_dev, y_dev, size_dev_x, size_dev_y) = load_data("dev_img.npy","dev_lbl.npy")

# Standard scaler sur le data set de training et dev
(X_trn,X_dev) = feature_scaling(X_trn, X_dev)

print('Données chargées et configurées')

# run_k_neighbors()

# comp_mlp()
run_svc()



