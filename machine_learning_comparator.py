import time
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from data_manager import DataManager
from algorithms_base import AlgorithmsBase


class MachineLearningComparator:
    def __init__(self):
        print('Chargement des données ...')
        start_time = time.time()
        (self.X_trn, self.y_trn, self.size_trn_x, self.size_trn_y) = DataManager.load_data("trn_img.npy", "trn_lbl.npy")
        (self.X_dev, self.y_dev, self.size_dev_x, self.size_dev_y) = DataManager.load_data("dev_img.npy", "dev_lbl.npy")

        # Standard scaler sur le data set de training et dev
        (self.X_trn, self.X_dev) = DataManager.feature_scaling(self.X_trn, self.X_dev)

        print('Données chargées et configurées')
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))

    @staticmethod
    def print_accuracy(algo_name, prediction, y_dev):
        print("Pourcentage de prédiction pour l'algorithme : {}".format(algo_name))
        print("{:2.2f} %".format(accuracy_score(prediction, y_dev) * 100))

    @staticmethod
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

    def run_gaussian(self):
        # Algo 1 - Gaussian without pca
        start_time = time.time()
        result_dev = AlgorithmsBase.gaussian_without_pca(self.X_trn, self.y_trn, self.X_dev)
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
        self.print_accuracy('Gaussien sans PCA', result_dev, self.y_dev)
        self.print_confusion_matrix('Gaussien sans PCA', result_dev, self.y_dev)

    def run_gaussian_pca(self):
        # Algo 2 - Gaussian with pca
        start_time = time.time()
        result_dev = AlgorithmsBase.gaussian_with_pca(self.X_trn, self.y_trn, self.X_dev, 50)
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
        self.print_accuracy('Gaussien avec PCA', result_dev, self.y_dev)
        self.print_confusion_matrix("Gaussien avec PCA avec nb_components = 50", result_dev, self.y_dev)

    def run_svc(self):
        # Algo 3 - Support Vector Machine
        # Best value so far : 70.66 seconde(s) 80.50 % with n_components = 50 and gamma = 0.01
        start_time = time.time()
        result_dev = AlgorithmsBase.svc_with_pca(self.X_trn, self.y_trn, self.X_dev, 50, 2)
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
        self.print_accuracy("Support Vector Machine avec PCA", result_dev, self.y_dev)
        self.print_confusion_matrix("Support Vector Machine avec gamma = 0.01", result_dev, self.y_dev)

    def run_k_neighbors(self):
        # Algo 4 - K-neighbors Algo
        # Best value so far : 71% with  n_components = 50 and neighbors = 20 17.01 seconde(s)
        start_time = time.time()
        result_dev = AlgorithmsBase.k_neighbors(self.X_trn, self.y_trn, self.X_dev, 50, 20)
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
        self.print_accuracy("K-neighbors", result_dev, self.y_dev)
        self.print_confusion_matrix("K-neighbors avec neighbors = 20", result_dev, self.y_dev)

    def run_decision_tree(self):
        # Algo 5 - Decision tree
        # Best value so far : 37.58 % 11.06 seconde(s) n_components 50 ; max_depth 10
        start_time = time.time()
        result_dev = AlgorithmsBase.decision_tree(self.X_trn, self.y_trn, self.X_dev, 50, 10)
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
        self.print_accuracy("Decision tree", result_dev, self.y_dev)
        self.print_confusion_matrix("Decision tree avec max_depth = 10 ", result_dev, self.y_dev)

    def run_mlp(self):
        # Algo 6 - MLP
        # Best value so far : 46.01 seconde(s) 89.88 % max_iter 1000
        start_time = time.time()
        result_dev = AlgorithmsBase.multi_layer_perceptron(self.X_trn, self.y_trn, self.X_dev, 50, 1000)
        print("Temps d'exécution : {:2.2f} seconde(s)".format(time.time() - start_time))
        self.print_accuracy("Multi Layer Perceptron", result_dev, self.y_dev)
        self.print_confusion_matrix("Multi Layer Perceptron avec max_iter = 1000", result_dev, self.y_dev)

    ## COMP

    def comp_gaussian_pca(self):
        nb_iteration = 10
        incremental = 10
        n_components = incremental

        indexes = []
        values = []
        for i in range(nb_iteration):
            result_dev = AlgorithmsBase.gaussian_with_pca(self.X_trn, self.y_trn, self.X_dev, int(n_components))
            self.print_accuracy('Gaussien avec PCA', result_dev, self.y_dev)

            indexes.append(n_components)
            values.append(accuracy_score(result_dev, self.y_dev))
            n_components += incremental

        plt.plot(indexes, values)
        plt.show()

    def comp_svc(self):
        nb_iteration = 10
        incremental = 0.02
        gamma = incremental

        indexes = []
        values = []
        for i in range(nb_iteration):
            result_dev = AlgorithmsBase.svc_with_pca(self.X_trn, self.y_trn, self.X_dev, 50, gamma)
            self.print_accuracy('SVC', result_dev, self.y_dev)
            indexes.append(gamma)
            values.append(accuracy_score(result_dev, self.y_dev))
            gamma += incremental
        plt.plot(indexes, values)
        plt.xlabel('Gamma')
        plt.ylabel('Précision')
        plt.title('Support Vector Machine')
        plt.show()

    def comp_k_neighbors(self):
        nb_iteration = 10
        incremental = 4
        n_neighbors = incremental

        indexes = []
        values = []
        for i in range(nb_iteration):
            result_dev = AlgorithmsBase.k_neighbors(self.X_trn, self.y_trn, self.X_dev, 50, n_neighbors)
            self.print_accuracy('K-neighbors', result_dev, self.y_dev)
            indexes.append(n_neighbors)
            values.append(accuracy_score(result_dev, self.y_dev))
            n_neighbors += incremental
        plt.plot(indexes, values)
        plt.xlabel('Nombre de voisins')
        plt.ylabel('Précision')
        plt.title('K-neighbors')
        plt.show()

    def comp_decision_tree(self):
        nb_iteration = 10
        incremental = 2
        max_depth = incremental

        indexes = []
        values = []
        for i in range(nb_iteration):
            result_dev = AlgorithmsBase.decision_tree(self.X_trn, self.y_trn, self.X_dev, 50, max_depth)
            self.print_accuracy('Decision Tree', result_dev, self.y_dev)
            indexes.append(max_depth)
            values.append(accuracy_score(result_dev, self.y_dev))
            max_depth += incremental
        plt.plot(indexes, values)
        plt.xlabel('Profondeur maximale')
        plt.ylabel('Précision')
        plt.title('Decision Tree')
        plt.show()

    def comp_mlp(self):
        nb_iteration = 10
        incremental = 200
        max_iter = incremental

        indexes = []
        values = []
        for i in range(nb_iteration):
            result_dev = AlgorithmsBase.multi_layer_perceptron(self.X_trn, self.y_trn, self.X_dev, 50, max_iter)
            self.print_accuracy('Multi Layer Perceptron', result_dev, self.y_dev)
            indexes.append(max_iter)
            values.append(accuracy_score(result_dev, self.y_dev))
            max_iter += incremental
        plt.plot(indexes, values)
        plt.xlabel('Itération maximale')
        plt.ylabel('Précision')
        plt.title('Multi Layer Perceptron')
        plt.show()
