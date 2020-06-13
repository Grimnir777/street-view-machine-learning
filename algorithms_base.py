from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn import svm


class AlgorithmsBase:
    @staticmethod
    def gaussian_without_pca(X_trn, y_trn, X_dev):
        """
        Premier algorithme de prédiction : Classifieur bayésien gaussien
        """
        classifier = GaussianNB().fit(X_trn, y_trn)
        GaussianNB(priors=None)
        return(classifier.predict(X_dev))

    @staticmethod
    def apply_pca(X_trn, X_dev, n_components):
        """
        Applique une analyse en composantes principales (APC ; PCA en anglais)
        """
        pca = IncrementalPCA(n_components= n_components).fit(X_trn)

        X_trn_pca = pca.transform(X_trn)
        X_dev_pca = pca.transform(X_dev)
        return (X_trn_pca, X_dev_pca)

    @staticmethod
    def gaussian_with_pca(X_trn, y_trn, X_dev, n_components):
        """
        Deuxième algorithme de prédiction :
        Classifieur bayésien gaussien avec analyse en composantes principales (APC)
        """
        X_trn_pca, X_dev_pca = AlgorithmsBase.apply_pca(X_trn, X_dev, n_components)

        classifier = GaussianNB().fit(X_trn_pca, y_trn)
        GaussianNB(priors=None)

        return(classifier.predict(X_dev_pca))

    @staticmethod
    def svc_with_pca(X_trn, y_trn, X_dev, n_components, gamma):
        """
        Troisième algorithme de prédiction :
        Support Vector Machine
        (l'utilisation de PCA est obligatoire sinon le temps de calcul est trop long)
        """
        X_trn_pca, X_dev_pca = AlgorithmsBase.apply_pca(X_trn, X_dev, n_components)
        classifier = svm.SVC(C=1).fit(X_trn_pca, y_trn)
        return classifier.predict(X_dev_pca)

    @staticmethod
    def k_neighbors(X_trn, y_trn, X_dev, n_components, n_neighbors):
        """
        Quatrième algorithme de prédiction :
        K-neighbors
        (l'utilisation de PCA est obligatoire sinon le temps de calcul est trop long)
        """
        X_trn_pca, X_dev_pca = AlgorithmsBase.apply_pca(X_trn, X_dev, n_components)

        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_trn_pca, y_trn)
        return clf.predict(X_dev_pca)

    @staticmethod
    def decision_tree(X_trn, y_trn, X_dev, n_components, max_depth):
        """
        Cinquième algorithme de prédiction :
        Decision Tree
        """
        X_trn_pca, X_dev_pca = AlgorithmsBase.apply_pca(X_trn, X_dev, n_components)
        clf = DecisionTreeClassifier(max_depth = max_depth, random_state = 0).fit(X_trn_pca, y_trn)
        return clf.predict(X_dev_pca)

    @staticmethod
    def multi_layer_perceptron(X_trn, y_trn, X_dev, n_components, max_iter):
        """
        Sixième algorithme de prédiction :
        Multi-layer Perceptron classifier
        """
        X_trn_pca, X_dev_pca = AlgorithmsBase.apply_pca(X_trn, X_dev, n_components)
        clf = MLPClassifier(alpha=1, max_iter=max_iter).fit(X_trn_pca, y_trn)
        return clf.predict(X_dev_pca)