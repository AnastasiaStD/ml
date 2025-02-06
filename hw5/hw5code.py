import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    if len(np.unique(feature_vector)) == 1:
        return [], [], -np.inf, -np.inf

    # как обычно строим джини, отсортили по возрастанию, так как суммировать будем по мере увеличение
    # затем настроили трешхолды - пороги, находящиеся посередине между каждым парой уникальных значений

    sorted_ = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_]
    thholds = (feature_vector[:-1] + feature_vector[1:]) / 2

    n = len(target_vector)
    left, right = np.array(range(1, n)), np.array(range(n-1, 0, -1))

    #кум суммы фичей и таргета - считаем кумулятивные суммы, удаляем последний элемент и делим на сумму для получения долей
    p1_left = np.cumsum(target_vector[sorted_])[:-1] / left
    p0_left = 1 - p1_left
    
    p1_right = (np.cumsum(target_vector[sorted_][::-1])[:-1] / left)[::-1]
    p0_right = 1 - p1_right

    Hl = 1 - p1_left**2 - p0_left**2
    Hr = 1 - p1_right**2 - p0_right**2

    g = - left/n* Hl - right/n * Hr

    g = g[(feature_vector[:-1] != thholds)]
    thholds = thholds[(feature_vector[:-1] != thholds)]


    best_index = np.argmax(g)
    thholds_best = thholds[best_index]
    g_best = np.max(g)

    return thholds, g, thholds_best, g_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return


        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): #
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0 
                    ratio[key] = current_click / current_count #
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) #
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) #
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]): #
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        if self._min_samples_leaf is not None and (len(sub_X[split]) < self._min_samples_leaf or len(sub_X[np.logical_not(split)]) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]
        feature_value = x[feature_split]
    
        if self._feature_types[feature_split] == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError
        
    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self):
        return {"max_depth": self._max_depth, "min_samples_split" : self._min_samples_split, "min_samples_leaf" : self._min_samples_leaf}


class LinearRegressionTree():
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_quantiles=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_quantiles = n_quantiles
        self.tree = None


    def find_best_split(self, X, y):

        best_split = 0
        min_loss = float('inf')
        
        m, n = X.shape

        for fid in range(n):
            unique_values = np.unique(X[:, fid])

            if len(unique_values) <= 1:
                continue

            thrholds = np.quantile(unique_values, q=np.linspace(0, 1, self.n_quantiles + 2)[1:-1])

            for t in thrholds:
                left_mask = X[:, fid] <= t
                right_mask = X[:, fid] > t

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]

                left_model = LinearRegression().fit(X_left, y_left)
                y_left_pred = left_model.predict(X_left)
                loss_left = mean_squared_error(y_left, y_left_pred)

                right_model = LinearRegression().fit(X_right, y_right)
                y_right_pred = right_model.predict(X_right)
                loss_right = mean_squared_error(y_right, y_right_pred)

                Q = (len(y_left) / m) * loss_left + (len(y_right) / m) * loss_right

                if Q < min_loss:
                    min_loss = Q
                    best_split = {
                        'feature_index': fid,
                        'threshold': t,
                        'left_loss': loss_left,
                        'right_loss': loss_right,
                        'left_model': left_model,
                        'right_model': right_model
                    }
        
        return best_split

    def _fit_node(self, sub_X, sub_y, node, current_depth=0):

        if current_depth >= self.max_depth or len(sub_y) < self.min_samples_split:
            node["type"] = "terminal"
            model = LinearRegression().fit(sub_X, sub_y)
            node["model"] = model
            return

        best_split = self.find_best_split(sub_X, sub_y)
        if best_split is None:
            node["type"] = "terminal"
            model = LinearRegression().fit(sub_X, sub_y)
            node["model"] = model
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_split['feature_index']
        node["threshold"] = best_split['threshold']
        node["left_model"] = best_split['left_model']
        node["right_model"] = best_split['right_model']

        left = sub_X[:, best_split['feature_index']] <= best_split['threshold']
        right = sub_X[:, best_split['feature_index']] > best_split['threshold']

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[left], sub_y[left], node["left_child"], current_depth + 1)
        self._fit_node(sub_X[right], sub_y[right], node["right_child"], current_depth + 1)


    def _predict_node(self, x, node):

        while node["type"] == "nonterminal":
            feature = node["feature_split"]
            if x[feature] <= node["threshold"]:
                node = node["left_child"]
            else:
                node = node["right_child"]
        return node["model"].predict([x])[0] if node["type"] == "terminal" else None

    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'n_quantiles': self.n_quantiles
        }
    
    def set_params(self, **params):
    
        for param, value in params.items():
            if param == 'max_depth':
                self.max_depth = value
            elif param == 'min_samples_split':
                self.min_samples_split = value
            elif param == 'min_samples_leaf':
                self.min_samples_leaf = value
            elif param == 'n_quantiles':
                self.n_quantiles = value
        return self


    def fit(self, X, y):
        self.tree = {}
        self._fit_node(X, y, self.tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self.tree) for x in X])