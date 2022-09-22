import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.colors import ListedColormap

class Perceptron(object):
    '''Классификатор на основе персептрона.
    Параметры
    ---------
    eta : float
     Скорость обучения (между 0.1 и 1.0)
    n_iter : int
     Проходы по обучающему набору данных.
    random_state : int
     Начальное значение генератора случайных чисел для инициализации случайными весами.
    Атрибуты
    -------
    w_ : одномерный массив
     Веса после подгонки.
    errors_ : список
     Количество неправильных классификаций (обновлений) в каждой эпохе.
    '''
    def __init__ (self, eta=0.01, n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    
    def fit(self, X, y):
        """ Подгоняет к обучающим данным.
        Параметры
        --------
        X : {подобен массиву}, форма = [n_examples, n_features]
         Обучающие векторы, где n_examples - количество образцов
         и n_features - количество признаков.
        y : подобен массиву, форма = [n_examples]
         Целевые значения.

        Возвращает
        ---------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi)) 
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update !=0.0)
            self.errors_.append(errors)
        return(self)

    def net_input(self, X):
        """Вычисляет общий вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Возвращает метку класса после единичного шага"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


s = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data').replace('\\','/')
# не забываем реплейсить \ на / (винда однако)
print ('URL:', s)
df = pd.read_csv (s,header=None,encoding='utf-8')
#df.tail()
print(df.reset_index(drop=True).tail())
# выбрать ирис щетинистый и ирис разноцветный
y= df.iloc[0:100,4].values
y= np.where(y == 'Iris-setosa', - 1, 1)
#извлечь длину чашелистика и длину лепестка
X = df.iloc[0:100,[0, 2]].values
# вычертить трафик для данных
plt.scatter(X[:50, 0], X[:50, 1], color= 'red', marker='o', label= 'щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color= 'blue', marker='x', label= 'разноцветный')
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпoxи')
plt.ylabel('Количество обновлений')
plt.show ()


#График с областями решения
def plot_decision_regions(X, y, classifier, resolution=0.02):
# на строить генерат ор маркер ов и кар ту цветов
    markers = ('s','x','o','4','4')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # вывести поверхность решения
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # вывести образцы по классам
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('длина чашелистика[см]')
plt.ylabel('длина лепестка[см]')
plt.legend(loc='upper left')
plt.show()


class AdalineGD(object) :
    """Классификатор на основе адаптивноголинейного нейрона.
    Параметры
        eta : float
            Скорость обучения (между 0.0 и 1.0)
        n_iter : int
            Проходы по обучающему набору данных.
        random_state : int
            Начальное значение генератора случайных чисел для инициализации случайными весами.
    Атрибуты
        w_ : одномерный массив
            Веса после подгонки.
        cost_ : список
            Значение функции издержек на основе суммы квадратов в каждой эпохе.
    """
    def __init__ (self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        """Подгоняет к обучающим данным .
        Параметры
        X : {подобен массиву}, форма = [n_examples , n_features]
            Обучающие векторы , где n_examples - количество образцов, 
            n_features - количество признаков .
        y : подобен массиву , форма = [n_examples]
            Целевые значения .
        Возвращает
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range (self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum () / 2.0
            self.cost_.append(cost)
        return self

    def net_input (self, X):
        """ Вычисляет общий вход """
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation (self, X):
        """ Вычисляет линейную активацию """
        return X

    def predict(self, X) :
        """ Возвращает метку класса после единичного шага """
        return np.where(self.activation(self.net_input(X)) > 0.0, 1, -1)


# График зависимости издержек от количества эпох для 2х скоростей обучения

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0] .set_xlabel ('Эпохи')
ax[0].set_ylabel('log(Cyммa квадратичных ошибок)')
ax[0].set_title('Adaline - скорость обучения 0.01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Cyммa квадратичных ошибок')
ax[1].set_title('Adaline - скорость обучения 0.0001')
plt.show()

# СТАНДАРТИЗАЦИЯ
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada_gd = AdalineGD(n_iter=15, eta=0.01)
ada_gd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - градиентный спуск')
plt.xlabel('дпина чашелистика [стандартизированная]')
plt.ylabel('длина лепестка [стандарти зированная]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_gd.cost_) + 1),ada_gd.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')
plt.tight_layout()
plt.show ()

#Стахостический градиентный спуск
class AdalineSGD(object):
    """Классификатор на основе адаптивного линейного нейрона .
    Параметры
    eta : float
        Скорость обучения (между 0.0 и 1.0)
    n_iter : int
        Проходы по обучающему набору данных.
    shuffle : bool (по умолчанию : True)
        Если True , тогда та совать обучающие данные воизбежание циклов.
    random_state : int
        Начальное значение генератора случайных чисел
        для инициализации случайными весами .
    Атрибуты
    w_ : одномерный массив
        Веса после подгонки.
    cost_ : список
        Значение функции издержек на основе суммы квадратов ,
        усредненное по всем обучающим образцам в каждой эпохе .
    """
    def __init__ (self, eta=0.01, n_iter=10, shuffle=True, random_state=None) :
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    def fit(self, X , y) :
        """Подгоняет к обучающим да нным .
        Параметры
        X : (подобен массиву } , форма = [n_ex amples , n_features]
            Обучающие векторы, где n_ex amples - количе ство образцов ,
            n_ features - количество признаков .
        y : подобен массиву , форма = [n_examples]
        Целевые значения .
        Возвращает
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range (self.n_iter):
            if self.shuffle:
                X, y= self._shuffle(X, y)
            cost = []
            for xi, target in zip (X, y) :
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append ( avg_cost)
        return self

    def partial_fit(self, X, y):
        """Подгоняет к обучающим данным без повторной инициализации весов """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip (X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X , y)
        return self

    def _shuffle (self, X, y) :
        """ Тасует обучающие данные """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights (self, m) :
        """Инициализирует веса небольшими случа йными числами """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights (self, xi, target) :
        """ Применяет правило обучения Adaline для обновления весов """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """ Вычисляет общий вход """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Вычисляет линейную активацию """
        return X
    
    def predict(self, X) :
        """Возвращает метку класса после единичного шага """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - стохастический градиентный спуск')
plt.xlabel('длина чашелистика [ стандартизированная ]')
plt.ylabel('длина лепестка [ стандартизированная ]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Усредненные издержки')
plt.tight_layout()
plt.show()