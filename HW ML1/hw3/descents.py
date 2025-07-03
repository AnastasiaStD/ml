from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p

class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        if self.loss_function is LossFunction.MSE:
            return (-2 / len(y))*x.T.dot(y - self.predict(x))
        
        elif self.loss_function is LossFunction.LogCosh:
            l = x.shape[0]
            gradient = (-1/l) * x.T @ np.tanh(y - (x @ self.w))
            return gradient

        elif self.loss_function is LossFunction.MAE:
            prediction = x @ self.w
            gradient = (1/len(y)) *np.where(prediction > y, x, -x)
            return gradient
        # наивная реализация
        elif self == LossFunction.Huber:
            gradient = np.zeros_like(self.w)
            diff = y - self.predict(x)
            delta = 1.0
            
            for i in range(len()):
                if np.abs(diff[i]) <= delta:
                    gradient += (diff[i] * x[i]) / len(y)  # (1/len(y)) * x[i] * (y[i] - prediction[i]) 
                else:
                    gradient += (delta * np.sign(diff[i]) * x[i]) / len(y)  # (1/len(y)) * delta * sign(y[i] - prediction[i]) * x[i]
            gradient = gradient*(1/len(y))
            return gradient

        
        else:
            raise NotImplementedError('для такой функции ошибок не придумали еще градиент.')
        
    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        if self.loss_function == LossFunction.MSE:
            return np.mean((y - self.predict(x)) ** 2)
        elif self.loss_function == LossFunction.MAE:
            return np.mean(np.abs(y - self.predict(x)))
        elif self.loss_function == LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(y - self.predict(x))))
        elif self.loss_function == LossFunction.Huber:
            delta = 1.0 
            diff = y - self.predict(x)
            return np.mean(np.where(np.abs(diff) <= delta,
                                    0.5 * diff ** 2,
                                    delta * (np.abs(diff) - 0.5 * delta)))
        else:
            raise NotImplementedError('для такой функции ошибок не придумали еще градиент.')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w

class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        lr = self.lr()
        wdiff = -lr*gradient
        self.w += wdiff
        return wdiff

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return super().calc_gradient(x, y)
  
class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_indices = np.random.randint(0, x.shape[0], size=self.batch_size)
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        return super().calc_gradient(x_batch, y_batch)

class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        lr = self.lr()
        self.h = self.alpha * self.h + lr * gradient
        self.w -= self.h
        return -self.h

class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.iteration +=1
        self.v = self.beta_2 * self.v + (1 - self.beta_2)* (gradient ** 2)
        self.m = self.beta_1 * self.m + (1 - self.beta_1)* gradient
        
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)

        wdiff = -(self.lr()) * m_hat / (np.sqrt(v_hat) + self.eps)
        self.w += wdiff
        return wdiff

class AdaMax(VanillaGradientDescent):
    """
    AdaMax optimization algorithm class.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)

        self.m: np.ndarray = np.zeros(dimension)
        self.u: np.ndarray = np.zeros(dimension)

        self.epsilon = 1e-8

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iterations = 0


    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights using the AdaMax algorithm.
        
        :param gradient: gradient computed from the loss function.
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        lr = self.lr()
       
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.u =  np.maximum(self.beta_2 * self.u, np.abs(gradient))

        m_hat = self.m / (1 - self.beta_1 ** self.lr.iteration)


        w_diff = -lr * m_hat / (self.u + self.epsilon)
        self.w += w_diff
        return w_diff

class Nadam(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8
        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.iteration: int = 0
        self.m_hat = 0
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        
        self.iteration += 1
        step_k = self.lr()
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        m_hat_prev = self.m_hat if self.iteration > 1 else 0

        self.m_hat = self.m / (1 - (self.beta_1 ** self.iteration))
        v_hat = self.v / (1 - (self.beta_2 ** self.iteration))
        wdiff = (-step_k * (self.beta_1 * m_hat_prev + ((1 - self.beta_1) * gradient)))/((v_hat ** 0.5) + self.eps) 
        self.w += wdiff
        return wdiff

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)
        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient = self.w.copy()
        l2_gradient[-1] = 0
        return super().calc_gradient(x, y) + l2_gradient*self.mu

class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """

class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """

class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """

class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class NadamReg(BaseDescentReg, Nadam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AdaMaxReg(BaseDescentReg, Nadam):
    """
    Adaptive gradient algorithm with regularization class
    """

def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'nadam': Nadam if not regularized else AdamReg,
        'adamax': AdaMax if not regularized else AdaMaxReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
