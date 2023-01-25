#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:37:28 2022

@author: jamie
"""

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_recall_curve
import copy


def model(W, H, mu, b1, b2):
    return tf.linalg.matmul(W, H) + mu + b1 + b2


def calculate_biases(X, C):
    X_new = copy.deepcopy(X)
    if type(C) is not int:
        X_new[C == 0] = np.nan
    mu = np.nanmean(X_new)
    
    X_new[np.all(np.isnan(X_new), axis=1)] = mu
    muw = np.expand_dims(np.nanmean(X_new, axis=1), axis=1)
    muh = np.expand_dims(np.nanmean(X_new, axis=0), axis=0)

    mu = tf.Variable(mu, dtype=tf.dtypes.float32)
    bw = tf.Variable(muw - mu, dtype=tf.dtypes.float32)
    bh = tf.Variable(muh - mu, dtype=tf.dtypes.float32)
    return mu, bw, bh


def wmse(X_true, X_pred, C=1): 
    C = tf.constant(C, dtype=tf.dtypes.float32)
    se = tf.math.multiply(C, tf.pow(X_true - X_pred, 2))  
    non_zero = tf.cast(se != 0, dtype=tf.dtypes.float32)  # summing this gives the number of non-zero elements
    return tf.reduce_sum(se) / tf.reduce_sum(non_zero)


def obj_fun(X_true, W, H, C, mu, b1, b2, lam):
    X_pred = model(W, H, mu, b1, b2)
    reg = lam * (tf.reduce_mean(tf.pow(W, 2)) +
                 tf.reduce_mean(tf.pow(H, 2)) + 
                 tf.reduce_mean(tf.pow(b1, 2)) + 
                 tf.reduce_mean(tf.pow(b2, 2)))
    return wmse(X_true, X_pred, C=C) + reg


def optimize_W(X, W, H, C, mu, b1, b2, lam, optimizer):
    with tf.GradientTape() as tape:
        loss = obj_fun(X, W, H, C, mu, b1, b2, lam)

    gradients = tape.gradient(loss, [W])

    optimizer.apply_gradients(zip(gradients, [W]))


def optimize_H(X, W, H, C, mu, b1, b2, lam, optimizer):
    with tf.GradientTape() as tape:
        loss = obj_fun(X, W, H, C, mu, b1, b2, lam)

    gradients = tape.gradient(loss, [H])

    optimizer.apply_gradients(zip(gradients, [H]))

def optimize_b1(X, W, H, C, mu, b1, b2, lam, optimizer):
    with tf.GradientTape() as tape:
        loss = obj_fun(X, W, H, C, mu, b1, b2, lam)

    gradients = tape.gradient(loss, [b1])

    optimizer.apply_gradients(zip(gradients, [b1]))

def optimize_b2(X, W, H, C, mu, b1, b2, lam, optimizer):
    with tf.GradientTape() as tape:
        loss = obj_fun(X, W, H, C, mu, b1, b2, lam)

    gradients = tape.gradient(loss, [b2])

    optimizer.apply_gradients(zip(gradients, [b2]))


def optimization_step(X, W, H, C, mu, b1, b2, lam, optimizer):
    optimize_W(X, W, H, C, mu, b1, b2, lam, optimizer)
    optimize_b1(X, W, H, C, mu, b1, b2, lam, optimizer)
    optimize_H(X, W, H, C, mu, b1, b2, lam, optimizer)
    optimize_b2(X, W, H, C, mu, b1, b2, lam, optimizer)




def optimize(X, W, H, C, mu, b1, b2, lam, optimizer, tol, max_iter,
             partial=False):
    step = 0

    X_tf = tf.constant(X, dtype=tf.dtypes.float32)

    loss = obj_fun(X_tf, W, H, C, mu, b1, b2, lam)

    while loss > tol:

        if partial:

            optimize_W(X_tf, W, H, C, mu, b1, b2, lam, optimizer)
            optimize_b1(X_tf, W, H, C, mu, b1, b2, lam, optimizer)

        else:

            optimization_step(X_tf, W, H, C, mu, b1, b2, lam, optimizer)

        loss = obj_fun(X_tf, W, H, C, mu, b1, b2, lam)

        step = step + 1

        if step % 50 == 0:
            print("epoch: %i, loss: %f" % (step, loss))

        if step == max_iter:
            print("Increase max_iter: unable to meet convergence criteria")
            break


class MatrixFactorization(BaseEstimator):

    def __init__(self,
                 latent_dim=5,
                 lam=0.0,
                 tol=0.0001,
                 max_iter=500,
                 learning_rate=0.1):
        self.latent_dim = latent_dim
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        self.initializer = keras.initializers.RandomUniform(minval=-0.01,
                                                            maxval=0.01,
                                                            seed=None)

        self.X_shape = None
        self.W = None
        self.W_new = None
        self.H = None
        self.mu = None
        self.b1 = None
        self.b2 = None

    def fit_transform(self, X, C=1):
        self.X_shape = np.shape(X)

        self.W = tf.Variable(self.initializer(shape=[self.X_shape[0],
                                                     self.latent_dim],
                                              dtype=tf.dtypes.float32),
                                            #   constraint=lambda z: tf.clip_by_value(z, 0, 1000),
                                              trainable=True)

        self.H = tf.Variable(self.initializer(shape=[self.latent_dim,
                                                     self.X_shape[1]],
                                              dtype=tf.dtypes.float32),
                                            #   constraint=lambda z: tf.clip_by_value(z, 0, 1000),
                                              trainable=True)

        self.mu, self.b1, self.b2 = calculate_biases(X, C)

        optimize(X, self.W, self.H, C, self.mu, self.b1, self.b2,
                 self.lam, self.optimizer, self.tol, self.max_iter)

        return self

    def partial_fit_transform(self, X, C=1):

        H_fixed = tf.constant(self.H)

        self.X_shape = np.shape(X)

        self.W_new = tf.Variable(self.initializer(shape=[self.X_shape[0],
                                                     self.latent_dim],
                                              dtype=tf.dtypes.float32),
                                            #   constraint=lambda z: tf.clip_by_value(z, 0, 1000),
                                              trainable=True)

        _, self.b1_new, _ = calculate_biases(X, C)

        optimize(X, self.W_new, H_fixed, C, self.mu, self.b1_new, self.b2,
                 self.lam, self.optimizer, self.tol, self.max_iter,
                 partial=True)

        return self

    def apply_transform(self, test=True):
        if test:
            W = self.W_new
            b1 = self.b1_new
        else:
            W = self.W
            b1 = self.b1

        X_new = model(W, self.H, self.mu, b1, self.b2)
        X_new = np.clip(X_new, a_min=0.0, a_max=1.0)
        return X_new
