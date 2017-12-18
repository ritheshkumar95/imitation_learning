import numpy as np
import tensorflow as tf


def StandardGaussianNLL(x, means, sigma=1):
    exponent = tf.pow(x - means, 2) / sigma**2
    constant = np.log(2 * np.pi * sigma**2)
    loglikelihood = -0.5 * (exponent + constant)
    return -loglikelihood


def GaussianNLL(x, means, logvars):
    exponent = tf.pow(x - means, 2)
    exponent = exponent / tf.exp(logvars)
    constant = np.log(2 * np.pi) + logvars

    loglikelihood = -0.5 * (exponent + constant)
    return -loglikelihood


def GaussianMixtureNLL(x, means, logvars, weights):
    exponent = tf.pow(x[:, None] - means, 2)
    exponent = exponent / tf.exp(logvars)
    constant = np.log(2 * np.pi) + logvars

    likelihood = -0.5 * (exponent + constant)
    likelihood += tf.nn.log_softmax(weights, -1)

    def logsumexp(x, dim=-1):
        a = tf.reduce_max(x, dim, keep_dims=True)
        return a + tf.log(tf.reduce_sum(tf.exp(x - a), dim, keep_dims=True))

    return -logsumexp(likelihood)
