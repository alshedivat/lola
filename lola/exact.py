"""
LOLA and NL with exact value functions.
"""
import numpy as np
import tensorflow as tf

from . import logger

from .utils import *


class ExactLOLA(object):
    """
    Implements LOLA with exact value functions.
    """
    def __init__(self, scope, num_actions, num_states,
                 gamma=0.96, alpha=1., decay=0.,
                 enable_corrections=True,
                 seed=42):
        self.name = scope
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.enable_corrections = enable_corrections

        with tf.variable_scope(scope):
            self.theta = tf.Variable(tf.random_normal([5, 1]))
            self.ac_probs = tf.nn.sigmoid(self.theta)

            with tf.variable_scope("update"):
                self.theta_opp = tf.placeholder(shape=[5, 1], dtype=tf.float32)
                self.R_1 = tf.placeholder(shape=[4, 1], dtype=tf.float32)
                self.R_2 = tf.placeholder(shape=[4, 1], dtype=tf.float32)
                self._build_update()

        self.rng = np.random.RandomState(seed)

    @property
    def parameters(self):
        params = self.sess.run(self.theta)
        return params

    def setup(self, sess, env_name, payout_mat):
        self.sess = sess
        self.env_name = env_name
        if env_name == 'IPD':
            self.payout_mat_1 = payout_mat.T
            self.payout_mat_2 = payout_mat
        elif env_name == 'IMP':
            self.payout_mat_1 = payout_mat
            self.payout_mat_2 = -payout_mat
        else:
            raise ValueError("Unknown env name: %s" % env_name)

    def act(self, state):
        idx = np.nonzero(state)[0][0]
        prob = self.sess.run(self.ac_probs)[idx]
        ac = 0 if self.rng.rand() < prob else 1
        return ac

    def _build_update(self):
        theta_1 = tf.slice(self.theta, [0, 0], [4, 1])
        theta_2 = tf.slice(self.theta_opp, [0, 0], [4, 1])

        theta_1_0 = tf.slice(self.theta, [4, 0], [1, 1])
        theta_2_0 = tf.slice(self.theta_opp, [4, 0], [1, 1])

        p_1 = tf.nn.sigmoid(theta_1)
        p_2 = tf.nn.sigmoid(theta_2)

        p_1_0 = tf.nn.sigmoid(theta_1_0)
        p_2_0 = tf.nn.sigmoid(theta_2_0)

        p_1_0_v = tf.concat([p_1_0, (1 - p_1_0)], 0)
        p_2_0_v = tf.concat([p_2_0, (1 - p_2_0)], 0)

        s_0 = tf.reshape(tf.matmul(p_1_0_v, tf.transpose(p_2_0_v)), [-1, 1])

        P = tf.concat([
            tf.multiply(p_1, p_2),
            tf.multiply(p_1, 1 - p_2),
            tf.multiply(1 - p_1, p_2),
            tf.multiply(1 - p_1, 1 - p_2)
        ], 1)

        I_m_P = tf.diag([1.0, 1.0, 1.0, 1.0]) - P * self.gamma
        self.v_1 = tf.matmul(tf.matmul(tf.matrix_inverse(I_m_P), self.R_1), s_0,
                             transpose_a=True)
        self.v_2 = tf.matmul(tf.matmul(tf.matrix_inverse(I_m_P), self.R_2), s_0,
                             transpose_a=True)

        grad_v_1_theta_1 = tf.gradients(self.v_1, self.theta)[0]
        grad_v_1_theta_2 = tf.gradients(self.v_1, self.theta_opp)[0]

        theta_size = self.theta.get_shape()[0].value
        if self.enable_corrections:
            grad_v_2_theta_2_theta_1 = tf.concat([
                tf.reshape(
                    tf.gradients(tf.gradients(self.v_2, self.theta)[0][i][0],
                                 self.theta_opp),
                    [1, theta_size])
                for i in range(theta_size)
            ], 0)
            second_order_1 = tf.transpose(
                tf.matmul(tf.reshape(grad_v_1_theta_2, [1, theta_size]),
                          tf.transpose(grad_v_2_theta_2_theta_1))
            )
            grad_v_1_theta_1 += second_order_1

        self.update_theta = self.theta.assign(
            self.theta + self.alpha * grad_v_1_theta_1)

    def update(self, theta_opp):
        # Ensure opponent's theta is represented correctly
        p_opp = np.exp(theta_opp) / (1 + np.exp(theta_opp))
        if self.env_name == 'IPD':
            p_opp[1], p_opp[2] = 1 - p_opp[1], 1 - p_opp[2]
        elif self.env_name == 'IMP':
            p_opp = 1 - p_opp
        p_opp = np.clip(p_opp, 1e-8, 1 - 1e-8)
        theta_opp = np.log(p_opp) - np.log(1 - p_opp)

        # Update parameters
        v1, v2, _ = self.sess.run(
            [self.v_1, self.v_2, self.update_theta],
            feed_dict={
                self.theta_opp: theta_opp,
                self.R_1: np.reshape(self.payout_mat_1, [-1, 1]),
                self.R_2: np.reshape(self.payout_mat_2, [-1, 1]),
            }
        )

        return v1[0][0] * (1 - self.gamma), v2[0][0] * (1 - self.gamma)
