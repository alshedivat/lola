"""
Trains LOLA on IPD or MatchingPennies with exact value functions.

Note: Interfaces are a little different form the code that estimates values,
      hence moved into a separate module.
"""
import numpy as np
import tensorflow as tf

from . import logger

from .utils import *


class Qnetwork:
    """
    Q-network that is either a look-up table or an MLP with 1 hidden layer.
    """
    def __init__(self, myScope, num_hidden, simple_net=True):
        with tf.variable_scope(myScope):
            self.input_place = tf.placeholder(shape=[5],dtype=tf.int32)
            if simple_net:
                self.p_act = tf.Variable(tf.random_normal([5, 1]))
            else:
                act = tf.nn.tanh(
                    layers.fully_connected(
                        tf.one_hot(self.input_place, 5, dtype=tf.float32),
                        num_outputs=num_hidden, activation_fn=None
                    )
                )
                self.p_act = layers.fully_connected(
                    act, num_outputs=1, activation_fn=None
                )
        self.parameters = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=myScope):
            self.parameters.append(i)   # i.name if you want just a name
        self.setparams = SetFromFlat(self.parameters)
        self.getparams = GetFlat(self.parameters)


def update(mainQN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainQN[0].setparams(
        mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    update_theta_2 = mainQN[1].setparams(
        mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def corrections_func(mainQN, corrections, gamma, pseudo, reg):
    mainQN[0].lr_correction = tf.placeholder(shape=[1],dtype=tf.float32)
    mainQN[1].lr_correction = tf.placeholder(shape=[1],dtype=tf.float32)

    theta_1_all = mainQN[0].p_act
    theta_2_all = mainQN[1].p_act
    theta_1 = tf.slice(theta_1_all, [0,0], [4,1])
    theta_2 = tf.slice(theta_2_all, [0,0], [4,1])

    theta_1_0 = tf.slice(theta_1_all, [4,0], [1,1])
    theta_2_0 = tf.slice(theta_2_all, [4,0], [1,1])

    p_1 = tf.nn.sigmoid(theta_1)
    p_2 = tf.nn.sigmoid(theta_2)

    p_1_0 = tf.nn.sigmoid(theta_1_0)
    p_2_0 = tf.nn.sigmoid(theta_2_0)

    p_1_0_v = tf.concat([p_1_0, (1-p_1_0)], 0)
    p_2_0_v = tf.concat([p_2_0, (1-p_2_0)], 0)

    s_0 = tf.reshape(tf.matmul(p_1_0_v, tf.transpose(p_2_0_v)), [-1, 1])

    P = tf.concat([
        tf.multiply(p_1, p_2),
        tf.multiply(p_1, 1 - p_2),
        tf.multiply(1 - p_1, p_2),
        tf.multiply(1 - p_1, 1 - p_2)
    ], 1)
    R_1 = tf.placeholder(shape=[4, 1], dtype=tf.float32)
    R_2 = tf.placeholder(shape=[4, 1], dtype=tf.float32)

    I_m_P = tf.diag([1.0, 1.0, 1.0, 1.0]) - P * gamma
    v_0 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), R_1), s_0,
        transpose_a=True
    )
    v_1 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), R_2), s_0,
        transpose_a=True
    )
    if reg > 0:
        for indx, _ in enumerate(mainQN[0].parameters):
            v_0 -= reg * tf.reduce_sum(
                tf.nn.l2_loss(tf.square(mainQN[0].parameters[indx]))
            )
            v_1 -= reg * tf.reduce_sum(
                tf.nn.l2_loss(tf.square(mainQN[1].parameters[indx]))
            )
    v_0_grad_theta_0 = flatgrad(v_0, mainQN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0, mainQN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1, mainQN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1, mainQN[1].parameters)


    v_0_grad_theta_0_wrong = flatgrad(v_0, mainQN[0].parameters)
    v_1_grad_theta_1_wrong = flatgrad(v_1, mainQN[1].parameters)
    param_len = v_0_grad_theta_0_wrong.get_shape()[0].value

    if pseudo:
        multiply0 = tf.matmul(
            tf.reshape(v_0_grad_theta_1, [1, param_len]),
            tf.reshape(v_1_grad_theta_1, [param_len, 1])
        )
        multiply1 = tf.matmul(
            tf.reshape(v_1_grad_theta_0, [1, param_len]),
            tf.reshape(v_0_grad_theta_0, [param_len, 1])
        )
    else:
        multiply0 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
            tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1])
       )
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_1_grad_theta_0), [1, param_len]),
            tf.reshape(v_0_grad_theta_0_wrong, [param_len, 1])
       )

    second_order0 = flatgrad(multiply0, mainQN[0].parameters)
    second_order1 = flatgrad(multiply1, mainQN[1].parameters)

    mainQN[0].R1 = R_1
    mainQN[1].R1 = R_2
    mainQN[0].v = v_0
    mainQN[1].v = v_1
    mainQN[0].delta = v_0_grad_theta_0
    mainQN[1].delta = v_1_grad_theta_1
    mainQN[0].delta += tf.multiply(second_order0, mainQN[0].lr_correction)
    mainQN[1].delta += tf.multiply(second_order1, mainQN[1].lr_correction)


def train(env, *, num_episodes=50, trace_length=200,
          simple_net=True, corrections=True, pseudo=False,
          num_hidden=10, reg=0.0, lr=1., lr_correction=0.5, gamma=0.96):
    # Get info about the env
    payout_mat_1 = env.payout_mat
    payout_mat_2 = env.payout_mat.T

    # Sanity
    tf.reset_default_graph()

    # Q-networks
    mainQN = []
    for agent in range(2):
        mainQN.append(Qnetwork('main' + str(agent), num_hidden, simple_net))

    # Corrections
    corrections_func(mainQN, corrections, gamma, pseudo, reg)

    results = []
    norm = 1 / (1 - gamma)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        lr_coor = np.ones(1) * lr_correction
        for episode in range(num_episodes):
            sess.run(init)

            log_items = {}
            log_items['episode'] = episode + 1

            res = []
            params_time = []
            delta_time = []
            input_vals = np.reshape(np.array(range(5)) + 1, [-1])
            for i in range(trace_length):
                params0 = mainQN[0].getparams()
                params1 = mainQN[1].getparams()
                outputs = [
                    mainQN[0].delta, mainQN[1].delta, mainQN[0].v, mainQN[1].v
                ]
                update1, update2, v1, v2 = sess.run(
                    outputs,
                    feed_dict={
                        mainQN[0].input_place: input_vals ,
                        mainQN[1].input_place: input_vals,
                        mainQN[0].R1: np.reshape(payout_mat_2, [-1, 1]),
                        mainQN[1].R1: np.reshape(payout_mat_1, [-1, 1]),
                        mainQN[0].lr_correction: lr_coor,
                        mainQN[1].lr_correction: lr_coor
                    }
                )
                update(mainQN, lr, update1, update2)
                params_time.append([params0, params1])
                delta_time.append([update1, update2])

                log_items['ret1'] = v1[0][0] / norm
                log_items['ret2'] = v2[0][0] / norm
                res.append([v1[0][0] / norm, v2[0][0] / norm])
            results.append(res)

            for k, v in sorted(log_items.items()):
                logger.record_tabular(k, v)
            logger.dump_tabular()
            logger.info('')

    return results
