"""
The magic corrections of LOLA.
"""
import tensorflow as tf

from .utils import flatgrad


def corrections_func(mainPN, batch_size, trace_length,
                     corrections=False, cube=None):
    """Computes corrections for policy gradients.

    Args:
    -----
        mainPN: list of policy/Q-networks
        batch_size: int
        trace_length: int
        corrections: bool (default: False)
            Whether policy networks should use corrections.
        cube: tf.Varialbe or None (default: None)
            If provided, should be constructed via `lola.utils.make_cube`.
            Used for variance reduction of the value estimation.
            When provided, the computation graph for corrections is faster to
            compile but is quite memory inefficient.
            When None, variance reduction graph is contructed dynamically,
            is a little longer to compile, but has lower memory footprint.
    """
    if cube is not None:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, 1, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length, 1])
        mat_1 = tf.reshape(tf.squeeze(tf.matmul(ac_logp1, ac_logp0)),
                           [batch_size, 1, trace_length * trace_length])

        v_0 = tf.matmul(tf.reshape(mainPN[0].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_0 = tf.reshape(v_0, [batch_size, trace_length, trace_length, trace_length])

        v_1 = tf.matmul(tf.reshape(mainPN[1].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_1 = tf.reshape(v_1, [batch_size, trace_length, trace_length, trace_length])

        v_0 = 2 * tf.reduce_sum(v_0 * cube) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1 * cube) / batch_size
    else:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length])

        # Static exclusive cumsum
        ac_logp0_cumsum = [tf.constant(0.)]
        ac_logp1_cumsum = [tf.constant(0.)]
        for i in range(trace_length - 1):
            ac_logp0_cumsum.append(tf.add(ac_logp0_cumsum[-1], ac_logp0[:, i]))
            ac_logp1_cumsum.append(tf.add(ac_logp1_cumsum[-1], ac_logp1[:, i]))

        # Compute v_0 and v_1
        mat_cumsum = ac_logp0[:, 0] * ac_logp1[:, 0]
        v_0 = mat_cumsum * mainPN[0].sample_reward[:, 0]
        v_1 = mat_cumsum * mainPN[1].sample_reward[:, 0]
        for i in range(1, trace_length):
            mat_cumsum = tf.add(mat_cumsum, ac_logp0[:, i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp0_cumsum[i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp1_cumsum[i] * ac_logp0[:, i])
            v_0 = tf.add(v_0, mat_cumsum * mainPN[0].sample_reward[:, i])
            v_1 = tf.add(v_1, mat_cumsum * mainPN[1].sample_reward[:, i])
        v_0 = 2 * tf.reduce_sum(v_0) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1) / batch_size

    v_0_pi_0 = 2*tf.reduce_sum(((mainPN[0].target-tf.stop_gradient(mainPN[0].value)) * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_0_pi_1 = 2*tf.reduce_sum(((mainPN[0].target-tf.stop_gradient(mainPN[0].value)) * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    v_1_pi_0 = 2*tf.reduce_sum(((mainPN[1].target-tf.stop_gradient(mainPN[1].value)) * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_1_pi_1 = 2*tf.reduce_sum(((mainPN[1].target-tf.stop_gradient(mainPN[1].value)) * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    v_0_grad_theta_0 = flatgrad(v_0_pi_0, mainPN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0_pi_1, mainPN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1_pi_0, mainPN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1_pi_1, mainPN[1].parameters)

    mainPN[0].grad = v_0_grad_theta_0
    mainPN[1].grad = v_1_grad_theta_1

    mainPN[0].grad_v_1 = v_1_grad_theta_0
    mainPN[1].grad_v_0 = v_0_grad_theta_1

    if corrections:
        v_0_grad_theta_0_wrong = flatgrad(v_0, mainPN[0].parameters)
        v_1_grad_theta_1_wrong = flatgrad(v_1, mainPN[1].parameters)

        param_len = v_0_grad_theta_0_wrong.get_shape()[0].value

        multiply0 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
            tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1])
        )
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_1_grad_theta_0), [1, param_len]),
            tf.reshape(v_0_grad_theta_0_wrong, [param_len, 1])
        )

        second_order0 = flatgrad(multiply0, mainPN[0].parameters)
        second_order1 = flatgrad(multiply1, mainPN[1].parameters)

        mainPN[0].v_0_grad_01 = second_order0
        mainPN[1].v_1_grad_10 = second_order1

        mainPN[0].delta = v_0_grad_theta_0 + second_order0
        mainPN[1].delta = v_1_grad_theta_1 + second_order1
    else:
        mainPN[0].delta = v_0_grad_theta_0
        mainPN[1].delta = v_1_grad_theta_1
