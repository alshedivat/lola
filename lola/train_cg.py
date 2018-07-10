"""
Training funcion for the Coin Game.
"""
import os
import numpy as np
import tensorflow as tf

from . import logger

from .corrections import *
from .networks import *
from .utils import *


def update(mainPN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainPN[0].setparams(
        mainPN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    update_theta_2 = mainPN[1].setparams(
        mainPN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def clone_update(mainPN_clone):
    for i in range(2):
        mainPN_clone[i].log_pi_clone = tf.reduce_mean(
            mainPN_clone[i].log_pi_action_bs)
        mainPN_clone[i].clone_trainer = \
            tf.train.GradientDescentOptimizer(learning_rate=0.1)
        mainPN_clone[i].update = mainPN_clone[i].clone_trainer.minimize(
            -mainPN_clone[i].log_pi_clone, var_list=mainPN_clone[i].parameters)


def train(env, *, num_episodes, trace_length, batch_size,
          corrections, opp_model, grid_size, gamma, hidden, bs_mul, lr,
          mem_efficient=True):
    #Setting the training parameters
    batch_size = batch_size #How many experience traces to use for each training step.
    trace_length = trace_length #How long each experience trace will be when training

    y = gamma
    num_episodes = num_episodes #How many episodes of game environment to train network with.
    load_model = False #Whether to load a saved model.
    path = "./drqn" #The path to save our model to.
    n_agents = env.NUM_AGENTS
    total_n_agents = n_agents
    h_size = [hidden] * total_n_agents
    max_epLength = trace_length+1 #The max allowed length of our episode.
    summary_len = 20 #Number of episodes to periodically save for analysis

    tf.reset_default_graph()
    mainPN = []
    mainPN_step = []
    agent_list = np.arange(total_n_agents)
    for agent in range(total_n_agents):
        mainPN.append(
            Pnetwork('main' + str(agent), h_size[agent], agent, env,
                trace_length=trace_length, batch_size=batch_size,))
        mainPN_step.append(
            Pnetwork('main' + str(agent), h_size[agent], agent, env,
                trace_length=trace_length, batch_size=batch_size,
                reuse=True, step=True))

    # Clones of the opponents
    if opp_model:
        mainPN_clone = []
        for agent in range(total_n_agents):
            mainPN_clone.append(
                Pnetwork('clone' + str(agent), h_size[agent], agent, env,
                         trace_length=trace_length, batch_size=batch_size))

    if not mem_efficient:
        cube, cube_ops = make_cube(trace_length)
    else:
        cube, cube_ops = None, None

    if not opp_model:
        corrections_func(mainPN, batch_size, trace_length, corrections, cube)
    else:
        corrections_func([mainPN[0], mainPN_clone[1]],
                         batch_size, trace_length, corrections, cube)
        corrections_func([mainPN[1], mainPN_clone[0]],
                         batch_size, trace_length, corrections, cube)
        clone_update(mainPN_clone)

    init = tf.global_variables_initializer()
    # saver = tf.train.Saver(max_to_keep=5)

    trainables = tf.trainable_variables()

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    aList = []

    total_steps = 0

    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    episodes_run = np.zeros(total_n_agents)
    episodes_run_counter =  np.zeros(total_n_agents)
    episodes_reward = np.zeros((total_n_agents, batch_size))
    episodes_actions = np.zeros((total_n_agents, env.NUM_ACTIONS))

    pow_series = np.arange(trace_length)
    discount = np.array([pow(gamma, item) for item in pow_series])
    discount_array = gamma**trace_length / discount
    discount = np.expand_dims(discount, 0)
    discount_array = np.reshape(discount_array,[1,-1])

    with tf.Session() as sess:
        # if load_model == True:
        #     print( 'Loading Model...')
        #     ckpt = tf.train.get_checkpoint_state(path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)
        if not mem_efficient:
            sess.run(cube_ops)

        sP = env.reset()
        updated =True
        for i in range(num_episodes):
            episodeBuffer = []
            for ii in range(n_agents):
                episodeBuffer.append([])
            np.random.shuffle(agent_list)
            if n_agents  == total_n_agents:
                these_agents = range(n_agents)
            else:
                these_agents = sorted(agent_list[0:n_agents])

            #Reset environment and get first new observation
            sP = env.reset()
            s = sP

            trainBatch0 = [[], [], [], [], [], []]
            trainBatch1 = [[], [], [], [], [], []]
            d = False
            rAll = np.zeros((4))
            aAll = np.zeros((env.NUM_ACTIONS * 2))
            j = 0

            lstm_state = []
            for agent in these_agents:
                episodes_run[agent] += 1
                episodes_run_counter[agent] += 1
                lstm_state.append(np.zeros((batch_size, h_size[agent]*2)))

            while j < max_epLength:
                lstm_state_old = lstm_state
                j += 1
                a_all = []
                lstm_state = []
                for agent_role, agent in enumerate(these_agents):
                    a, lstm_s = sess.run(
                        [
                            mainPN_step[agent].predict,
                            mainPN_step[agent].lstm_state_output
                        ],
                        feed_dict={
                            mainPN_step[agent].state_input: s,
                            mainPN_step[agent].lstm_state: lstm_state_old[agent]
                        }
                    )
                    lstm_state.append(lstm_s)
                    a_all.append(a)

                trainBatch0[0].append(s)
                trainBatch1[0].append(s)
                trainBatch0[1].append(a_all[0])
                trainBatch1[1].append(a_all[1])

                a_all = np.transpose(np.vstack(a_all))

                s1P,r,d = env.step(actions=a_all)
                s1 = s1P

                trainBatch0[2].append(r[0])
                trainBatch1[2].append(r[1])
                trainBatch0[3].append(s1)
                trainBatch1[3].append(s1)
                trainBatch0[4].append(d)
                trainBatch1[4].append(d)
                trainBatch0[5].append(lstm_state[0])
                trainBatch1[5].append(lstm_state[1])

                total_steps += 1
                for agent_role, agent in enumerate(these_agents):
                    episodes_reward[agent] += r[agent_role]

                for index in range(batch_size):
                    r_pb = [r[0][index], r[1][index]]
                    if np.array(r_pb).any():
                        if r_pb[0] == 1 and r_pb[1] == 0:
                            rAll[0] += 1
                        elif r_pb[0] == 0 and r_pb[1] == 1:
                            rAll[1] += 1
                        elif r_pb[0] == 1 and r_pb[1] == -2:
                            rAll[2] += 1
                        elif r_pb[0] == -2 and r_pb[1] == 1:
                            rAll[3] += 1

                aAll[a_all[0]] += 1
                aAll[a_all[1] + 4] += 1
                s_old = s
                s = s1
                sP = s1P
                if d.any():
                    break

            jList.append(j)
            rList.append(rAll)
            aList.append(aAll)

            # training after one batch is obtained
            sample_return0 = np.reshape(
                get_monte_carlo(trainBatch0[2], y, trace_length, batch_size),
                [batch_size, -1])
            sample_return1 = np.reshape(
                get_monte_carlo(trainBatch1[2], y, trace_length, batch_size),
                [batch_size, -1])
            # need to multiple with
            pow_series = np.arange(trace_length)
            discount = np.array([pow(gamma, item) for item in pow_series])

            sample_reward0 = discount * np.reshape(
                trainBatch0[2] - np.mean(trainBatch0[2]), [-1, trace_length])
            sample_reward1 = discount * np.reshape(
                trainBatch1[2]- np.mean(trainBatch1[2]), [-1, trace_length])

            state_input0 = np.concatenate(trainBatch0[0], axis=0)
            state_input1 = np.concatenate(trainBatch1[0], axis=0)
            actions0 = np.concatenate(trainBatch0[1], axis=0)
            actions1 = np.concatenate(trainBatch1[1], axis=0)

            last_state = np.reshape(
                np.concatenate(trainBatch1[3], axis=0),
                [batch_size, trace_length, env.ob_space_shape[0],
                 env.ob_space_shape[1], env.ob_space_shape[2]])[:,-1,:,:,:]

            value_0_next, value_1_next = sess.run(
                [mainPN_step[0].value, mainPN_step[1].value],
                feed_dict={
                    mainPN_step[0].state_input: last_state,
                    mainPN_step[1].state_input: last_state,
                    mainPN_step[0].lstm_state: lstm_state[0],
                    mainPN_step[1].lstm_state: lstm_state[1],
                })

            if opp_model:
                ## update local clones
                update_clone = [mainPN_clone[0].update, mainPN_clone[1].update]
                feed_dict = {
                    mainPN_clone[0].state_input: state_input1,
                    mainPN_clone[0].actions: actions1,
                    mainPN_clone[0].sample_return: sample_return1,
                    mainPN_clone[0].sample_reward: sample_reward1,
                    mainPN_clone[1].state_input: state_input0,
                    mainPN_clone[1].actions: actions0,
                    mainPN_clone[1].sample_return: sample_return0,
                    mainPN_clone[1].sample_reward: sample_reward0,
                    mainPN_clone[0].gamma_array: np.reshape(discount,[1,-1]),
                    mainPN_clone[1].gamma_array: np.reshape(discount,[1,-1]),
                }
                num_loops = 50 if i == 0 else 1
                for i in range(num_loops):
                    sess.run(update_clone, feed_dict=feed_dict)

                theta_1_vals = mainPN[0].getparams()
                theta_2_vals = mainPN[1].getparams()
                theta_1_vals_clone = mainPN_clone[0].getparams()
                theta_2_vals_clone = mainPN_clone[1].getparams()

                if len(rList) % summary_len == 0:
                    print('params check before optimization')
                    print('theta_1_vals', theta_1_vals)
                    print('theta_2_vals_clone', theta_2_vals_clone)
                    print('theta_2_vals', theta_2_vals)
                    print('theta_1_vals_clone', theta_1_vals_clone)
                    print('diff between theta_1 and theta_2_vals_clone',
                        np.linalg.norm(theta_1_vals - theta_2_vals_clone))
                    print('diff between theta_2 and theta_1_vals_clone',
                        np.linalg.norm(theta_2_vals - theta_1_vals_clone))

            # Update policy networks
            feed_dict={
                mainPN[0].state_input: state_input0,
                mainPN[0].sample_return: sample_return0,
                mainPN[0].actions: actions0,
                mainPN[1].state_input: state_input1,
                mainPN[1].sample_return: sample_return1,
                mainPN[1].actions: actions1,
                mainPN[0].sample_reward: sample_reward0,
                mainPN[1].sample_reward: sample_reward1,
                mainPN[0].gamma_array: np.reshape(discount, [1, -1]),
                mainPN[1].gamma_array: np.reshape(discount, [1, -1]),
                mainPN[0].next_value: value_0_next,
                mainPN[1].next_value: value_1_next,
                mainPN[0].gamma_array_inverse:
                    np.reshape(discount_array, [1, -1]),
                mainPN[1].gamma_array_inverse:
                    np.reshape(discount_array, [1, -1]),
            }
            if opp_model:
                feed_dict.update({
                    mainPN_clone[0].state_input:state_input1,
                    mainPN_clone[0].actions: actions1,
                    mainPN_clone[0].sample_return: sample_return1,
                    mainPN_clone[0].sample_reward: sample_reward1,
                    mainPN_clone[1].state_input:state_input0,
                    mainPN_clone[1].actions: actions0,
                    mainPN_clone[1].sample_return: sample_return0,
                    mainPN_clone[1].sample_reward: sample_reward0,
                    mainPN_clone[0].gamma_array: np.reshape(discount,[1,-1]),
                    mainPN_clone[1].gamma_array:  np.reshape(discount,[1,-1]),
                })
            values, _, _, update1, update2 = sess.run(
                [
                    mainPN[0].value,
                    mainPN[0].updateModel,
                    mainPN[1].updateModel,
                    mainPN[0].delta,
                    mainPN[1].delta
                ],
                feed_dict=feed_dict)

            update(mainPN, lr, update1 / bs_mul, update2 / bs_mul)
            updated = True
            print('update params')

            episodes_run_counter[agent] = episodes_run_counter[agent] * 0
            episodes_actions[agent] = episodes_actions[agent] * 0
            episodes_reward[agent] = episodes_reward[agent] * 0

            if len(rList) % summary_len == 0 and len(rList) != 0 and updated:
                updated = False
                print(total_steps, 'reward', np.sum(rList[-summary_len:], 0))
                rlog = np.sum(rList[-summary_len:], 0)
                for ii in range(len(rlog)):
                    logger.record_tabular('rList['+str(ii)+']', rlog[ii])
                logger.dump_tabular()
                logger.info('')
