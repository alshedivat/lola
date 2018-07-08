"""
PG training for the Iterated Prisoner's Dilemma and Matching Pennies.
"""
import numpy as np
import tensorflow as tf

from . import logger

from .corrections import *
from .networks import *
from .utils import *


def update(mainQN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainQN[0].setparams(
        mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    update_theta_2 = mainQN[1].setparams(
        mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def train(env, *, num_episodes, trace_length, batch_size, gamma,
          set_zero, lr, corrections, simple_net, hidden,
          mem_efficient=True):
    observation_space = env.NUM_STATES
    y = gamma
    load_model = False #Whether to load a saved model.
    n_agents = env.NUM_AGENTS
    total_n_agents = n_agents
    max_epLength = trace_length + 1 #The max allowed length of our episode.
    summaryLength = 20 #Number of epidoes to periodically save for analysis

    tf.reset_default_graph()
    mainQN = []

    agent_list = np.arange(total_n_agents)
    for agent in range(total_n_agents):
        mainQN.append(Qnetwork('main' + str(agent), agent, env, lr=lr, gamma=gamma, batch_size=batch_size, trace_length=trace_length, hidden=hidden, simple_net=simple_net))

    if not mem_efficient:
        cube, cube_ops = make_cube(trace_length)
    else:
        cube, cube_ops = None, None

    corrections_func(mainQN,
                     batch_size=batch_size,
                     trace_length=trace_length,
                     corrections=corrections,
                     cube=cube)

    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()

    buffers = []
    for i in range(total_n_agents):
        buffers.append(ExperienceBuffer(batch_size))

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    aList = []

    total_steps = 0

    episodes_run = np.zeros(total_n_agents)
    episodes_run_counter =  np.zeros(total_n_agents)
    episodes_reward = np.zeros(total_n_agents)
    episodes_actions = np.zeros((total_n_agents, env.NUM_ACTIONS))
                # need to multiple with
    pow_series = np.arange(trace_length)
    discount = np.array([pow(gamma, item) for item in pow_series])
    discount_array = gamma**trace_length / discount
    # print('discount_array',discount_array.shape)
    discount = np.expand_dims(discount, 0)
    discount_array = np.reshape(discount_array,[1,-1])


    array = np.eye(env.NUM_STATES)
    feed_dict_log_pi = {mainQN[0].scalarInput: array,
                        mainQN[1].scalarInput: array,}

    with tf.Session() as sess:
        sess.run(init)
        if cube_ops is not None:
            sess.run(cube_ops)

        if set_zero == 1:
            for i in range(2):
                mainQN[i].setparams(np.zeros((5)))
                theta_2_vals =  mainQN[i].getparams()

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

            # Reset environment and get first new observation
            sP = env.reset()
            s = sP
            state = []

            d = False
            rAll = np.zeros((n_agents))
            aAll = np.zeros((env.NUM_STATES))
            j = 0

            for agent in these_agents:
                episodes_run[agent] += 1
                episodes_run_counter[agent] += 1
            a_all_old = [0,0]

            # The Q-Network
            while j < max_epLength:
                j += 1
                a_all = []
                for agent_role, agent in enumerate(these_agents):
                    a = sess.run(
                        [mainQN[agent].predict],
                        feed_dict={
                            mainQN[agent].scalarInput: [s[agent_role]]
                        }
                    )
                    a_all.append(a[0])

                a_all_old = a_all
                if a_all[0] > 1 or a_all[1] > 1:
                    print('warning!!!', a_all, 's', s)
                s1P, r, d = env.step(a_all)
                s1 = s1P

                total_steps += 1
                for agent_role, agent in enumerate(these_agents):
                    episodeBuffer[agent_role].append([
                        s[0], a_all[agent_role], r[agent_role], s1[0],
                        d, these_agents[agent_role]
                    ])
                    episodes_reward[agent] += r[agent_role]
                rAll += [r[ii]*gamma**(j-1) for ii in range(2)]

                aAll[a_all[0]] += 1
                aAll[a_all[1]+2] += 1
                s_old = s
                s = s1
                sP = s1P
                if d == True:
                    break

            # Add the episode to the experience buffer
            for agent_role, agent in enumerate(these_agents):
                buffers[agent].add(np.array(episodeBuffer[agent_role]))

            jList.append(j)
            rList.append(rAll)
            aList.append(aAll)

            if (episodes_run[agent] % batch_size == 0 and
                episodes_run[agent] > 0):
                trainBatch0 = buffers[0].sample(batch_size, trace_length) #Get a random batch of experiences.
                trainBatch1 = buffers[1].sample(batch_size, trace_length)

                sample_return0 = np.reshape(get_monte_carlo(trainBatch0[:,2], y, trace_length, batch_size), [batch_size, -1])
                sample_return1 = np.reshape(get_monte_carlo(trainBatch1[:,2], y, trace_length, batch_size), [batch_size, -1])

                sample_reward0 = np.reshape(trainBatch0[:,2]- np.mean(trainBatch0[:,2]), [-1, trace_length]) * discount
                sample_reward1 = np.reshape(trainBatch1[:,2] - np.mean(trainBatch1[:,2]), [-1, trace_length]) * discount

                last_state = np.reshape(np.vstack(trainBatch0[:,3]), [-1, trace_length, env.NUM_STATES])[:,-1,:]

                value_0_next, value_1_next = sess.run(
                    [mainQN[0].value, mainQN[1].value],
                    feed_dict={
                        mainQN[0].scalarInput: last_state,
                        mainQN[1].scalarInput: last_state,
                    }
                )

                fetches = [
                    mainQN[0].values,
                    mainQN[0].updateModel,
                    mainQN[1].updateModel,
                    mainQN[0].delta, mainQN[1].delta,
                    mainQN[0].grad,
                    mainQN[1].grad,
                    mainQN[0].v_0_grad_01,
                    mainQN[1].v_1_grad_10
                ]
                feed_dict = {
                    mainQN[0].scalarInput: np.vstack(trainBatch0[:,0]),
                    mainQN[0].sample_return: sample_return0,
                    mainQN[0].actions: trainBatch0[:,1],
                    mainQN[1].scalarInput: np.vstack(trainBatch1[:,0]),
                    mainQN[1].sample_return: sample_return1,
                    mainQN[1].actions: trainBatch1[:,1],
                    mainQN[0].sample_reward: sample_reward0,
                    mainQN[1].sample_reward: sample_reward1,
                    mainQN[0].next_value: value_0_next,
                    mainQN[1].next_value: value_1_next,
                    mainQN[0].gamma_array: discount,
                    mainQN[1].gamma_array: discount,
                    mainQN[0].gamma_array_inverse: discount_array,
                    mainQN[1].gamma_array_inverse: discount_array,
                }
                if episodes_run[agent]  %  batch_size == 0 and  episodes_run[agent] > 0:
                    values, _, _, update1, update2, grad_1, grad_2, v0_grad_01, v1_grad_10 = sess.run(fetches, feed_dict=feed_dict)

                if episodes_run[agent]  %  batch_size == 0  and  episodes_run[agent] > 0:
                    update(mainQN, lr, update1, update2)
                    updated =True
                    print('update params')
                    print('grad_1', grad_1)
                    print('grad_2', grad_2)
                    print('v0_grad_01',v0_grad_01)
                    print('v1_grad_10', v1_grad_10)
                    print('values', values)
                episodes_run_counter[agent] = episodes_run_counter[agent] *0
                episodes_actions[agent] = episodes_actions[agent]*0
                episodes_reward[agent] =episodes_reward[agent] *0

            if len(rList) % summaryLength == 0 and len(rList) != 0 and updated == True:
                updated = False
                gamma_discount = 1 / (1-gamma)
                print(total_steps,'reward', np.mean(rList[-summaryLength:], 0)/gamma_discount, 'action', (np.mean(aList[-summaryLength:], 0)*2.0/ np.sum(np.mean(aList[-summaryLength:], 0)))*100//1)


                action_prob = np.mean(aList[-summaryLength:], 0)*2.0/ np.sum(np.mean(aList[-summaryLength:], 0))
                log_items = {}
                log_items['reward_agent0'] = np.mean(rList[-summaryLength:], 0)[0]
                log_items['reward_agent1'] = np.mean(rList[-summaryLength:], 0)[1]
                log_items['agent0_C'] = action_prob[0]
                log_items['agent0_D'] = action_prob[1]
                log_items['agent1_C'] = action_prob[2]
                log_items['agent1_D'] = action_prob[3]
                if simple_net:
                    theta_1_vals =  mainQN[0].getparams()
                    theta_2_vals =  mainQN[1].getparams()
                    print('theta_1_vals', theta_1_vals)
                    print('theta_2_vals', theta_2_vals)

                    log_items['theta_1_0'] = theta_1_vals[0]
                    log_items['theta_1_1'] = theta_1_vals[1]
                    log_items['theta_1_2'] = theta_1_vals[2]
                    log_items['theta_1_3'] = theta_1_vals[3]
                    log_items['theta_1_4'] = theta_1_vals[4]
                    log_items['theta_2_0'] = theta_2_vals[0]
                    log_items['theta_2_1'] = theta_2_vals[1]
                    log_items['theta_2_2'] = theta_2_vals[2]
                    log_items['theta_2_3'] = theta_2_vals[3]
                    log_items['theta_2_4'] = theta_2_vals[4]
                else:
                    log_pi0, log_pi1 = sess.run([mainQN[0].log_pi, mainQN[1].log_pi], feed_dict=feed_dict_log_pi)
                    print('pi 0', np.exp(log_pi0))
                    print('pi 1', np.exp(log_pi1))

                    log_items['pi_1_0'] = np.exp(log_pi0[0][0])
                    log_items['pi_1_1'] = np.exp(log_pi0[1][0])
                    log_items['pi_1_2'] = np.exp(log_pi0[2][0])
                    log_items['pi_1_3'] = np.exp(log_pi0[3][0])
                    log_items['pi_1_4'] = np.exp(log_pi0[4][0])

                    log_items['pi_2_0'] = np.exp(log_pi1[0][0])
                    log_items['pi_2_1'] = np.exp(log_pi1[1][0])
                    log_items['pi_2_2'] = np.exp(log_pi1[2][0])
                    log_items['pi_2_3'] = np.exp(log_pi1[3][0])
                    log_items['pi_2_4'] = np.exp(log_pi1[4][0])


                for key in sorted(log_items.keys()):
                    logger.record_tabular(key, log_items[key])
                logger.dump_tabular()
                logger.info('')
