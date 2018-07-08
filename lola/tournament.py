"""
Round-robin tournament for matrix games.

Agents used in the tournament:
- LOLA (with exact value functions)
- Naive Policy Gradient Learner
- Naive Q-learner (+ epsilon-greedy exploration)
- JAL (= Q-learner + trivial opponent modeling + epsilon-greedy exploration)
- Policy Hill-Climbing
- WoLF + Policy Hill-Climbing
"""
import numpy as np
import tensorflow as tf

from functools import partial

from . import logger

from .baselines import *
from .exact import *
from .utils import *


AGENTS= {
    'NL': NonLearner,
    'NL-Q': NaiveQLearner,
    'JAL-Q': JointActionLearner,
    'PHC': PolicyHillClimbing,
    'WoLF': WoLF_PolicyHillClimbing,
    'NL-PG': partial(ExactLOLA, enable_corrections=False),
    'LOLA': partial(ExactLOLA, enable_corrections=True),
}


def train(env, *, num_episodes=1000, trace_length=10, lr=1., gamma=0.96, seed=42):
    agent_names = ['NL-Q', 'JAL-Q', 'PHC', 'WoLF', 'NL-PG', 'LOLA']

    # Get info about the env
    payout_mat_1 = env.payout_mat
    payout_mat_2 = env.payout_mat.T

    # Sanity
    tf.reset_default_graph()

    # Construct agents for all pairs of tournament matches
    num_agents = 0
    agent_pairs = []
    for a1 in range(len(agent_names)):
        for a2 in range(a1, len(agent_names)):
            agent_pair = []
            for i, name in enumerate([agent_names[a1], agent_names[a2]]):
                num_agents += 1
                agent = AGENTS[name](
                    name + str(num_agents), env.NUM_ACTIONS, env.NUM_STATES,
                    seed=seed + num_agents * 1000)
                agent_pair.append(agent)
            pair_name = agent_names[a1] + '_vs_' + agent_names[a2]
            agent_pairs.append((pair_name, agent_pair))
    assert len(agent_pairs) == len(agent_names) * (len(agent_names) + 1) // 2

    # Setup TF and LOLA stuff
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for name, agent_pair in agent_pairs:
        for agent in  agent_pair:
            if hasattr(agent, 'setup'):
                agent.setup(sess, env.NAME, env.payout_mat)

    # Run the tournament
    results = []
    actions = []

    T = [0 for _ in range(len(agent_pairs))]
    for episode in range(num_episodes):
        log_items = {}
        log_items['episode'] = episode + 1

        acs = [[] for _ in range(len(agent_pairs))]
        rets = [None for _ in range(len(agent_pairs))]
        rews = [[] for _ in range(len(agent_pairs))]

        # Iterate over the matches in the tournament for each agent pair
        for j, (name, agent_pair) in enumerate(agent_pairs):
            ob1, ob2 = env.reset()
            for _ in range(trace_length):
                T[j] += 1

                a1, a2 = agent_pair

                # Act
                ac1 = a1.act(ob1)
                ac2 = a2.act(ob2)

                # Make transition
                ob1_prev, ob2_prev = ob1, ob2
                (ob1, ob2), (rew1, rew2), done = env.step((ac1, ac2))

                # Update agents
                if not isinstance(a1, ExactLOLA):
                    a1.update(T[j], ob1_prev, (ac1, ac2), rew1)
                if not isinstance(a2, ExactLOLA):
                    a2.update(T[j], ob2_prev, (ac2, ac1), rew2)

                # Update arrays
                rews[j].append((rew1, rew2))
                acs[j].append((ac1, ac2))

            rets[j] = [
                (1 - gamma) * np.sum([
                    rew[0] * (gamma**t) for t, rew in enumerate(rews[j])
                ]),
                (1 - gamma) * np.sum([
                    rew[1] * (gamma**t) for t, rew in enumerate(rews[j])
                ]),
            ]
            log_items[name + '_ret1'] = rets[j][0]
            log_items[name + '_ret2'] = rets[j][1]
            log_items[name + '_pi1'] = 1 - np.mean([jac[0] for jac in acs[j]])
            log_items[name + '_pi2'] = 1 - np.mean([jac[1] for jac in acs[j]])

            # Do LOLA updates
            if isinstance(a1, ExactLOLA) or isinstance(a2, ExactLOLA):
                a1_params = a1.parameters
                a2_params = a2.parameters
            if isinstance(a1, ExactLOLA):
                v1, v2 = a1.update(a2_params)
                log_items[name + '_v1-1'] = v1
                log_items[name + '_v1-2'] = v2
            if isinstance(a2, ExactLOLA):
                v1, v2 = a2.update(a1_params)
                log_items[name + '_v2-1'] = v1
                log_items[name + '_v2-2'] = v2

        results.append(rets)
        actions.append(acs)

        for k, v in sorted(log_items.items()):
            logger.record_tabular(k, v)
        logger.dump_tabular()

    sess.close()
