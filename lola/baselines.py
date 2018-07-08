"""
A collection of baselines used in the tournament.
All baselines implemented in Numpy.

Reference:
http://www.sciencedirect.com/science/article/pii/S0004370202001212
"""
import numpy as np


class NonLearner(object):
    def __init__(self, name, num_actions, num_states, seed=42):
        self.name = name
        self.rng = np.random.RandomState(seed)

    def act(self, state):
        return 0 if self.rng.rand() < 0.2 else 1

    def update(self, step, state, joint_ac, rew):
        pass



class NaiveQLearner(object):
    """
    Q-learner with epsilon-greedy exploration.
    """
    def __init__(self, name, num_actions, num_states,
                 gamma=0.96, alpha=1e-2, decay=1e-4, eps=0.05, seed=42):
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.eps = eps

        self.rng = np.random.RandomState(seed)

        self.Q = self.rng.randn(num_states, num_actions)

    @property
    def parameters(self):
        p = np.exp(self.Q[:, 0]) / np.exp(self.Q).sum(axis=1)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        logits = (np.log(p) - np.log(1 - p)).reshape((-1, 1))
        return logits

    def _joint_ac_to_state(self, joint_ac):
        ac0, ac1 = joint_ac
        state = np.zeros(self.num_states)
        state[ac0 * 2 + ac1] = 1
        return state

    def act(self, state):
        idx = np.nonzero(state)[0][0]
        greedy_ac = np.argmax(self.Q[idx])
        # greedy_ac = 1 - greedy_ac # DELETE: for debugging only!
        ac = greedy_ac if self.rng.rand() > self.eps else 1 - greedy_ac
        return ac

    def update(self, step, state, joint_ac, rew):
        # Compute V-function
        new_state = self._joint_ac_to_state(joint_ac)
        new_idx = np.nonzero(new_state)[0][0]
        V = np.max(self.Q[new_idx])

        # Update Q-function
        ac = joint_ac[0]
        idx = np.nonzero(state)[0][0]
        alpha = 1 / (1 / self.alpha + step * self.decay)
        self.Q[idx, ac] = (1 - alpha) * self.Q[idx, ac] + \
                          alpha * (rew + self.gamma * V)


class JointActionLearner(object):
    def __init__(self, name, num_actions, num_states,
                 gamma=0.96, alpha=1e-2, decay=1e-4, eps=0.05, seed=42):
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.eps = eps

        self.rng = np.random.RandomState(seed)

        self.Q = self.rng.randn(num_states, num_actions, num_actions)
        self.C = np.zeros([num_states, num_actions])
        self.N = np.zeros([num_states, 1])

    @property
    def parameters(self):
        M = np.reshape(self.C / (self.N + 1e-8),
                       (self.num_states, self.num_actions, 1))
        Q = np.sum(self.Q * M, axis=-1)
        p = np.exp(Q[:, 0]) / np.exp(Q).sum(axis=1)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        logits = (np.log(p) - np.log(1 - p)).reshape((-1, 1))
        return logits

    def _comp_q_prime(self, state, eps=1e-8):
        idx = np.nonzero(state)[0][0]
        q, c, n = self.Q[idx], self.C[idx], self.N[idx]
        q_prime = np.sum(q * c / (n + eps), axis=-1)
        return q_prime

    def _joint_ac_to_state(self, joint_ac):
        ac0, ac1 = joint_ac
        state = np.zeros(self.num_states)
        state[ac0 * 2 + ac1] = 1
        return state

    def act(self, state):
        greedy_ac = np.argmax(self._comp_q_prime(state))
        # greedy_ac = 1 - greedy_ac # DELETE: for debugging only!
        ac = greedy_ac if self.rng.rand() > self.eps else 1 - greedy_ac
        return ac

    def update(self, step, state, joint_ac, rew):
        # Compute V-function
        new_state = self._joint_ac_to_state(joint_ac)
        V = np.max(self._comp_q_prime(new_state))

        # Update Q-function
        ac0, ac1 = joint_ac
        idx = np.nonzero(state)[0][0]
        alpha = 1 / (1 / self.alpha + step * self.decay)
        self.Q[idx, ac0, ac1] = (1 - alpha) * self.Q[idx, ac0, ac1] + \
                                alpha * (rew + self.gamma * V)

        # Update C and N counts
        ac1 = joint_ac[1]
        self.C[idx, ac1] += 1
        self.N[idx] += 1


class PolicyHillClimbing(object):
    def __init__(self, name, num_actions, num_states,
                 gamma=0.96, alpha=1e-2, alpha_decay=1e-4,
                 delta=5e-5, delta_decay=1., eps=0.05, seed=42):
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.delta = delta
        self.delta_decay = delta_decay
        self.eps = eps

        self.rng = np.random.RandomState(seed)

        self.Q = self.rng.randn(num_states, num_actions)
        self.pi = np.ones([num_states, num_actions]) / num_actions

    @property
    def parameters(self):
        p = np.clip(self.pi[:, 0], 1e-8, 1 - 1e-8)
        logits = (np.log(p) - np.log(1 - p)).reshape((-1, 1))
        return logits

    def _joint_ac_to_state(self, joint_ac):
        ac0, ac1 = joint_ac
        state = np.zeros(self.num_states)
        state[ac0 * 2 + ac1] = 1
        return state

    def act(self, state):
        idx = np.nonzero(state)[0][0]
        ac = np.nonzero(self.rng.multinomial(1, self.pi[idx]))[0][0]
        return ac

    def update(self, step, state, joint_ac, rew, eps=1e-8):
        # Compute V-function
        new_state = self._joint_ac_to_state(joint_ac)
        new_idx = np.nonzero(new_state)[0][0]
        V = np.max(self.Q[new_idx])

        # Update Q-function
        ac = joint_ac[0]
        idx = np.nonzero(state)[0][0]
        alpha = 1 / (1 / self.alpha + step * self.alpha_decay)
        self.Q[idx, ac] = (1 - alpha) * self.Q[idx, ac] + \
                          alpha * (rew + self.gamma * V)

        # Compute step size
        ac = joint_ac[0]
        best_ac = np.argmax(self.Q[idx])
        delta = 1 / (1 / self.delta + step * self.delta_decay)
        delta_s = np.minimum(self.pi[idx], delta / (self.num_actions - 1))
        delta_pi = -delta_s[ac] if ac != best_ac else delta_s.sum() - delta_s[ac]

        # Update pi
        ac = joint_ac[0]
        self.pi[idx, ac] += delta_pi
        self.pi[idx, 1 - ac] -= delta_pi
        self.pi[idx] = np.clip(self.pi[idx], eps, 1. - eps)
        # self.pi[idx] /= self.pi[idx].sum()


class WoLF_PolicyHillClimbing(object):
    def __init__(self, name, num_actions, num_states,
                 gamma=0.96, alpha=1e-2, alpha_decay=1e-4,
                 delta=1e-2, delta_decay=5e-5, eps=0.05, seed=42):
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.delta = delta
        self.delta_decay = delta_decay
        self.eps = eps

        self.rng = np.random.RandomState(seed)

        self.Q = self.rng.randn(num_states, num_actions)
        self.C = np.zeros([num_states, 1])

        self.pi = np.ones([num_states, num_actions]) / num_actions
        self.pi_avg = np.ones([num_states, num_actions]) / num_actions

    @property
    def parameters(self):
        p = np.clip(self.pi[:, 0], 1e-8, 1 - 1e-8)
        logits = (np.log(p) - np.log(1 - p)).reshape((-1, 1))
        return logits

    def _joint_ac_to_state(self, joint_ac):
        ac0, ac1 = joint_ac
        state = np.zeros(self.num_states)
        state[ac0 * 2 + ac1] = 1
        return state

    def act(self, state):
        idx = np.nonzero(state)[0][0]
        ac = np.nonzero(self.rng.multinomial(1, self.pi[idx]))[0][0]
        return ac

    def update(self, step, state, joint_ac, rew, eps=1e-8):
        # Compute V-function
        new_state = self._joint_ac_to_state(joint_ac)
        new_idx = np.nonzero(new_state)[0][0]
        V = np.max(self.Q[new_idx])

        # Update Q-function
        ac = joint_ac[0]
        idx = np.nonzero(state)[0][0]
        alpha = 1 / (1 / self.alpha + step * self.alpha_decay)
        self.Q[idx, ac] = (1 - alpha) * self.Q[idx, ac] + \
                          alpha * (rew + self.gamma * V)

        # Update estimate of pi_avg
        self.C[idx] += 1
        self.pi_avg[idx] += (self.pi[idx] - self.pi_avg[idx]) / self.C[idx]
        self.pi_avg[idx] = np.clip(self.pi_avg[idx], eps, 1. - eps)
        self.pi_avg[idx] /= self.pi_avg[idx].sum()

        # Compute step size
        Q_exp_pi = np.sum(self.pi[idx] * self.Q[idx])
        Q_exp_pi_avg = np.sum(self.pi_avg[idx] * self.Q[idx])
        delta_w = 1 / (1 / self.delta + step * self.delta_decay)
        delta = delta_w if Q_exp_pi > Q_exp_pi_avg else 2 * delta_w

        best_ac = np.argmax(self.Q[idx])
        delta_s = np.minimum(self.pi[idx], delta / (self.num_actions - 1))
        delta_pi = -delta_s[ac] if ac != best_ac else delta_s.sum() - delta_s[ac]

        # Update pi
        ac = joint_ac[0]
        self.pi[idx, ac] += delta_pi
        self.pi[idx, 1 - ac] -= delta_pi
        self.pi[idx] = np.clip(self.pi[idx], eps, 1. - eps)
