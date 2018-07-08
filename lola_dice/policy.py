"""A collection of policy networks."""
import numpy as np
import tensorflow as tf
import sonnet as snt


class Policy(object):
    """The base class for policy networks.

    Policy parameters are allowed to be functions of other policies. To keep
    track of such dependencies, each policy stores a list of parent policies on
    which it depends. To make an action or update a policy with a non-empty
    list of dependencies, we need to ensure that all parent placeholders are
    fed-in with appropriate values.
    """

    def __init__(self, ob_size, num_actions, prev=None):
        self.ob_size = ob_size
        self.num_actions = num_actions
        self._root = self
        self._parents = tuple()
        if prev is not None:
            self._root = prev.root
            self._parents = prev.parents + (prev, )
        self._params = []
        self._opponents = None

    def build(self, scope, reuse=None):
        raise NotImplementedError

    @property
    def opponents(self):
        return self._opponents

    @opponents.setter
    def opponents(self, opponents):
        self._opponents = opponents

    @property
    def parameters(self):
        raise NotImplementedError

    @property
    def parents(self):
        return self._parents

    @property
    def root(self):
        return self._root

    def get_feed_list(self, trace):
        obs, acs, rets, values, infos = trace
        aa = np.asarray([info['available_actions'] for info in infos])
        feed_list = [
            (self.obs_ph, obs),
            (self.acs_ph, acs),
            (self.rets_ph, rets),
            (self.values_ph, values),
            (self.avail_acs_ph, aa)
        ]
        return feed_list

    def act(self, ob, info, sess, parent_feed_list=[]):
        aa = info['available_actions']
        feed_list = [(self.obs_ph, [ob]), (self.avail_acs_ph, [aa])] + \
                    parent_feed_list
        ac = sess.run(self.action, feed_dict=dict(feed_list))
        return ac

    def predict(self, ob, sess, parent_feed_list=[]):
        feed_list = [(self.obs_ph, [ob])] + parent_feed_list
        vpred = sess.run(self.vpred, feed_dict=dict(feed_list))
        return vpred

    @property
    def parameters(self):
        return self._params


class SimplePolicy(Policy):
    """A single layer network that maps states to action probabilities."""

    def build(self, scope, reuse=None):
        self.scope = scope
        with tf.variable_scope(scope, reuse=reuse):
            # Placeholders
            self.acs_ph = tf.placeholder(
                shape=[None, None], dtype=tf.int32, name="acs")
            self.obs_ph = tf.placeholder(
                shape=[None, None, self.ob_size], dtype=tf.float32, name="obs")
            self.rets_ph = tf.placeholder(
                shape=[None, None], dtype=tf.float32, name="rets")
            self.avail_acs_ph = tf.placeholder(
                shape=[None, None, self.num_actions],
                dtype=tf.int32,
                name="avail_acs")
            self.values_ph = tf.placeholder(
                shape=[None, None], dtype=tf.float32, name="target_values")
            self.gamma_ph = tf.placeholder(
                shape=[1, 1], dtype=tf.float32, name="gamma_ph")
            self.discount = tf.cumprod(
                self.gamma_ph * tf.ones_like(self.rets_ph),
                axis=0, exclusive=True, name="discount")
            with tf.variable_scope("policy", reuse=reuse):
                pol_lin = snt.Linear(1, use_bias=False)
                logits = snt.BatchApply(pol_lin)(self.obs_ph)
                pol_params = [pol_lin.w]
                # logits, pol_params = Linear3D(1)(self.obs_ph)
                logits = tf.concat([logits, tf.zeros_like(logits)], -1)
                # Mask out unavailable actions
                # MA: Not sure how that affects the gradients. Maybe better for
                #     the environment to mask out the actions?
                mask = -9999999 * tf.ones_like(logits)
                logits = tf.where(
                    tf.equal(self.avail_acs_ph, 1), x=logits, y=mask)
                # Log probs and actions
                self.log_pi = tf.nn.log_softmax(logits)
                self.acs_onehot = tf.one_hot(
                    self.acs_ph, self.num_actions, dtype=tf.float32)
                self.log_pi_acs = tf.reduce_sum(
                    tf.multiply(self.log_pi, self.acs_onehot), axis=-1)
                self.log_pi_acs_cumsum = tf.cumsum(self.log_pi_acs, axis=0)
                self.action = tf.squeeze(tf.multinomial(
                    tf.reshape(self.log_pi, shape=(-1, self.num_actions)), 1))
            # Value
            with tf.variable_scope("value", reuse=reuse):
                val_lin = snt.Linear(1, use_bias=True)
                self.vpred = snt.BatchApply(val_lin)(self.obs_ph)
                self.vpred = tf.squeeze(self.vpred)
                val_params = [val_lin.w, val_lin.b]
            # Parameters
            self._params += pol_params + val_params


class MLPPolicy(Policy):
    """A feed-forward network with one or multiple hidden layers."""

    def __init__(self, ob_size, num_actions, hidden_sizes=[16], prev=None):
        super(MLPPolicy, self).__init__(ob_size, num_actions, prev=prev)
        self.hidden_sizes = hidden_sizes

    def build(self, scope, reuse=None):
        self.scope = scope
        with tf.variable_scope(scope, reuse=reuse):
            # Placeholders
            self.acs_ph = tf.placeholder(
                shape=[None, None], dtype=tf.int32)
            self.obs_ph = tf.placeholder(
                shape=[None, None, self.ob_size], dtype=tf.float32)
            self.rets_ph = tf.placeholder(
                shape=[None, None], dtype=tf.float32)
            self.avail_acs_ph = tf.placeholder(
                shape=[None, None, self.num_actions], dtype=tf.int32)
            self.values_ph = tf.placeholder(
                shape=[None, None], dtype=tf.float32, name="target_values")
            self.gamma_ph = tf.placeholder(
                shape=[1, 1], dtype=tf.float32, name="gamma_ph")
            self.discount = tf.cumprod(
                self.gamma_ph * tf.ones_like(self.rets_ph),
                axis=0, exclusive=True, name="discount")
            with tf.variable_scope("policy", reuse=reuse):
                # Hidden layers
                pol_params = []
                last = self.obs_ph
                for i, units in enumerate(self.hidden_sizes):
                    pol_lin = snt.Linear(units, name="h_%d" % i)
                    last = snt.BatchApply(pol_lin)(last)
                    last = tf.nn.relu(last)
                    pol_params += [pol_lin.w, pol_lin.b]
                pol_lin = snt.Linear(self.num_actions)
                logits = snt.BatchApply(pol_lin)(last)
                pol_params += [pol_lin.w, pol_lin.b]
                # Mask out unavailable actions
                # MA: Not sure how that affects the gradients. Maybe better for
                    #     the environment to mask out the actions?
                mask = -9999999 * tf.ones_like(logits)
                logits = tf.where(
                    tf.equal(self.avail_acs_ph, 1), x=logits, y=mask)
                # Log probs and actions
                self.log_pi = tf.nn.log_softmax(logits)
                self.acs_onehot = tf.one_hot(
                    self.acs_ph, self.num_actions, dtype=tf.float32)
                self.log_pi_acs = tf.reduce_sum(
                    tf.multiply(self.log_pi, self.acs_onehot), axis=-1)
                self.log_pi_acs_cumsum = tf.cumsum(self.log_pi_acs, axis=0)
                self.action = tf.squeeze(tf.multinomial(
                    tf.reshape(self.log_pi, shape=(-1, self.num_actions)), 1))
            # Value
            with tf.variable_scope("value", reuse=reuse):
                val_params = []
                last = self.obs_ph
                for i, units in enumerate(self.hidden_sizes):
                    val_lin = snt.Linear(units, name="h_%d" % i)
                    last = snt.BatchApply(val_lin)(last)
                    last = tf.nn.relu(last)
                    val_params += [val_lin.w, val_lin.b]
                val_lin = snt.Linear(1)
                self.vpred = snt.BatchApply(val_lin)(last)
                self.vpred = tf.squeeze(self.vpred)
                val_params += [val_lin.w, val_lin.b]
            # Parameters
            self._params += pol_params + val_params


class RecurrentPolicy(Policy):
    """A recurrent network with one or multiple hidden layers."""

    def __init__(self, ob_size, num_actions, hidden_sizes=[16], prev=None):
        super(MLPPolicy, self).__init__(ob_size, num_actions, prev=prev)
        self.hidden_sizes = hidden_sizes

    def build(self, scope, reuse=None):
        self.scope = scope
        with tf.variable_scope(scope, reuse=reuse):
            # Placeholders
            self.acs_ph = tf.placeholder(
                shape=[None, None], dtype=tf.int32)
            self.obs_ph = tf.placeholder(
                shape=[None, None, self.ob_size], dtype=tf.float32)
            self.rets_ph = tf.placeholder(
                shape=[None, None], dtype=tf.float32)
            self.avail_acs_ph = tf.placeholder(
                shape=[None, None, self.num_actions], dtype=tf.int32)
            self.values_ph = tf.placeholder(
                shape=[None, None], dtype=tf.float32, name="target_values")
            self.gamma_ph = tf.placeholder(
                shape=[1, 1], dtype=tf.float32, name="gamma_ph")
            self.discount = tf.cumprod(
                self.gamma_ph * tf.ones_like(self.rets_ph),
                axis=0, exclusive=True, name="discount")
            with tf.variable_scope("policy", reuse=reuse):
                # Hidden layers
                pol_params = []
                last = self.obs_ph
                for i, units in enumerate(self.hidden_sizes):
                    pol_lin = snt.Linear(units, name="h_%d" % i)
                    last = snt.BatchApply(pol_lin)(last)
                    last = tf.nn.relu(last)
                    pol_params += [pol_lin.w, pol_lin.b]
                pol_lin = snt.Linear(self.num_actions)
                logits = snt.BatchApply(pol_lin)(last)
                pol_params += [pol_lin.w, pol_lin.b]
                # Mask out unavailable actions
                # MA: Not sure how that affects the gradients. Maybe better for
                    #     the environment to mask out the actions?
                mask = -9999999 * tf.ones_like(logits)
                logits = tf.where(
                    tf.equal(self.avail_acs_ph, 1), x=logits, y=mask)
                # Log probs and actions
                self.log_pi = tf.nn.log_softmax(logits)
                self.acs_onehot = tf.one_hot(
                    self.acs_ph, self.num_actions, dtype=tf.float32)
                self.log_pi_acs = tf.reduce_sum(
                    tf.multiply(self.log_pi, self.acs_onehot), axis=-1)
                self.log_pi_acs_cumsum = tf.cumsum(self.log_pi_acs, axis=0)
                self.action = tf.squeeze(tf.multinomial(
                    tf.reshape(self.log_pi, shape=(-1, self.num_actions)), 1))
            # Value
            with tf.variable_scope("value", reuse=reuse):
                val_params = []
                last = self.obs_ph
                for i, units in enumerate(self.hidden_sizes):
                    val_lin = snt.Linear(units, name="h_%d" % i)
                    last = snt.BatchApply(val_lin)(last)
                    last = tf.nn.relu(last)
                    val_params += [val_lin.w, val_lin.b]
                val_lin = snt.Linear(1)
                self.vpred = snt.BatchApply(val_lin)(last)
                self.vpred = tf.squeeze(self.vpred)
                val_params += [val_lin.w, val_lin.b]
            # Parameters
            self._params += pol_params + val_params
