"""
Policy and value networks used in LOLA experiments.
"""
import copy
import tensorflow as tf
import tensorflow.contrib.layers as layers

from .utils import *


class Pnetwork:
    """
    Recurrent policy network used in Coin Game experiments.
    """
    def __init__(self, myScope, h_size, agent, env, trace_length, batch_size,
                 reuse=None, step=False):
        if step:
            trace_length = 1
        else:
            trace_length = trace_length
        with tf.variable_scope(myScope, reuse=reuse):
            self.batch_size = batch_size
            zero_state = tf.zeros((batch_size, h_size * 2), dtype=tf.float32)
            self.gamma_array = tf.placeholder(
                shape=[1, trace_length],
                dtype=tf.float32,
                name='gamma_array')
            self.gamma_array_inverse = tf.placeholder(
                shape=[1, trace_length],
                dtype=tf.float32,
                name='gamma_array_inv')

            self.lstm_state = tf.placeholder(
                shape=[batch_size, h_size*2], dtype=tf.float32,
                name='lstm_state')

            if step:
                self.state_input =  tf.placeholder(
                    shape=[self.batch_size] + env.ob_space_shape,
                    dtype=tf.float32,
                    name='state_input')
                lstm_state = self.lstm_state
            else:
                self.state_input =  tf.placeholder(
                    shape=[batch_size * trace_length] + env.ob_space_shape,
                    dtype=tf.float32,
                    name='state_input')
                lstm_state = zero_state

            self.sample_return = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name='sample_return')
            self.sample_reward = tf.placeholder(
                shape=[None, trace_length],
                dtype=tf.float32,
                name='sample_reward')
            with tf.variable_scope('input_proc', reuse=reuse):
                output = layers.convolution2d(self.state_input,
                    stride=1, kernel_size=3, num_outputs=20,
                    normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu)
                output = layers.convolution2d(output,
                    stride=1, kernel_size=3, num_outputs=20,
                    normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu)
                output = layers.flatten(output)
                print('values', output.get_shape())
                self.value = tf.reshape(layers.fully_connected(
                    tf.nn.relu(output), 1), [-1, trace_length])
            if step:
                output_seq = batch_to_seq(output, self.batch_size, 1)
            else:
                output_seq = batch_to_seq(output, self.batch_size, trace_length)
            output_seq, state_output = lstm(output_seq, lstm_state,
                                            scope='rnn', nh=h_size)
            output = seq_to_batch(output_seq)

            output = layers.fully_connected(output,
                                            num_outputs=env.NUM_ACTIONS,
                                            activation_fn=None)
            self.log_pi = tf.nn.log_softmax(output)
            self.lstm_state_output = state_output

            self.actions = tf.placeholder(
                shape=[None], dtype=tf.int32, name='actions')
            self.actions_onehot = tf.one_hot(
                self.actions, env.NUM_ACTIONS, dtype=tf.float32)

            predict = tf.multinomial(self.log_pi, 1)
            self.predict = tf.squeeze(predict)

            self.next_value = tf.placeholder(
                shape=[None,1], dtype=tf.float32, name='next_value')
            self.next_v = tf.matmul(self.next_value, self.gamma_array_inverse)
            self.target = self.sample_return + self.next_v

            self.td_error = tf.square(self.target-self.value) / 2
            self.loss = tf.reduce_mean(self.td_error)

        self.parameters = []
        self.value_params = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=myScope):
            if not ('value_params' in i.name):
                self.parameters.append(i)  # i.name if you want just a name
            if 'input_proc' in i.name:
                self.value_params.append(i)

        if not step:
            self.log_pi_action = tf.reduce_mean(tf.multiply(
                self.log_pi, self.actions_onehot), reduction_indices=1)
            self.log_pi_action_bs = tf.reduce_sum(tf.reshape(
                self.log_pi_action, [-1, trace_length]),1)
            self.log_pi_action_bs_t = tf.reshape(
                self.log_pi_action, [self.batch_size, trace_length])
            self.trainer = tf.train.GradientDescentOptimizer(learning_rate=1)
            self.updateModel = self.trainer.minimize(
                self.loss, var_list=self.value_params)

        self.setparams= SetFromFlat(self.parameters)
        self.getparams= GetFlat(self.parameters)
        self.param_len = len(self.parameters)

        for var in self.parameters:
            print(var.name, var.get_shape())


class Qnetwork:
    """
    Simple Q-network used in IPD experiments.
    """
    def __init__(self, myScope, agent, env, batch_size, gamma, trace_length, hidden, simple_net, lr):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        with tf.variable_scope(myScope):
            self.scalarInput =  tf.placeholder(shape=[None, env.NUM_STATES],dtype=tf.float32)
            self.gamma_array = tf.placeholder(shape=[1, trace_length], dtype=tf.float32, name='gamma_array')
            self.gamma_array_inverse = tf.placeholder(shape=[1, trace_length], dtype=tf.float32, name='gamma_array')

            if simple_net:
                self.logit_vals = tf.Variable(tf.random_normal([5,1]))
                self.temp = tf.matmul(self.scalarInput, self.logit_vals)
                temp_concat = tf.concat([self.temp, self.temp * 0], 1)
                self.log_pi = tf.nn.log_softmax(temp_concat)
            else:
                act = tf.nn.relu(layers.fully_connected(self.scalarInput, num_outputs=hidden, activation_fn=None))
                self.log_pi = tf.nn.log_softmax(layers.fully_connected(act, num_outputs=2, activation_fn=None))
            self.values = tf.Variable(tf.random_normal([5,1]), name='value_params')
            self.value = tf.reshape(tf.matmul(self.scalarInput, self.values), [batch_size, -1])
            self.sample_return = tf.placeholder(shape=[None, trace_length],dtype=tf.float32, name='sample_return')
            self.sample_reward = tf.placeholder(shape=[None, trace_length], dtype=tf.float32, name='sample_reward_new')

            self.next_value = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name='next_value')
            self.next_v = tf.matmul(self.next_value, self.gamma_array_inverse)
            self.target = self.sample_return + self.next_v
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, env.NUM_ACTIONS, dtype=tf.float32)

            self.predict = tf.multinomial(self.log_pi ,1)
            self.predict = tf.squeeze(self.predict)
            self.log_pi_action = tf.reduce_mean(
                tf.multiply(self.log_pi, self.actions_onehot),
                reduction_indices=1)

            self.td_error = tf.square(self.target - self.value) / 2
            self.loss = tf.reduce_mean(self.td_error)

        self.parameters = []
        self.value_params = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=myScope):
            if not ('value_params' in i.name):
                self.parameters.append(i)   # i.name if you want just a name
            else:
                self.value_params.append(i)

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=1)# / arglist.bs)
        self.updateModel = self.trainer.minimize(self.loss, var_list=self.value_params)

        self.log_pi_action_bs = tf.reduce_sum(tf.reshape(self.log_pi_action, [-1, trace_length]),1)
        self.log_pi_action_bs_t = tf.reshape(self.log_pi_action, [batch_size, trace_length])
        self.setparams= SetFromFlat(self.parameters)
        self.getparams= GetFlat(self.parameters)


class ExperienceBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self,batch_size,trace_length):
        sampled_episodes = self.buffer
        sampledTraces = []
        for episode in sampled_episodes:
            this_episode = list(copy.deepcopy(episode))
            point = np.random.randint(0,len(this_episode)+1-trace_length)
            sampledTraces.append(this_episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,6])
