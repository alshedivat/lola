"""Run script for LOLA-DiCE on IPD."""

import click

import tensorflow as tf

from lola_dice.envs import IPD
from lola_dice.policy import SimplePolicy, MLPPolicy, RecurrentPolicy
from lola_dice.rpg import train


@click.command()
@click.option("--use-dice/--no-dice", default=True,
              help="Whether to use the DiCE operator in the policy objective.")
@click.option("--use-opp-modeling/--no-opp-modeling", default=False,
              help="Whether to use opponent modeling.")
@click.option("--batch-size", default=64)
@click.option("--epochs", default=200)
@click.option("--runs", default=5)
@click.option("--save-dir", default="results_ipd")
def main(use_dice, use_opp_modeling, epochs, batch_size, runs, save_dir):
    n_agents = 2
    env = IPD(max_steps=150, batch_size=batch_size)

    def make_simple_policy(ob_size, num_actions, prev=None, root=None):
        return SimplePolicy(ob_size, num_actions, prev=prev)

    def make_mlp_policy(ob_size, num_actions, prev=None):
        return MLPPolicy(ob_size, num_actions, hidden_sizes=[64], prev=prev)

    def make_sgd_optimizer(*, lr):
        return tf.train.GradientDescentOptimizer(learning_rate=lr)

    for r in range(runs):
        print("-" * 10, "Run: %d/%d" % (r + 1, runs), "-" * 10)
        train(env, make_simple_policy, make_sgd_optimizer,
              epochs=epochs,
              gamma=.96,
              lr_inner=.1,
              lr_outer=.2,
              lr_value=.1,
              lr_om=.1,
              inner_asymm=True,
              n_agents=n_agents,
              n_inner_steps=2,
              value_batch_size=16,
              value_epochs=0,
              om_batch_size=16,
              om_epochs=0,
              use_baseline=False,
              use_dice=use_dice,
              use_opp_modeling=use_opp_modeling,
              save_dir='%s/run-%d' % (save_dir, r + 1))


if __name__ == '__main__':
    main()
