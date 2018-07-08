"""The main scripts for running different tournaments."""

import click
import time

from lola import logger

from lola.envs import *


@click.command()
@click.option("--exp_name", type=str, default="IPD",
              help="Name of the experiment (and correspondingly environment).")
@click.option("--num_episodes", type=int, default=None,
              help="Number of episodes.")
@click.option("--trace_length", type=int, default=None,
              help="Lenght of the traces.")
@click.option("--trials", type=int, default=2, help="Number of trials.")
@click.option("--lr", type=float, default=None,
              help="Learning rate for Adam optimizer.")
@click.option("--gamma", type=float, default=None,
              help="Discount factor.")

def main(exp_name, num_episodes, trace_length, trials, lr, gamma):
    # Sanity
    assert exp_name in {"IPD", "IMP"}, "Tournament is only for matrix games."

    # Resolve default parameters
    num_episodes = 50 if num_episodes is None else num_episodes
    trace_length = 200 if trace_length is None else trace_length
    lr = 1. if lr is None else lr

    # Import the right training function
    def run(env, seed):
        from lola.tournament import train
        train(env,
              num_episodes=num_episodes,
              trace_length=trace_length,
              lr=lr,
              gamma=gamma,
              seed=seed)

    # Instantiate the environment
    if exp_name == "IPD":
        env = IPD(trace_length)
        gamma = 0.96 if gamma is None else gamma
    elif exp_name == "IMP":
        env = IMP(trace_length)
        gamma = 0.9 if gamma is None else gamma

    # Run training
    for seed in range(trials):
        logger.reset()
        logger.configure(dir='logs/TNT/{}/seed-{}'.format(exp_name, seed))
        start_time = time.time()
        run(env, seed)
        end_time  = time.time()


if __name__ == '__main__':
    main()
