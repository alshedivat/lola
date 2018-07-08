"""Recursive Policy Gradients."""

import os
import sys
import numpy as np
import tensorflow as tf

from . import utils as U
from .envs import IPD, IMP
from .meta import make_with_custom_variables


def stop_forward(x):
    """Implements the Magic Box operator."""
    with tf.name_scope("stop_forward"):
        op = tf.exp(x - tf.stop_gradient(x))
    return op


def build_policy(env, make_policy, scope, reuse=None, prev=None):
    """Creates and builds a new policy."""
    pi = make_policy(env.NUM_STATES, env.NUM_ACTIONS, prev=prev)
    pi.build(scope, reuse=reuse)
    return pi


def build_losses(scope, policies, use_baseline=True, use_dice=True):
    """Builds policy and value loss tensors for the given policies.

    Args:
        scope (str): The name of the scope to use for internals.
        policies (list): A list of Policy objects. Assumed to be built.
        use_baseline (bool): A flag for whether to use a baseline for PG.
        use_dice (bool): Whether to use the DiCE operator.

    Returns:
        policy_losses: list of <float32> [] tensors for policy losses.
        value_losses: list of <float32> [] tensors for value function losses.
    """
    with tf.name_scope(scope):
        # Build return weights.
        if use_dice:
            ret_weights = stop_forward(sum(
                pi.log_pi_acs_cumsum for pi in policies))
        else:
            ret_weights = sum(
                pi.log_pi_acs_cumsum for pi in policies)

        # Build policy losses.
        if use_baseline:
            with tf.name_scope("baseline"):
                if use_dice:
                    baseline_weights = 1 - stop_forward(sum(
                        pi.log_pi_acs for pi in policies))
                else:
                    baseline_weights = -sum(
                        pi.log_pi_acs for pi in policies)
                baselines = [
                    tf.multiply(tf.stop_gradient(pi.vpred), pi.discount)
                    for pi in policies]
            policy_losses = [
                -tf.reduce_mean(tf.reduce_sum(
                    tf.multiply(pi.rets_ph, ret_weights), axis=0))
                -tf.reduce_mean(tf.reduce_sum(
                    tf.multiply(b, baseline_weights), axis=0))
                for pi, b in zip(policies, baselines)]
        else:
            rets = [
                pi.rets_ph - tf.reduce_mean(pi.rets_ph, axis=1, keepdims=True)
                for pi in policies]
            policy_losses = [
                -tf.reduce_mean(tf.reduce_sum(
                    tf.multiply(ret, ret_weights), axis=0))
                for pi, ret in zip(policies, rets)]

        # Build value function losses.
        value_losses = [
            tf.reduce_mean(tf.square(pi.vpred - pi.values_ph))
            for pi in policies]

    return policy_losses, value_losses


def build_grads(scope, losses, params):
    """Builds gradients of the loss functions w.r.t. parameters.

    Args:
        scope (str): The name of the scope to use for internals.
        losses (list): A list of loss tensors.
        params (list): A list of (lists of) tf.Variables.

    Returns:
        grads: A list of gradient tensors of the losses w.r.t. params.
    """
    assert len(losses) == len(params)
    with tf.name_scope(scope):
        grads = [
            tf.gradients(loss, param)
            for loss, param in zip(losses, params)]
    return grads


def build_new_params(scope, policies, k, *,
                     lr,
                     asymm=True,
                     use_baseline=True,
                     use_dice=True):
    """Builds new parameters for each policy performing a MAML-like update.

    To understand how this works, consider 3 policies with parameters
    `old_params_1`, `old_params_2`, `old_params_3`. If `k == 1` and
    `asymm == True`, we have:

        new_params_1 = old_params_1
        new_params_2 = old_params_2 - lr * grad loss_2
        new_params_3 = old_params_3 - lr * grad loss_3

    If `asymm == False`, `new_params_1` will be also updated. The asymmetric
    updates are used as lookahead steps performed by the LOLA agents. In the
    given example, agent 1 "imagines" other agents do gradient updates with the
    specified learning rate (lr); it will then backpropagate through these
    updates and update its own parameters respectively.

    Args:
        scope (str): The name of the scope to use for internals.
        policies (list): A list of Policy objects. Assumed to be built.
        k (int): The policy index which parameters are NOT updated but instead
            copied over.
        asymm (bool): Whether to perform symmetric or asymmetric update.
        use_baseline (bool): A flag for whether to use a baseline for PG.
        use_dice (bool): Whether to use the DiCE operator.
    """
    with tf.name_scope(scope):
        # Build inner losses.
        policy_losses, value_losses = build_losses(
            None, policies, use_baseline=use_baseline, use_dice=use_dice)
        losses = policy_losses
        # losses = [pl + vl for pl, vl in zip(policy_losses, value_losses)]
        params = [pi.parameters for pi in policies]

        # Build gradients.
        grads = build_grads(None, losses, params)

        # Build new parameters.
        new_params = []
        for i, (pi, grad) in enumerate(zip(policies, grads)):
            if (i != k) or (not asymm):
                new_p = [
                    (p - lr * g) if g is not None else p
                    for p, g in zip(pi.parameters, grad)]
            else:
                new_p = pi.root.parameters
            new_params.append(new_p)

    return new_params


def get_update(policies, losses, update_ops, sess, gamma=.96):
    """Creates an update function.

    Args:
        policies (list): A list of Policy objects. Assumed to be built.
            Used to construct the `feed_dict` for the `sess.run`.
        losses (list): A list of <float32> [] tensors values for which will be
            computed and returned.
        update_ops (list): A list of update ops.
        sess (tf.Session): A tf.Session instance that will be used for running.
        gamma (float): The discount factor that will be fed into the graph.
            TODO: perhaps move this argument somewhere else?
    """
    def update(traces, *, parent_traces=[]):
        feed_list = sum([
            pi.get_feed_list(trace) + [(pi.gamma_ph, [[gamma]])]
            for pi, trace in zip(policies, traces)
        ], [])
        # Construct the parent feed list.
        parent_policies = zip(*[pi.parents for pi in policies])
        parent_feed_list = sum([
            pi.get_feed_list(trace) + [(pi.gamma_ph, [[gamma]])]
            for parents, traces in zip(parent_policies, parent_traces)
            for pi, trace in zip(parents, traces)
        ], [])
        # Compute.
        feed_dict = dict(feed_list + parent_feed_list)
        results = sess.run(losses + update_ops, feed_dict=feed_dict)
        return results[:len(losses)]
    return update


def compute_values(rews, last_vpreds, *, gamma, use_gae=False):
    """Compute the estimated values for the given sequence of rewards."""
    # TODO: add GAE as an option.
    T = len(rews)
    values = [last_vpreds]
    for t in reversed(range(T)):
        values_t = [gamma * v + r for v, r in zip(values[-1], rews[t])]
        values.append(values_t)
    return list(reversed(values[1:]))


def rollout(env, policies, rollout_policies, sess, *, gamma, parent_traces=[]):
    """Rolls out a single episode of the policies in the given environment.

    To avoid quadratic time complexity of the rollout in the number time steps
    for the recursively generated policies, we never use their graphs directly
    for rollouts. Instead, we copy the values of the policy parameters into the
    corresponding rollout policies and run those in the environment.

    Args:
        env (gym.Env): An instance of the environment.
        policies (list): A list of Policy objects. Assumed to be built.
        rollout_policies (list): Another set of policies which parameters are
            plain variables (not function of other policies and rollouts).
        sess (tf.Session): A tf.Session instance that will be used for running.
        gamma (float): The discount factor that will be fed into the graph.
        parent_traces (list): A list of traces that are fed into the
            corresponding placeholders if the parameters of the policies depend
            on other (parent) policies.

    Returns:
        trace: A list of obs, acs, rets, values, infos.
    """
    obs, acs, rets, rews, values, infos = [], [], [], [], [], []

    # Construct the parent feed list.
    parent_policies = zip(*[pi.parents for pi in policies])
    parent_feed_list = sum([
        pi.get_feed_list(trace) + [(pi.gamma_ph, [[gamma]])]
        for parents, traces in zip(parent_policies, parent_traces)
        for pi, trace in zip(parents, traces)], [])

    # Cache parameters and push them into rollout policies.
    assign_ops = [
        tf.assign(pr, p)
        for pi, pi_roll in zip(policies, rollout_policies)
        for p, pr in zip(pi.parameters, pi_roll.parameters)]
    sess.run(assign_ops, feed_dict=dict(parent_feed_list))

    # Roll out
    t = 0
    ob, info = env.reset()
    done = False
    gamma_t = 1.

    while not done:
        obs.append(ob)
        infos.append(info)

        ac = [
            pi.act(o, i, sess)
            for pi, o, i in zip(rollout_policies, ob, info)
        ]

        ob, rew, done, info = env.step(ac)
        acs.append(ac)
        rews.append(rew)
        rets.append([r * gamma_t for r in rew])
        gamma_t *= gamma
        t += 1

    # Adjust rets and compute value estimates
    last_vpreds =  [
        pi.predict(o, sess) * 0
        for pi, o in zip(rollout_policies, ob)]
    # for k, last_vpred in enumerate(last_vpreds):
    #     rets[-1][k] += gamma_t * last_vpred
    values = compute_values(rews, last_vpreds, gamma=gamma)

    obs = list(map(np.asarray, zip(*obs)))
    acs = list(map(np.asarray, zip(*acs)))
    rets = list(map(np.asarray, zip(*rets)))
    values = list(map(np.asarray, zip(*values)))
    infos = list(map(np.asarray, zip(*infos)))
    trace = list(zip(obs, acs, rets, values, infos))

    return trace


def gen_trace_batches(trace, *, batch_size):
    """Splits the trace and yields batches."""
    obs, acs, rets, values, infos = zip(*trace)
    permutation = np.random.permutation(len(obs[0]))
    for i in range(0, len(obs[0]), batch_size):
        idx = permutation[i:i+batch_size]
        trace_batch = list(zip(
            [ob[idx] for ob in obs],
            [ac[idx] for ac in acs],
            [ret[idx] for ret in rets],
            [val[idx] for val in values],
            [info[idx] for info in infos]))
        yield trace_batch


def build_graph(env, make_policy, make_optimizer, *,
                lr_inner=1.,          # lr for the inner loop steps
                lr_outer=1.,          # lr for the outer loop steps
                lr_value=.1,          # lr for the value function estimator
                lr_om=1.,             # lr for opponent modeling
                n_agents=2,
                n_inner_steps=1,
                inner_asymm=True,
                use_baseline=True,
                use_dice=True,
                use_opp_modeling=False):
    """Builds all components of the graph."""
    # Root policies.
    print("Building root policies...", end=""); sys.stdout.flush()
    root_policies = []
    for k in range(n_agents):
        pi = build_policy(env, make_policy, "root/pi_%d" % k)
        root_policies.append(pi)
    print("Done.")

    # Opponent models.
    if use_opp_modeling:
        for k, pi in enumerate(root_policies):
            pi.opponents = [
                build_policy(env, make_policy, "root/pi_%d/opp_%d" % (k, j))
                for j in range(n_agents - 1)]
    else:
        for k, pi in enumerate(root_policies):
            pi.opponents = [
                make_with_custom_variables(
                    lambda: build_policy(
                        env, make_policy,
                        "root/pi_%d/opp_%d" % (k, j - (j > k))),
                    opp.parameters)
                for j, opp in enumerate(root_policies) if j != k]

    # Rollout policies (used to speed up rollouts).
    print("Building rollout policies...", end=""); sys.stdout.flush()
    rollout_policies = []
    for k in range(n_agents):
        pi = build_policy(env, make_policy, "rollout/pi_%d" % k)
        rollout_policies.append(pi)
    print("Done.")

    # Build asymmetric inner loops recursively.
    print("Building asymmetric inner loops...", end=""); sys.stdout.flush()
    policies = root_policies
    for m in range(n_inner_steps):
        new_policies = []
        for k in range(n_agents):
            # Build new parameters.
            new_params, *new_params_opp = build_new_params(
                "inner_%d/params_%d" % (m + 1, k),
                [policies[k]] + policies[k].opponents, 0,
                lr=lr_inner,
                asymm=True,
                use_baseline=use_baseline,
                use_dice=use_dice)
            # Build new policy and opponents.
            new_policy = make_with_custom_variables(
                lambda: build_policy(
                    env, make_policy, "inner_%d/pi_%d" % (m + 1, k),
                    prev=policies[k]),
                new_params)
            new_policy.opponents = [
                make_with_custom_variables(
                    lambda: build_policy(
                        env, make_policy,
                        "inner_%d/pi_%d/opp_%d" % (m + 1, k, i),
                        prev=prev_opp),
                    opp_params)
                for i, (opp_params, prev_opp) in
                enumerate(zip(new_params_opp, policies[k].opponents))
            ]
            new_policies.append(new_policy)
        policies = new_policies
        print("%d..." % (m + 1), end=""); sys.stdout.flush()
    print("Done.")

    # Build the outer loop.
    print("Building the outer loop...", end=""); sys.stdout.flush()
    pol_losses, val_losses = [], []
    update_pol_ops, update_val_ops = [], []
    for k in range(n_agents):
        params = policies[k].root.parameters
        pol_loss, val_loss = build_losses(
            "outer_%d" % k,
            [policies[k]] + policies[k].opponents,
            use_baseline=use_baseline,
            use_dice=use_dice)
        pol_losses.append([pol_loss[0]])
        val_losses.append([val_loss[0]])
        opt_pol = make_optimizer(lr=lr_outer)
        opt_val = make_optimizer(lr=lr_value)
        upd_pol = [opt_pol.minimize(pol_loss[0], var_list=params)]
        upd_val = [opt_val.minimize(val_loss[0], var_list=params)]
        update_pol_ops.append(upd_pol)
        update_val_ops.append(upd_val)
    print("Done.")

    # Build opponent modeling.
    om_losses = []
    update_om_ops = []
    if use_opp_modeling:
        for k in range(n_agents):
            opp_models = policies[k].root.opponents
            true_opponents = [
                pi.root for j, pi in enumerate(policies) if j != k]
            losses = [-tf.reduce_mean(opp.log_pi_acs) for opp in opp_models]
            params = [opp.parameters for opp in opp_models]
            opts = [make_optimizer(lr=lr_om)
                    for opp in policies[k].root.opponents]
            upds = [
                opt.minimize(loss, var_list=param)
                for opt, loss, param in zip(opts, losses, params)]
            om_losses.append(losses)
            update_om_ops.append(upds)

    return (
        policies, rollout_policies, pol_losses, val_losses, om_losses,
        update_pol_ops, update_val_ops, update_om_ops)


def train(env, make_policy, make_optimizer, *,
          epochs=100,
          gamma=.96,
          lr_inner=1.,          # lr for the inner loop steps
          lr_outer=1.,          # lr for the outer loop steps
          lr_value=.1,          # lr for the value function estimator
          lr_om=.1,             # lr for opponent modeling
          n_agents=2,
          n_inner_steps=1,
          inner_asymm=True,
          om_batch_size=64,     # batch size used for fitting opponent models
          om_epochs=5,          # epochs per iteration to fit opponent models
          value_batch_size=64,  # batch size used for fitting the values
          value_epochs=5,       # epochs per iteration to fit value functions
          use_baseline=True,
          use_dice=True,
          use_opp_modeling=False,
          save_dir='.'):
    """The main training function."""
    os.makedirs(save_dir, exist_ok=True)

    # Build.
    tf.reset_default_graph()

    (policies, rollout_policies, pol_losses, val_losses, om_losses,
     update_pol_ops, update_val_ops, update_om_ops) = build_graph(
        env, make_policy, make_optimizer,
        lr_inner=lr_inner, lr_outer=lr_outer, lr_value=lr_value, lr_om=lr_om,
        n_agents=n_agents, n_inner_steps=n_inner_steps,
        use_baseline=use_baseline, use_dice=use_dice,
        use_opp_modeling=use_opp_modeling)

    # Train.
    acs_all = []
    rets_all = []
    params_all = []
    params_om_all = []
    times_all = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Construct update functions.
        update_funcs = {
            'policy': [
                get_update(
                    [policies[k]] + policies[k].opponents,
                    pol_losses[k], update_pol_ops[k], sess,
                    gamma=gamma)
                for k in range(n_agents)],
            'value': [
                get_update(
                    [policies[k]],
                    val_losses[k], update_val_ops[k], sess,
                    gamma=gamma)
                for k in range(n_agents)],
            'opp': [
                get_update(
                    policies[k].root.opponents,
                    om_losses[k], update_om_ops[k], sess,
                    gamma=gamma)
                for k in range(n_agents)
            ] if om_losses else None,
        }

        root_policies = [pi.root for pi in policies]

        # Train for a number of epochs.
        for e in range(epochs):
            times = []

            # Model opponents.
            if use_opp_modeling:
                with U.elapsed_timer() as om_timer:
                    # Fit opponent models for several epochs.
                    om_losses = np.zeros((n_agents, n_agents - 1))
                    for om_ep in range(om_epochs):
                        traces = rollout(
                            env, root_policies, rollout_policies, sess,
                            gamma=gamma, parent_traces=[])
                        om_traces = [
                            [tr for j, tr in enumerate(traces) if j != k]
                            for k in range(n_agents)]
                        for k in range(n_agents):
                            update_om = update_funcs['opp'][k]
                            for trace_batch in gen_trace_batches(
                                    om_traces[k], batch_size=om_batch_size):
                                update_om(trace_batch)
                            loss = update_om(om_traces[k])
                            om_losses[k] += np.asarray(loss)
                    om_losses /= om_epochs
                times.append(om_timer())
            else:
                om_losses = np.array([])

            # Fit value functions.
            with U.elapsed_timer() as val_timer:
                # Fit value functions for several epochs.
                value_losses = np.zeros(n_agents)
                for v_ep in range(value_epochs):
                    traces = rollout(
                        env, root_policies, rollout_policies, sess,
                        gamma=gamma, parent_traces=[])
                    for k in range(n_agents):
                        update_val = update_funcs['value'][k]
                        for trace_batch in gen_trace_batches(
                                [traces[k]], batch_size=value_batch_size):
                            update_val(trace_batch)
                        loss = update_val([traces[k]])
                        value_losses[k] += loss[0]
                    value_losses /= value_epochs
            times.append(val_timer())

            # Save parameters of the agents (for debug purposes).
            params = sess.run([
                tf.squeeze(pi.root.parameters[0])
                for pi in policies])
            params_all.append(params)

            # Save parameters of the opponent models (for debug purposes).
            params = [
                sess.run([
                    tf.squeeze(opp.root.parameters[0])
                    for opp in pi.opponents])
                for pi in policies]
            params_om_all.append(params)

            # Inner loop rollouts (lookahead steps).
            with U.elapsed_timer() as inner_timer:
                inner_traces = []
                for k in range(n_agents):
                    parent_traces = []
                    for m in range(n_inner_steps):
                        policies_k = [policies[k].parents[m]] + [
                            opp.parents[m] for opp in policies[k].opponents]
                        traces = rollout(
                            env, policies_k, rollout_policies, sess,
                            gamma=gamma, parent_traces=parent_traces)
                        parent_traces.append(traces)
                    inner_traces.append(parent_traces)
            times.append(inner_timer())

            # Outer loop rollouts (each agent plays against updated opponents).
            with U.elapsed_timer() as outer_timer:
                outer_traces = []
                for k in range(n_agents):
                    parent_traces = inner_traces[k]
                    policies_k = [policies[k]] + policies[k].opponents
                    traces = rollout(
                        env, policies_k, rollout_policies, sess,
                        gamma=gamma, parent_traces=parent_traces)
                    outer_traces.append(traces)
            times.append(outer_timer())

            # Updates.
            update_time = 0
            policy_losses = []
            for k in range(n_agents):
                # Policy
                with U.elapsed_timer() as pol_upd_timer:
                    parent_traces = inner_traces[k]
                    update_pol = update_funcs['policy'][k]
                    loss = update_pol(
                        outer_traces[k], parent_traces=parent_traces)
                    policy_losses.append(loss)
                update_time += pol_upd_timer()

            # Logging.
            if n_inner_steps > 0:
                obs, acs, rets, vals, infos = list(zip(*inner_traces[0][0]))
            else:
                obs, acs, rets, vals, infos = list(zip(*outer_traces[0]))
            times_all.append(times)
            acs_all.append([ac.mean() for ac in acs])

            rets_all.append([r.sum(axis=0).mean() * (1 - gamma) for r in rets])
            # rets_all.append([r.sum(axis=0).mean() for r in rets])
            print("Epoch:", e + 1, '-' * 60)
            # print("Policy losses:", list(map(sum, policy_losses)))
            print("Value losses:", value_losses.tolist())
            print("OM losses:", om_losses.tolist())
            print("Returns:", rets_all[-1])
            print("Defection rate:", acs_all[-1])

            # Save stuff
            np.save(save_dir + '/acs.npy', acs_all)
            np.save(save_dir + '/rets.npy', rets_all)
            np.save(save_dir + '/params.npy', params_all)
            np.save(save_dir + '/params_om.npy', params_om_all)
            np.save(save_dir + '/times.npy', times_all)
