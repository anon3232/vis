"""
A set of functions for computing distributions in the kalman filter
"""
import torch 

import funsor
import funsor.distributions as dist
import funsor.ops as ops


def next_state(p_x_t, t, var_names, trans_eqs, trans_noises):
    """Computes p(x_{t+1}) from p(x_t)"""
    log_prob = p_x_t

    x_ts = [funsor.Variable(name + '_{}'.format(t), funsor.reals())
            for name in var_names]
    x_tp1s = [funsor.Variable(name + '_{}'.format(t+1),
                              funsor.reals()) for name in var_names]

    for i, x_tp1 in enumerate(x_tp1s):
        log_prob += dist.Normal(trans_eqs[var_names[i]]
                                (x_ts), torch.exp(trans_noises[i]), value=x_tp1)

    log_prob = log_prob.reduce(
        ops.logaddexp, frozenset([x_t.name for x_t in x_ts]))
    return log_prob


def one_step_prediction(p_x_tp1, t,  var_names, emit_eq, emit_noise):
    """Computes p(y_{t+1}) from p(x_{t+1}). We assume y_t is scalar, so only one emit_eq"""
    log_prob = p_x_tp1

    x_tp1s = [funsor.Variable(name + '_{}'.format(t+1),
                              funsor.reals()) for name in var_names]
    y_tp1 = funsor.Variable('y_{}'.format(t+1), funsor.reals())
    log_prob += dist.Normal(emit_eq(x_tp1s), torch.exp(emit_noise), value=y_tp1)
    log_prob = log_prob.reduce(ops.logaddexp, frozenset(
        [x_tp1.name for x_tp1 in x_tp1s]))

    return log_prob


def update(p_x_tp1, t, y, var_names, emit_eq, emit_noise):
    """Computes p(x_{t+1} | y_{t+1}) from p(x_{t+1}). This is useful for iterating 1-step ahead predictions"""
    log_prob = p_x_tp1

    x_tp1s = [funsor.Variable(name + '_{}'.format(t+1),
                              funsor.reals()) for name in var_names]
    log_p_x = log_prob

    log_prob += dist.Normal(emit_eq(x_tp1s), emit_noise, value=y)
    log_p_y = log_prob.reduce(ops.logaddexp, frozenset(
        [x_tp1.name for x_tp1 in x_tp1s]))

    log_p_x_y = log_prob + log_p_x - log_p_y
    return log_p_x_y
