from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import lazy

from kalman_utils import next_state, one_step_prediction, update

# Example call:
# python kalman.py -v -n10 -t5
# Deprecated! See dlm_big.py instead


def main(args):
    # Declare parameters.
    trans_noise = torch.tensor(0.2, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    def trans_eq(x): return x

    def emit_eq(x): return x

    # A Gaussian HMM model.
    def model(data):
        log_prob = funsor.to_funsor(0.)

        x_curr = funsor.Tensor(torch.tensor(0.))
        for t, y in enumerate(data):
            x_prev = x_curr

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals())
            log_prob += dist.Normal(trans_eq(x_prev),
                                    trans_noise, value=x_curr)

            # Optionally marginalize out the previous state.
            if t > 0:
                log_prob = log_prob.reduce(ops.logaddexp, x_prev.name)

            # An observe statement.
            log_prob += dist.Normal(emit_eq(x_curr), emit_noise, value=y)

        # Marginalize out all remaining delayed variables.
        return log_prob.reduce(ops.logaddexp), log_prob.gaussian

    # Train model parameters.
    torch.manual_seed(2)
    data = torch.randn(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        log_prob, p_x_t = model(data)
        assert not log_prob.inputs, 'free variables remain'
        loss = -log_prob.data
        loss.backward()
        optim.step()
        if args.verbose and step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))
            print(p_x_t)

    print(params)

    p_x_tp1 = next_state(p_x_t, args.time_steps-1, trans_eq, trans_noise)
    p = one_step_prediction(p_x_tp1, args.time_steps-1, emit_eq, emit_noise)
    print(p)
    p = update(p_x_tp1, args.time_steps-1, -2*data[0], emit_eq, emit_noise)
    print(-2*data[0])
    print(p)

    p_x_t_old = p_x_t

    print('Sequential 1-step ahead forecasts')
    new_obs = -2*data[0:4]
    for i, y in enumerate(new_obs):
        p_x_tp1 = next_state(p_x_t, args.time_steps-1+i, trans_eq, trans_noise)
        pred = one_step_prediction(
            p_x_tp1, args.time_steps-1+i, emit_eq, emit_noise)
        print(pred)
        p_x_t = update(p_x_tp1, args.time_steps-1+i, y, emit_eq, emit_noise)

    print('k-step ahead forecasts')
    k = 10
    p_x_t = p_x_t_old
    for i in range(k):
        p_x_tp1 = next_state(p_x_t, args.time_steps-1+i, trans_eq, trans_noise)
        pred = one_step_prediction(
            p_x_tp1, args.time_steps-1+i, emit_eq, emit_noise)
        print(pred)   # note decay in precision
        p_x_t = p_x_tp1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--lazy", action='store_true')
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
