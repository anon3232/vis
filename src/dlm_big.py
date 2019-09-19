from __future__ import absolute_import, division, print_function

import argparse
import time

import numpy as np

import funsor
import funsor.distributions as dist
import funsor.ops as ops
import torch
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import lazy
from kalman_utils import next_state, one_step_prediction, update
from relbo import refine
from torch.autograd import grad

# Example call:
# python kalman.py -v -n10 -t5


def main(args):
    # Declare parameters.
    seasonality = 12
    level_noise = torch.tensor(np.log(.1), requires_grad=True)
    slope_noise = torch.tensor(np.log(.1), requires_grad=True)
    emit_noise = torch.tensor(np.log(.5), requires_grad=True)
    trans_noises = [level_noise, slope_noise] + \
        [torch.tensor(np.log(.1), requires_grad=True)
         for _ in range(seasonality)]
    params = [level_noise, slope_noise, emit_noise] + trans_noises

    def linear_trend():
        var_names = ['level', 'slope']
        trans_eqs = {}
        trans_eqs['level'] = lambda var: var[0] + var[1]
        trans_eqs['slope'] = lambda var: var[1]

        def emit_eq(var): return var[0]
        return var_names, trans_eqs, emit_eq

    def linear_trend_seasonal(T=seasonality):
        var_names = ['level', 'slope'] + ['p_' + str(i) for i in range(T)]
        trans_eqs = {}
        trans_eqs['level'] = lambda var: var[0] + var[1]
        trans_eqs['slope'] = lambda var: var[1]
        for i in range(T-1):
            trans_eqs['p_' + str(i)] = lambda var: var[3+i]
        trans_eqs['p_' + str(T-1)] = lambda var: var[2]

        def emit_eq(var): return var[0] + var[2]
        return var_names, trans_eqs, emit_eq

    specification = linear_trend_seasonal
    var_names, trans_eqs, emit_eq = specification()

    # A Gaussian HMM model.

    def model(data):
        log_prob = funsor.to_funsor(0.)
        xs_curr = [funsor.Tensor(torch.tensor(0.)) for var in var_names]

        for t, y in enumerate(data):
            xs_prev = xs_curr

            # A delayed sample statement.
            xs_curr = [funsor.Variable(name + '_{}'.format(t), funsor.reals())
                       for name in var_names]

            for i, x_curr in enumerate(xs_curr):
                log_prob += dist.Normal(trans_eqs[var_names[i]]
                                        (xs_prev), torch.exp(trans_noises[i]), value=x_curr)

            if t > 0:
                log_prob = log_prob.reduce(
                    ops.logaddexp, frozenset([x_prev.name for x_prev in xs_prev]))

            # An observe statement.
            log_prob += dist.Normal(emit_eq(xs_curr),
                                    torch.exp(emit_noise), value=y)

        # Marginalize out all remaining delayed variables.
        return log_prob.reduce(ops.logaddexp), log_prob.gaussian

    # Train model parameters.
    torch.manual_seed(0)

    use_co2_data = True
    if use_co2_data:
        # python dlm_big.py -v -t 10 -lr 0.001 -n10 -r0

        k = 24
        co2_data = np.loadtxt('../data/co2.csv')
        m = np.mean(co2_data)
        s = np.std(co2_data)

        co2_data = (co2_data - m)/s

        ts = 120
        data = co2_data[:ts]
        data_test = co2_data[ts:(ts+k)]
        args.time_steps = len(data)
    else:
        k = 5
        # torch.randn(args.time_steps)
        syn_data = torch.tensor(np.resize([1, 0], args.time_steps + k))
        ts = args.time_steps
        data = syn_data[:ts]
        data_test = syn_data[ts:(ts+k)].detach().numpy()

    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        print(step)
        optim.zero_grad()

        t0 = time.time()
        log_prob, p_x_t = model(data)
        assert not log_prob.inputs, 'free variables remain'
        loss = -log_prob.data
        #loss, p_x_t = refine(loss, params, model, data, p_x_t, args)
        if True:
            lr_inn = args.learning_rate
            for i in range(args.n_ref):
                g_p = grad(loss, params, create_graph=True)

                for i, p in enumerate(params):
                    p.data = p - lr_inn*g_p[i]
                    #params[i] = p

                log_prob, p_x_t = model(data)
        loss = -log_prob.data
        loss.backward()
        optim.step()
        t1 = time.time()
        if args.verbose:
            print('step {} loss = {}'.format(step, loss.item()))
            print('dt ', t1 - t0)
            # print(p_x_t)
            # print(np.linalg.inv(p_x_t.precision.detach().numpy()))

    print(params)

    experiments = True

    if experiments:
        var_names, trans_eqs, emit_eq = specification()
        #emit_eq = lambda xs: xs[0]
        #trans_noises = [level_noise, slope_noise]
        #p_x_tp1 = next_state(p_x_t, args.time_steps-1, var_names, trans_eqs, trans_noises)
        # print(p_x_tp1)

        # p = one_step_prediction(
        #    p_x_tp1, args.time_steps-1, var_names, emit_eq, emit_noise)
        # print(p)
        #p = update(p_x_tp1, args.time_steps-1, -2*data[0], var_names, emit_eq, emit_noise)
        # print(-2*data[0])
        # print(p)

        p_x_t_old = p_x_t

        #print('Sequential 1-step ahead forecasts')
        #new_obs = -2*data[0:4]
        # for i, y in enumerate(new_obs):
        #    p_x_tp1 = next_state(p_x_t, args.time_steps -
        #                         1+i, var_names, trans_eqs, trans_noises)
        #    pred = one_step_prediction(
        #        p_x_tp1, args.time_steps-1+i, var_names, emit_eq, emit_noise)
        #    print(pred)
        #    p_x_t = update(p_x_tp1, args.time_steps -
        #                   1+i, y, var_names, emit_eq, emit_noise)

        print('k-step ahead forecasts')
        errors = np.zeros(k)
        scores = np.zeros(k)
        entropies = np.zeros(k)
        p_x_t = p_x_t_old

        print(np.linalg.inv(p_x_t.precision.detach().numpy()))
        for i in range(k):
            print(i)
            p_x_tp1 = next_state(p_x_t, len(data) -
                                 1+i, var_names, trans_eqs, trans_noises)
            pred = one_step_prediction(
                p_x_tp1, len(data)-1+i, var_names, emit_eq, emit_noise)
            print(pred)   # note decay in precision

            mu_y = pred.gaussian.loc.detach().numpy()
            errors[i] = np.abs(
                mu_y - data_test[i])

            sigma = 1/np.sqrt(pred.gaussian.precision.detach().numpy())

            l = mu_y - 2*sigma
            u = mu_y + 2*sigma

            if data_test[i] > u:
                scores[i] += 2/0.05*(data_test[i] - u)
            if data_test[i] < l:
                scores[i] += 2/0.05*(l - data_test[i])
            scores[i] += u - l

            entropies[i] = np.log(sigma*np.sqrt(2*np.pi*np.e))

            p_x_t = p_x_tp1

        print('MAE ', errors.mean())
        print('entropy', entropies.mean())
        print('score ', scores.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-r", "--n-ref", default=0, type=int)
    parser.add_argument("--lazy", action='store_true')
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
