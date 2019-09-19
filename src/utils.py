import funsor
import funsor.distributions as dist
import funsor.ops as ops
import torch


def generate_HMM_dataset(model, args):
    """ Generates a sequence of observations from a given funsor model
    """

    data = [funsor.Variable('y_{}'.format(t), funsor.bint(
        args.hidden_dim)) for t in range(args.time_steps)]

    log_prob = model(data)
    var = [key for key, value in log_prob.inputs.items()]
    # TODO: move sample to model definition, to avoid memory explosion
    r = log_prob.sample(frozenset(var))
    data = torch.tensor([r.deltas[i].point.data for i in range(
        len(r.deltas)) if r.deltas[i].name.startswith('y')])

    return data
