
from torch.autograd import grad

def refine(original_loss, params, model, data, p_x_t, args):

    loss = original_loss
    lr_inn = args.learning_rate

    for i in range(args.n_ref):
        g_p = grad(loss, params, create_graph=True)

        for i, p in enumerate(params):
            p.data = p - lr_inn*g_p[i]

        log_prob, p_x_t = model(data)
        loss = -log_prob.data
    return loss, p_x_t