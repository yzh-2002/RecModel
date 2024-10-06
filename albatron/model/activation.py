import torch


class Dice(torch.nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """

    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1

        ps = (x - avg) / torch.sqrt(var)  # N * 1

        ps = torch.nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function

    Returns:
        act_layer: activation layer
    """
    act_layer = None
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = torch.nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = torch.nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            act_layer = Dice()
        elif act_name.lower() == 'prelu':
            act_layer = torch.nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = torch.nn.Softmax(dim=1)
        elif act_name.lower() == 'leakyrelu':
            act_layer = torch.nn.LeakyReLU()
    elif issubclass(act_name, torch.nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer
