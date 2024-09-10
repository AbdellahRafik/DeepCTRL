import torch
import torch.nn as nn

class RuleEncoder(nn.Module):
    """
    Neural network module to encode input data based on predefined rules.

    Args:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 6.

    Methods:
        forward(x): Performs the forward pass through the network.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=6):
        super(RuleEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


  
class DataEncoder(nn.Module):
    """
    Neural network module to encode input data.

    Args:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 6.

    Methods:
        forward(x): Performs the forward pass through the network.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=6):
        super(DataEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)

class DataonlyNet(nn.Module):
    """
    Neural network module that uses only data encoding for decision making.

    Args:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        data_encoder (nn.Module): The data encoder module.
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 4.
        n_layers (int, optional): Number of layers. Defaults to 2.
        skip (bool, optional): Whether to skip connections. Defaults to False.
        input_type (str, optional): Type of input, 'state' or 'seq'. Defaults to 'state'.

    Methods:
        get_z(x, alpha): Returns the latent representation from the data encoder.
        forward(x, alpha): Forward pass through the network.
    """
    def __init__(self, input_dim, output_dim, data_encoder, hidden_dim=4, n_layers=2, skip=False, input_type='state'):
        super(DataonlyNet, self).__init__()
        self.skip = skip
        self.input_type = input_type
        self.data_encoder = data_encoder
        self.n_layers = n_layers
        self.input_dim_decision_block = self.data_encoder.output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )

    def get_z(self, x, alpha=0.0):
        return self.data_encoder(x)

    def forward(self, x, alpha=0.0):
        data_z = self.data_encoder(x)
        z = data_z

        if self.skip:
            if self.input_type == 'seq':
                return self.net(z) + x[:,-1,:]
            else:
                return self.net(z) + x
        else:
            return self.net(z)
 

class Net(nn.Module):
    """
    Neural network module combining rule encoding and data encoding.

    Args:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        rule_encoder (nn.Module): The rule encoder module.
        data_encoder (nn.Module): The data encoder module.
        n_layers (int, optional): Number of layers. Defaults to 2.

    Methods:
        get_z(x, alpha): Returns latent representations from both encoders.
        forward(x, alpha): Forward pass through the network with rule and data encoders combined.
    """
    def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, n_layers=2):
        super(Net, self).__init__()
        self.rule_encoder = rule_encoder
        self.data_encoder = data_encoder
        self.n_layers = n_layers
        assert self.rule_encoder.input_dim == self.data_encoder.input_dim
        assert self.rule_encoder.output_dim == self.data_encoder.output_dim
        self.input_dim = self.rule_encoder.output_dim + self.data_encoder.output_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )

    def get_z(self, x, alpha=0):
        rule_z = self.rule_encoder(x)
        data_z = self.data_encoder(x)
        return rule_z, data_z

    def forward(self, x, alpha=0):
        rule_z = self.rule_encoder(x)
        data_z = self.data_encoder(x)
        z = torch.cat((alpha * rule_z, (1 - alpha) * data_z), dim=-1)
        return self.net(z)

   

class SimpleNet(nn.Module):
    """
    A simple feed-forward neural network.

    Args:
        input_dim (int): The input dimension.
        hidden_dim (int): The hidden layer dimension.
        output_dim (int): The output dimension.

    Methods:
        forward(x): Forward pass through the network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 8)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(8, output_dim)

    def forward(self, x, alpha=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
       



