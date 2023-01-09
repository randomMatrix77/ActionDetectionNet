import action_model
import torch

act = action_model.Actron('./network.yml')
inp = torch.randn(1, 3, 128, 128)
out = act.forward([inp]*5)

# implement training loop here
