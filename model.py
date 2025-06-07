import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mod = nn.Sequential( 
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            

            nn.Linear(64, 64),
            nn.SiLU(),

            nn.Linear(64, 64),
            nn.SiLU(),

            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.mod(x)
    
    @torch.no_grad()
    def predict(self, x):
        return self.mod(x).argmax(dim=1)
    
    @torch.no_grad()
    def get_max_q(self, x):
        res = self.forward(x)
        return res.max(dim=1, keepdim=True).values
    
    def get_action_qs(self, x, actions):
        return self.forward(x).gather(index=actions, dim=1)