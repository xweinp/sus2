import torch
from torch import nn, cat
from collections import deque

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mod = nn.Sequential( 
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # First I scale the variavbles:
        # 0th value is in [-4.8, 4.8] (from docs)
        # 1st value is in ~[-5, 5] (empirical)
        # 2nd value is in ~[-0.4, 0.4] (form docs)
        # 3rd value is in ~[-4, 4] (empirical)
        x_s = x.clone()
        x_s[:, 0] = x_s[:, 0] / 4.8
        x_s[:, 1] = x_s[:, 1] / 5.0
        x_s[:, 2] = x_s[:, 2] / 0.4
        x_s[:, 3] = x_s[:, 3] / 4.0
        
        return self.mod(x_s)
    
    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(dim=1)
    
    @torch.no_grad()
    def get_max_q(self, x):
        res = self.forward(x)
        return res.max(dim=1, keepdim=True).values
    
    def get_action_qs(self, x, actions):
        return self.forward(x).gather(index=actions, dim=1)
    

class ReplayMemory():
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)
        self.max_capacity = max_capacity

    def __len__(self):
        return len(self.memory)
    
    def add(self, transition):
        if len(self.memory) == self.max_capacity:
            self.memory.popleft()
            
        self.memory.append(transition)
        
    def sample(self, batch_size):
        ln = len(self.memory)
        indices = torch.randint(0, ln, (batch_size,))

        non_terms = [
            self.memory[i] 
            for i in indices 
            if self.memory[i]['next_state'] is not None
        ]
        terms = [
            self.memory[i] 
            for i in indices 
            if self.memory[i]['next_state'] is None
        ]
        
        return {
            'terms': {
                'state': cat([el['state'] for el in terms]),
                'action': cat([el['action'] for el in terms]),
                'reward': cat([el['reward'] for el in terms]),
            } if len(terms) > 0 else None, 
            'non_terms': {
                'state': cat([el['state'] for el in non_terms]),
                'action': cat([el['action'] for el in non_terms]),
                'reward': cat([el['reward'] for el in non_terms]),
                'next_state': cat([el['next_state'] for el in non_terms]),
            } if len(non_terms) > 0 else None
        }
    

def should_random(eps_greedy):
    return torch.empty(1).uniform_(0.0, 1.0) < eps_greedy
def random_action():
    return torch.randint(low=0, high=2, size=(1,))  # [0, 1]


def dict_to_device(dict, device):
    if device == 'cpu' or dict is None:
        return dict
    return {k: v.to(device) for k, v in dict.items()}

# returns X, y, actions
def terms_batch(terms):
    return terms['state'], terms['reward'], terms['action']

# returns X, y, actions
def non_terms_batch(non_terms, target_model, gamma):
    max_q = target_model.get_max_q(non_terms['next_state'])
    y = non_terms['reward'] + gamma * max_q
    return non_terms['state'], y, non_terms['action']

def torch_state(state, dtype):
    return torch.tensor(
        state,
        dtype=dtype
    ).unsqueeze(0)