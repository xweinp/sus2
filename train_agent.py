import torch
from torch import nn
import gymnasium
from utils import *


def process_batch(
    batch, 
    gamma, 
    target_model,
    policy_model, 
    criterion, 
    optimizer,
    device
):
    policy_model.train()

    non_terms = dict_to_device(batch['non_terms'], device)
    terms = dict_to_device(batch['terms'], device)

    if terms is None:
        x, y, actions = non_terms_batch(non_terms, target_model, gamma)

    elif non_terms is None:
        x, y, actions = terms_batch(terms)
    
    else:
        x_t, y_t, actions_t = terms_batch(terms)
        x_nt, y_nt, actions_nt = non_terms_batch(non_terms, target_model, gamma)
        
        y = torch.cat([y_t, y_nt])
        actions = torch.cat([actions_t, actions_nt])
        x = torch.cat([x_t, x_nt])
    
    
    preds = policy_model.get_action_qs(x, actions)
    loss = criterion(preds, y)

    optimizer.zero_grad()
    loss.backward()
    # I clip the gradient to prevent radical changes in the model
    nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()



def train(
    batch_size = 128,
    learning_rate = 0.001,
    n_episodes = 300,
    eps_greedy = 0.1,
    eps_decay = 0.95,
    gamma = 0.99,
    update_target_steps = 100,
    quality_check_freq = 25,
    hidden_dim = 32
):
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gymnasium.make("CartPole-v1")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    max_reward = 500


    print(f"Using device: {device}")
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")


    target_model = Model(input_dim, hidden_dim, output_dim).to(device)
    policy_model = Model(input_dim, hidden_dim, output_dim).to(device)
    replay_memory = ReplayMemory(int(1e4))

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=0.1, 
        total_iters=n_episodes
    )
    
    episode_lens = []
    sm = 0
    update_it = 0

    for episode in range(1, n_episodes + 1):
        finished = False
        
        state, _ = env.reset()
        state = torch_state(state, dtype)

        episode_len = 0

        while not finished:
            episode_len += 1

            if should_random(eps_greedy):
                action = random_action()
            else:
                policy_model.eval()
                state_c = state.to(device)
                action = policy_model.predict(state_c).to('cpu')
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            next_state = torch_state(next_state, dtype)
            reward = torch.tensor([reward], dtype=dtype).unsqueeze(0)
            action = action.unsqueeze(0)

            finished = finished or terminated or truncated
            if finished:
                next_state = None

            replay_memory.add({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })

            state = next_state

            if len(replay_memory) < batch_size:
                continue
            
            batch = replay_memory.sample(batch_size)
            loss = process_batch(
                batch,
                gamma,
                target_model,
                policy_model,
                criterion,
                optimizer,
                device
            )

            update_it += 1
            if update_it == update_target_steps:
                target_model.load_state_dict(policy_model.state_dict())
                update_it = 0

        episode_lens.append(episode_len)
        sm += episode_len
        
        if len(replay_memory) >= batch_size:
            eps_greedy *= eps_decay
            scheduler.step()
        
        if episode % quality_check_freq == 0:
            sm /= quality_check_freq

            if sm == max_reward:
                print(f"Max reward reached: {max_reward} at episode {episode}")
                break
            print(f"Episode {episode}, Average length: {sm:.2f}")

            sm = 0
    
    env.close()
    torch.save(target_model, 'agent.pt')
    print("Training complete. Model saved to 'agent.pt'.")

if __name__ == "__main__":
    train()