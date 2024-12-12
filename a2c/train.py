import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from config.config import Config
from controller.controller import Controller
from view.screen import Screen
import pygame as p
import os


# Actor Network
class Actor(nn.Module):
    def __init__(self, input_shape):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 104 * 111, 512)  # Adjust based on input shape after convs
        self.fc_out = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        action_prob = torch.sigmoid(self.fc_out(x))
        return action_prob


# Critic Network
class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 104 * 111, 512)  # Adjust based on input shape after convs
        self.fc_out = nn.Linear(512, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        state_value = self.softplus(self.fc_out(x))  # Positive values
        return state_value


# Helper Function: Normalize Observations
def preprocess_observation(observation):
    observation = observation / 255.0  # Normalize RGB values to [0, 1]
    observation = np.transpose(observation, (2, 0, 1))  # Change to (C, H, W) for PyTorch
    return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension



def get_observation(screen):
    """
    Generate an observation of the current game state.
    Captures the RGB values of the screen except for the score display.
    """
    # Capture the screen's pixel data
    screen_pixels = p.surfarray.array3d(screen.screen)  # Shape: (width, height, 3)

    # Transpose to make it (height, width, 3) for typical image processing
    screen_pixels = np.transpose(screen_pixels, (1, 0, 2))

    # Define the region to exclude (score part)
    observation = screen_pixels[Config.DRAW_TEXT_Y:, :, :]  # Exclude the top `score_height` pixels

    return observation

# Function to Save Models
def save_models(actor, critic, save_dir, step):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    actor_path = os.path.join(save_dir, f"actor_step_{step}.pt")
    critic_path = os.path.join(save_dir, f"critic_step_{step}.pt")
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print(f"Models saved at step {step}")

# Training Loop
if __name__ == "__main__":
    p.init()
    controller = Controller()
    screen = Screen(controller)

    actor = Actor(input_shape=(3, 864, 916))
    critic = Critic(input_shape=(3, 864, 916))

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    gamma = 0.99  # Discount factor
    max_steps = 100000
    save_interval = 1000  # Save models every 1000 steps
    reward_history = deque(maxlen=100)

    running = True
    for step in range(max_steps):
        # Reset environment
        observation = preprocess_observation(get_observation(screen))
        controller.reset()
        total_reward = 0
        done = False

        # Storage for a single episode
        states, actions, rewards, log_probs, state_values = [], [], [], [], []

        while not done:
            # Get action from actor
            action_prob = actor(observation)
            action = (torch.rand(1).item() < action_prob.item())  # Sample action
            action_tensor = torch.tensor([action], dtype=torch.float32)

            # Store log probability
            log_prob = torch.log(action_prob if action else 1 - action_prob)

            # Interact with environment
            _, reward, done, _ = controller.step(action)
            screen.draw_game()
            next_observation = preprocess_observation(get_observation(screen))

            # Critic estimate
            state_value = critic(observation)

            # Store experience
            states.append(observation)
            actions.append(action_tensor)
            rewards.append(reward)
            log_probs.append(log_prob)
            state_values.append(state_value)

            # Update for next step
            observation = next_observation
            total_reward += reward

            if done:
                controller.reset()

        # Compute Returns and Advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        state_values = torch.cat(state_values)
        advantages = returns - state_values.squeeze()

        # Update Critic
        critic_loss = ((state_values.squeeze() - returns) ** 2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update Actor
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Log progress
        reward_history.append(total_reward)
        print(f"Step: {step}, Total Reward: {total_reward}, Avg Reward: {np.mean(reward_history)}")

        # Save models at intervals
        if step % save_interval == 0 and step > 0:
            save_models(actor, critic, save_dir="models", step=step)

    p.quit()
