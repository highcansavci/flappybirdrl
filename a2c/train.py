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
import cv2

# Enhanced Actor Network
class Actor(nn.Module):
    def __init__(self, input_shape=(3, 256, 256)):
        super(Actor, self).__init__()
        # Extensive convolutional block with multiple layers and pooling
        self.conv_layers = nn.Sequential(
            # First Convolution Block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128x128

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x64

            # Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x32

            # Fourth Convolution Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x16

            # Fifth Convolution Block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8x8
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            test_input = torch.zeros(1, *input_shape)
            conv_out = self.conv_layers(test_input)
            flattened_size = conv_out.view(1, -1).size(1)

        # Extensive fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layer
        self.fc_out = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        action_prob = torch.sigmoid(self.fc_out(x))
        return action_prob


# Enhanced Critic Network
class Critic(nn.Module):
    def __init__(self, input_shape=(3, 256, 256)):
        super(Critic, self).__init__()
        # Extensive convolutional block with multiple layers and pooling
        self.conv_layers = nn.Sequential(
            # First Convolution Block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128x128

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x64

            # Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x32

            # Fourth Convolution Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x16

            # Fifth Convolution Block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8x8
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            test_input = torch.zeros(1, *input_shape)
            conv_out = self.conv_layers(test_input)
            flattened_size = conv_out.view(1, -1).size(1)

        # Extensive fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layer with softplus
        self.fc_out = nn.Linear(512, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        state_value = self.softplus(self.fc_out(x))
        return state_value


# Helper Function: Normalize Observations
def preprocess_observation(observation):
    observation = observation / 255.0  # Normalize RGB values to [0, 1]
    return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def get_observation(screen):
    """
    Generate an observation of the current game state.
    Captures the RGB values of the screen except for the score display,
    and resizes it to (batch_size, 3, 256, 256).
    """
    # Capture the screen's pixel data
    screen_pixels = p.surfarray.array3d(screen.screen)  # Shape: (width, height, 3)

    # Transpose to make it (height, width, 3) for typical image processing
    screen_pixels = np.transpose(screen_pixels, (1, 0, 2))

    # Define the region to exclude (score part)
    observation = screen_pixels[Config.DRAW_TEXT_Y:, :, :]  # Exclude the top `score_height` pixels

    # Resize the observation to (256, 256)
    resized_observation = cv2.resize(observation, (256, 256))  # Shape: (256, 256, 3)

    # Change the shape to (batch_size, 3, 256, 256)
    batch_observation = np.transpose(resized_observation, (2, 0, 1))  # Shape: (batch_size, 3, 256, 256)

    return batch_observation


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

    actor = Actor(input_shape=(3, 256, 256))
    critic = Critic(input_shape=(3, 256, 256))

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
