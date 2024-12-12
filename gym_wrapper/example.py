import numpy as np

from config.config import Config
from controller.controller import Controller
import pygame as p

from view.screen import Screen


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


if __name__ == "__main__":
    p.init()
    controller = Controller()
    screen = Screen(controller)

    running = True
    while running:
        for event in p.event.get():
            if event.type == p.QUIT:
                running = False

        action = np.random.choice([True, False])  # Replace with user input or AI agent
        _, reward, done, info = controller.step(action)
        screen.draw_game()
        obs = get_observation(screen)

        if done:
            controller.reset()

        screen.clock.tick(Config.SCREEN_FPS)

    p.quit()

