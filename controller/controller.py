import random

import pygame as p
from config.config import Config
from model.bird import Bird
from model.pipe import Pipe
from view.bird import BirdView
from view.pipe import PipeView


class Controller:
    def __init__(self):
        self.model = Bird()
        self.bird_view = BirdView(self.model)
        self.pipe_group = p.sprite.Group()
        self.bird_group = p.sprite.Group(self.bird_view)
        self.game_over = False
        self.score = 0
        self.pipe_timer = 0  # Timer to control pipe generation
        self.pass_pipe = False  # Added to track pipe passing

    def reset(self):
        self.model.reset()
        self.pipe_group.empty()
        self.bird_view.rect.x = Config.BIRD_X_COORDINATE
        self.bird_view.rect.y = Config.SCREEN_HEIGHT // 2
        self.game_over = False
        self.score = 0
        self.pipe_timer = 0
        self.pass_pipe = False  # Added to track pipe passing
        return self._get_observation()

    def step(self, action):
        """
        Step through the game environment.
        :param action: Boolean, True if the bird flaps, False otherwise.
        :return: observation, reward, done, info
        """
        # Update model state
        self.model.update(action)

        # Handle pipe creation and updating
        self._generate_pipes()
        self.pipe_group.update()
        self.bird_group.update()

        # Check if the bird passes through the pipe
        self._check_score()

        # Check for collisions
        done = self._check_game_over()
        reward = 1 if not done and self.model.pass_pipe else -1 if done else 0

        if self.model.pass_pipe:
            self.score += 1
            self.model.pass_pipe = False

        return self._get_observation(), reward, done, {"score": self.score}

    def _check_score(self):
        """
        Check if the bird passes through the pipes and update the score.
        """
        if len(self.pipe_group) > 0:
            bird = self.bird_group.sprites()[0]
            pipe = self.pipe_group.sprites()[0]  # The closest pipe to the bird

            # Check if bird is between the pipe's boundaries
            if bird.rect.left > pipe.rect.left and bird.rect.right < pipe.rect.right and not self.pass_pipe:
                self.pass_pipe = True

            # Check if bird has completely passed the pipe
            if self.pass_pipe and bird.rect.left > pipe.rect.right:
                self.score += 1
                self.pass_pipe = False

    def _generate_pipes(self):
        """
        Generate pipes at regular intervals based on the pipe timer.
        """
        self.pipe_timer += 1
        if self.pipe_timer >= Config.PIPE_FREQUENCY:
            # Reset the timer
            self.pipe_timer = 0

            pipe_height = random.randint(-Config.RANDOM_PIPE_HEIGHT, Config.RANDOM_PIPE_HEIGHT)
            btm_pipe = PipeView(Pipe(Config.SCREEN_WIDTH, int(Config.SCREEN_HEIGHT / 2) + pipe_height, -1))
            top_pipe = PipeView(Pipe(Config.SCREEN_WIDTH, int(Config.SCREEN_HEIGHT / 2) + pipe_height, 1))
            self.pipe_group.add(btm_pipe)
            self.pipe_group.add(top_pipe)

    def _check_game_over(self):
        if (
                p.sprite.groupcollide(self.bird_group, self.pipe_group, False, False)
                or self.bird_view.rect.top < 0
                or self.bird_view.rect.bottom >= Config.GROUND_TOP
        ):
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        """
        Generate an observation of the current game state.
        For simplicity, this can include the bird's position, velocity, and the closest pipes.
        """
        closest_pipes = [pipe for pipe in self.pipe_group if pipe.rect.right > self.bird_view.rect.left]
        closest_pipes.sort(key=lambda pipe: pipe.rect.left)

        observation = {
            "bird_y": self.model.y,
            "bird_velocity": self.model.velocity,
            "pipes": [
                {
                    "x": pipe.rect.x,
                    "y": pipe.rect.y,
                    "position": getattr(pipe.model, "position", 0),
                }
                for pipe in closest_pipes[:2]  # Include only the closest two pipes
            ],
        }
        return observation
