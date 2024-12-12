from config.config import Config


class Bird:
    def __init__(self):
        self.x = Config.BIRD_X_COORDINATE
        self.y = Config.SCREEN_HEIGHT // 2
        self.velocity = 0
        self.click_interval = 1
        self.pass_pipe = False
        self.game_over = False
        self.score = 0

    def update(self, mouse_pressed):
        self.velocity = max(self.velocity + Config.VELOCITY_INCREASE, Config.MAX_VELOCITY)
        if mouse_pressed and self.click_interval % 3 == 0:
            self.velocity -= Config.JUMP_VELOCITY
        self.click_interval += 1
        self.click_interval %= 3

    def reset(self):
        self.x = Config.BIRD_X_COORDINATE
        self.y = Config.SCREEN_HEIGHT // 2
        self.velocity = 0
        self.click_interval = 1
        self.pass_pipe = False
        self.game_over = False
        self.score = 0
