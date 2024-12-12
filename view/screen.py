import pygame as p
from config.config import Config
from controller.controller import Controller
from view.image_prototype import ImagePrototype


class Screen:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.screen = p.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self.clock = p.time.Clock()
        self.font = p.font.SysFont("Bauhaus 93", 60)

    def draw_game(self):
        background = ImagePrototype.IMAGES[("environment", "bg")]
        ground = ImagePrototype.IMAGES[("environment", "ground")]
        self.screen.blit(background, (0, 0))
        self.screen.blit(ground, (Config.GROUND_SCROLL, Config.GROUND_TOP))

        Config.GROUND_SCROLL -= Config.SCROLL_SPEED
        if abs(Config.GROUND_SCROLL) > Config.MIN_GROUND_SCROLL:
            Config.GROUND_SCROLL = 0

        self.controller.bird_group.draw(self.screen)
        self.controller.pipe_group.draw(self.screen)
        self._draw_text(str(self.controller.score), Config.WHITE, Config.SCREEN_WIDTH // 2, Config.DRAW_TEXT_Y)

        p.display.update()

    def _draw_text(self, text, text_col, x, y):
        img = self.font.render(text, True, text_col)
        self.screen.blit(img, (x, y))

    def reset_game(self):
        return self.controller.reset()
