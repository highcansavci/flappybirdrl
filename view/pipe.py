import pygame as p

from config.config import Config
from view.image_prototype import ImagePrototype


class PipeView(p.sprite.Sprite):
    def __init__(self, model):
        p.sprite.Sprite.__init__(self)
        self.model = model
        self.image = ImagePrototype.IMAGES[("environment", "pipe")]
        self.rect = self.image.get_rect()
        if model.position == -1:
            self.image = p.transform.flip(self.image, False, True)
            self.rect.bottomleft = [model.x, model.y - Config.PIPE_GAP // 2]
        if model.position == 1:
            self.rect.topleft = [model.x, model.y + Config.PIPE_GAP // 2]

    def update(self, *args, **kwargs):
        self.rect.x -= Config.SCROLL_SPEED
        if self.rect.right < 0:
            self.kill()