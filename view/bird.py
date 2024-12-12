import pygame.sprite
from config.config import Config
from model.bird import Bird
from view.image_prototype import ImagePrototype


class BirdView(pygame.sprite.Sprite):
    def __init__(self, model: Bird):
        pygame.sprite.Sprite.__init__(self)
        self.model = model
        self.images = [ImagePrototype.IMAGES[("bird", "high_wing")], ImagePrototype.IMAGES[("bird", "moderate_wing")], ImagePrototype.IMAGES[("bird", "low_wing")] ]
        self.index = 0
        self.counter = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [model.x, model.y]

    def update(self, *args, **kwargs):
        if self.rect.bottom < Config.GROUND_TOP:
            self.rect.y += int(self.model.velocity)
        self.counter += 1
        if self.counter > Config.FLAP_COOLDOWN:
            self.counter = 0
            self.index += 1
            self.index %= 3

        self.image = pygame.transform.rotate(self.images[self.index], self.model.velocity * -2)
