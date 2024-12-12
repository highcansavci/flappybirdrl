import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import pygame as p
from config.config import Config

class ImagePrototype:
    IMAGES = dict()
    IMAGES[("environment", "bg")] = p.transform.scale(p.image.load("../assets/images/bg.png"), (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    IMAGES[("environment", "ground")] = p.transform.scale(p.image.load("../assets/images/ground.png"), (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    IMAGES[("bird", "high_wing")] = p.image.load("../assets/images/bird1.png")
    IMAGES[("bird", "moderate_wing")] = p.image.load("../assets/images/bird2.png")
    IMAGES[("bird", "low_wing")] = p.image.load("../assets/images/bird3.png")
    IMAGES[("environment", "pipe")] = p.image.load("../assets/images/pipe.png")

class Image:
    def __init__(self, type_, image_path):
        self.type_ = type_
        self.image_path = image_path

    def get_type(self):
        return self.type_

    def get_image(self):
        return ImagePrototype.IMAGES[self.type_]