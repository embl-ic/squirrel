
class Renderer:

    def __init__(self, world_scale=1.0):
        self.world_scale = world_scale

    def show(self, scene):
        raise NotImplementedError

    def screenshot(self, scene, filename):
        raise NotImplementedError
