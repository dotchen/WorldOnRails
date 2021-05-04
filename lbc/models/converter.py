import numpy as np
import torch

class Converter(torch.nn.Module):
    def __init__(
        self, w=480, h=224, fov=60,
        map_size=96, pixels_per_meter=3,
        offset=2.0, scale=[1.0, 10.0], cam_height=1.4
        ):

        super().__init__()

        F = w / (2 * np.tan(fov * np.pi / 360))

        self.map_size = map_size
        self.pixels_per_meter = pixels_per_meter
        self.w = w
        self.h = h
        self.fy = F
        self.fx = 1.1 * F
        self.offset = offset
        self.cam_height = cam_height
        self.scale = torch.nn.Parameter(torch.tensor(scale).float(), requires_grad=False)
        self.position = torch.nn.Parameter(torch.tensor([map_size//2, map_size//2]).float(), requires_grad=False)

    def forward(self, map_coords):
        return self.map_to_cam(map_coords)

    def map_to_cam(self, map_coords):
        world_coords = self.map_to_world(map_coords)
        cam_coords = self.world_to_cam(world_coords)

        return cam_coords

    def cam_to_world(self, points):

        z = (self.fy * self.cam_height) / torch.clamp(points[..., 1] - self.h / 2, self.offset, 100)
        x = (points[..., 0] - self.w / 2) * (z / self.fx)
        y = z - self.offset

        result = torch.stack([y, x], points.ndim-1) / self.scale
        result = result.reshape(*points.shape) * self.pixels_per_meter

        return result

    def world_to_cam(self, world):
        world = world / (self.pixels_per_meter) * self.scale
        z = world[..., 0] + self.offset
        x = (self.fx * world[..., 1]) / z + self.w / 2
        y = (self.fy * self.cam_height) / z + self.h / 2

        result = torch.stack([x, y], world.ndim-1)
        result[..., 0] = torch.clamp(result[..., 0], 0, self.w-1)
        result[..., 1] = torch.clamp(result[..., 1], 0, self.h-1)
        result = result.reshape(*world.shape)

        return result

    def world_to_map(self, world):
        map_coord = world * self.pixels_per_meter
        map_coord[..., 1] *= -1
        map_coord += self.position

        return map_coord
