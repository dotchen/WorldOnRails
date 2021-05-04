import math
import numpy as np

from agents.navigation.local_planner import RoadOption


class Waypointer:
    
    EARTH_RADIUS = 6371e3 # 6371km
    
    def __init__(self, global_plan, current_gnss, threshold_lane=10., threshold_before=4.5, threshold_after=4.5, pop_lane_change=True):
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after
        self._threshold_lane = threshold_lane
        self._pop_lane_change = pop_lane_change

        self._lane_change_counter = 0
        
        # Convert lat,lon to x,y
        cos_0 = 0.
        for gnss, _ in global_plan:
            cos_0 += gnss['lat'] * (math.pi / 180)
        cos_0 = cos_0 / (len(global_plan))
        self.cos_0 = cos_0
        
        self.global_plan = []
        for node in global_plan:
            gnss, cmd = node

            x, y = self.latlon_to_xy(gnss['lat'], gnss['lon'])
            self.global_plan.append((x, y, cmd))

        lat, lon, _ = current_gnss
        cx, cy = self.latlon_to_xy(lat, lon)
        self.checkpoint = (cx, cy, RoadOption.LANEFOLLOW)
        
        self.current_idx = -1

    def tick(self, gnss):
        
        lat, lon, _ = gnss
        x, y = self.latlon_to_xy(lat, lon)

        for i, (wx, wy, cmd) in enumerate(self.global_plan):

            # CMD remap... HACK...
            distance = np.linalg.norm([x-wx, y-wy])

            if self.checkpoint[2] == RoadOption.LANEFOLLOW and cmd != RoadOption.LANEFOLLOW:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            if distance < threshold and i-self.current_idx == 1:
                self.checkpoint = (wx, wy, cmd)
                self.current_idx += 1
                break

        return self.checkpoint


    def latlon_to_xy(self, lat, lon):
        
        x = self.EARTH_RADIUS * lat * (math.pi / 180)
        y = self.EARTH_RADIUS * lon * (math.pi / 180) * math.cos(self.cos_0)

        return x, y
