
import numpy as np

class PlanarPoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)
    
    def get_cartisian(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def get_length(self) -> float:
        return self.r
    
    def unit_vector(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta)])
    
    def rotate_angle(self, angle: float) -> np.ndarray:
        angle = angle / 180 * np.pi
        return self.r * np.array([np.cos(self.theta+angle), np.sin(self.theta+angle)])

class PolarPoint(PlanarPoint):
    def __init__(self, r: float, theta: float) -> None:
        theta = theta / 180 * np.pi
        super().__init__(r*np.cos(theta), r*np.sin(theta))
