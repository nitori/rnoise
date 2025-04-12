class Perlin:
    def __init__(self, seed: int, octaves: int = 1): ...

    def noise(
            self, coords: list[float], freq=1.0, octaves=1, persistence=1.0, lacunarity=1.0
    ) -> float: ...

    def noise_img(
            self,
            width: int,
            height: int,
            coords: list[float | int],
            freq=1.0,
            octaves=1,
            persistence=1.0,
            lacunarity=1.0
    ) -> list[list[float]]: ...
