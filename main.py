from typing import Protocol
from contextlib import contextmanager
import time
import math

from perlin_noise import PerlinNoise
from PIL import Image
import numpy as np

import rnoise

py_perlin = PerlinNoise(seed=123)
rust_perlin = rnoise.Perlin(seed=123)


class Noise(Protocol):
    def noise(self, coords: list[float | int]) -> float:
        ...


@contextmanager
def timed(name: str):
    start = time.perf_counter()
    yield
    diff = time.perf_counter() - start
    print(f'Timed {name}: {diff:.6f}s')


def apply_noise(
        p: Noise,
        coords: list[float | int],
        freq: float,
        octaves: int = 1,
        persistence: float = 1.0,
        lacunarity: float = 1.0
) -> float:
    total = 0
    amplitude = 1
    total_amplitude = 0
    for _ in range(octaves):
        total += p.noise([c * freq for c in coords]) * amplitude
        total_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

    return total / total_amplitude


def create_image_py(
        width, height,
        freq: float,
        octaves: int = 1,
        persistence: float = 1.0,
        lacunarity: float = 1.0
) -> Image.Image:
    im = Image.new('L', (width, height))

    for y in range(height):
        for x in range(width):
            v = apply_noise(py_perlin, [x, y], freq, octaves, persistence, lacunarity)
            im.putpixel((x, y), 255 if v > 0.0 else 0)

    return im


def create_image_rust(
        width,
        height,
        coords: list[float | int] = None,
        freq: float = 1.0,
        octaves: int = 1,
        persistence: float = 1.0,
        lacunarity: float = 1.0
) -> Image.Image:
    data = rust_perlin.noise_img(width, height, coords or [], freq, octaves, persistence, lacunarity)
    data = np.array(data)

    new_data = np.zeros_like(data, dtype='uint8')

    new_data[data > 0] = 255
    new_data[data <= 0] = 0

    return Image.fromarray(new_data, 'L')


def main():
    freq = 1 / 128
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    stretch = 5.0

    print('starting rust...')
    with timed('Rust'):
        frames = []
        number_of_frames = 50

        for frame_no in range(number_of_frames):
            progress = frame_no / number_of_frames
            print(f'\r{progress * 100:.1f}%', end='')

            r = progress * math.pi * 2
            w, z = math.cos(r) * stretch, math.sin(r) * stretch

            frame = create_image_rust(100, 100, [w, z], freq, octaves, persistence, lacunarity)
            frames.append(frame)
        print('\r100%  ')

    frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)

    # print('starting python...')
    # with timed('Python'):
    #     im1 = create_image_py(1000, 1000, freq, octaves, persistence, lacunarity)
    # im1.save('im_py.png')


main()
