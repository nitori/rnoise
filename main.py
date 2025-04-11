import rnoise
from perlin_noise import PerlinNoise

py_perlin = PerlinNoise(seed=123, octaves=1)
print(f'\n{py_perlin = }')

rust_perlin = rnoise.Perlin(seed=123, octaves=1)
print(f'\n{rust_perlin = }')

freq = 16

print(f'{py_perlin.noise([1/freq, 2/freq]) = }')

# the rng methods are different, but up until the "vec" is applied in
# RandVec.get_weighted_val, the values all match with the python version.
print(f'{rust_perlin.noise([1/freq, 2/freq]) = }')
