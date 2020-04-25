import math
from typing import List

import PIL
from PIL.Image import Image
from PIL import ImageFilter
import os
import numpy as np

constrain = lambda v, min_value, max_value: min(max(v, min_value), max_value)
constrain2d = lambda v, min_corner, max_corner: np.array([constrain(v[0], min_corner[0], max_corner[0]), constrain(v[1], min_corner[1], max_corner[1])])
def distance_image (img:Image, reference_color=(0, 0, 0)) -> Image:
    new_image = PIL.Image.new('RGB', (img.width, img.height))
    new_image.resize((img.width, img.height))
    for y in range(img.height):
        for x in range(img.width):
            col = img.getpixel((x, y))
            bw = math.sqrt(
                (col[0]-reference_color[0]) ** 2 +
                (col[1]-reference_color[1]) ** 2 +
                (col[2]-reference_color[2]) ** 2
            )
            bw = int(bw)
            new_image.putpixel((x, y), (bw,)*3)
    return new_image
def blur_image (img:Image, times=1) -> Image:
    for i in range(times):
        img = img.filter(ImageFilter.BLUR)
    return img
def get_sides (xy, angle_rad, radius) -> np.ndarray:
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    return np.array([
        np.array([ xy[0] + radius * ca, xy[1] + radius * sa ]),
        np.array([ xy[0] - radius * ca, xy[1] - radius * sa ])
    ])

def _int (tup) -> list:
    t = []
    for v in tup:
        t.append(int(v))
    return t

def spiral_process (img: Image, draw_color=(200, 200, 200)) -> Image:
    res_increase = 2.0 # 4x bigger
    print('original dimensions: %sx%s'%(img.width, img.height))
    img = img.resize(_int((img.width * res_increase, img.height * res_increase)))
    print('new dimensions: %sx%s'%(img.width, img.height))
    img_min_corner = np.array([0, 0])
    img_max_corner = np.array([img.width - 1, img.height - 1])
    img_center = _int((img_min_corner + (img_max_corner - img_min_corner) / 2))
    dim = max(img.width, img.height)
    radial_rows = 20
    turns = 10
    resolution = 2000
    thickness = dim / 2 / radial_rows * 0.5 * 0.7
    left_line:List[List[float]] = []
    right_line:List[List[float]] = []
    radius = 0
    θ = 0
    while radius < dim:
        new_point = np.array([
            math.cos(θ) * radius,
            math.sin(θ) * radius
        ])
        clean_point = _int(constrain2d(new_point + img_center, img_min_corner, img_max_corner))
        brightness = 1 - img.getpixel((clean_point[0], clean_point[1]))[0] / 255 # assuming paper is white
        sides = get_sides(new_point, θ, thickness * brightness / 2)
        left_line.append(constrain2d(_int(sides[0] + img_center), img_min_corner, img_max_corner))
        right_line.append(constrain2d(_int(sides[1] + img_center), img_min_corner, img_max_corner))
        θ += turns / radial_rows * 2 * math.pi / resolution
        radius = thickness * θ / (2 * math.pi)

    print('done creating points -- drawing points')
    new_img = PIL.Image.new('RGB', (img.width, img.height), (255, 255, 255))
    for point in left_line:
        new_img.putpixel(point, draw_color)
    for point in right_line:
        new_img.putpixel(point, draw_color)
    final_scale = 2.0
    return new_img.resize(_int((new_img.width * final_scale, new_img.height * final_scale)), PIL.Image.ANTIALIAS)
def main (in_path='./in', out_path='./out'):
    for f in os.listdir(in_path):
        img:Image = PIL.Image.open(f'{in_path}/{f}')
        img = img.convert('RGB')
        bw_i = distance_image(img)
        bw_i = blur_image(bw_i, 3)
        # bw_i.show()
        final_image = spiral_process(bw_i)
        final_image.save(f'{out_path}/{f}')
        # final_image.show()

if __name__ == '__main__':
    main()