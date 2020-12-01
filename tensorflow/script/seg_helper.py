import os

import numpy as np

np.random.seed(100)


ANNFASS_LABELS = [
    "undetermined",
    "wall",
    "window",
    "vehicle",
    "roof",
    "plant_tree",
    "door",
    "tower_steeple",
    "furniture",
    "ground_grass",
    "beam_frame",
    "stairs",
    "column",
    "railing_baluster",
    "floor",
    "chimney",
    "ceiling",
    "fence",
    "pond_pool",
    "corridor_path",
    "balcony_patio",
    "garage",
    "dome",
    "road",
    "entrance_gate",
    "parapet_merlon",
    "buttress",
    "dormer",
    "lantern_lamp",
    "arch",
    "awning",
    "shutters",
    "empty",
]


ANNFASS_COLORS = [
    "#000000",  # 0
    "#ff4500",  # 1
    "#0000ff",  # 2
    "#396073",  # 3
    "#4b008c",  # 4
    "#fa8072",  # 5
    "#7f0000",  # 6
    "#d6f2b6",  # 7
    "#0d2133",  # 8
    "#204035",  # 9
    "#ff4040",  # 10
    "#60b9bf",  # 11
    "#3d4010",  # 12
    "#733d00",  # 13
    "#400000",  # 14
    "#999673",  # 15
    "#ff00ff",  # 16
    "#394173",  # 17
    "#553df2",  # 18
    "#bf3069",  # 19
    "#301040",  # 20
    "#ff9180",  # 21
    "#997391",  # 22
    "#ffbfd9",  # 23
    "#00aaff",  # 24
    "#8a4d99",  # 25
    "#40ff73",  # 26
    "#8c6e69",  # 27
    "#cc00ff",  # 28
    "#b24700",  # 29
    "#ffbbdd",  # 30
    "#0dd3ff",  # 31
    "#000000",  # -1
]


def decimal_to_rgb(decimal):
    hexadecimal_str = '{:06x}'.format(decimal)
    return tuple(int(hexadecimal_str[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb(hex):
    return tuple(int(hex[1:][i:i + 2], 16) for i in (0, 2, 4))


def to_rgb(value):
    if isinstance(value, str) and '#' in value:
        return hex_to_rgb(value)
    assert isinstance(value, np.int64) , " given value {} is of type {}".format(value, type(value))
    return decimal_to_rgb(value)


def save_ply(filename, points, colors):
    pts_num = points.shape[0]

    header = "ply\n" \
             "format ascii 1.0\n" \
             "element vertex %d\n" \
             "property float x\n" \
             "property float y\n" \
             "property float z\n" \
             "property uchar red\n" \
             "property uchar green\n" \
             "property uchar blue\n" \
             "end_header\n"
    with open(filename, 'w') as fid:
        fid.write(header % pts_num)
        for point, color in zip(points, colors):
            fid.write(" ".join([str(i) for i in point]) + " " +
                      " ".join([str(int(i)) for i in color]) + "\n")

