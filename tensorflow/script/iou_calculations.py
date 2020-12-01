import numpy as np

toplabels = {0: "undetermined", 1: "wall", 2: "window", 3: "vehicle", 4: "roof", 5: "plant_tree", 6: "door",
             7: "tower_steeple",
             8: "furniture", 9: "ground_grass", 10: "beam_frame", 11: "stairs", 12: "column", 13: "railing_baluster",
             14: "floor", 15: "chimney", 16: "ceiling", 17: "fence", 18: "pond_pool", 19: "corridor_path",
             20: "balcony_patio",
             21: "garage", 22: "dome", 23: "road", 24: "entrance_gate", 25: "parapet_merlon", 26: "buttress",
             27: "dormer",
             28: "lantern_lamp", 29: "arch", 30: "awning", 31: "shutters"}  # , 32:"ramp", 33:"canopy_gazebo" }

# Most frequent labels
freqlabels = {1: "wall", 2: "window", 3: "vehicle", 4: "roof", 5: "plant_tree", 6: "door", 7: "tower_steeple",
              8: "furniture",
              10: "beam_frame", 11: "stairs", 12: "column", 17: "fence", 20: "balcony_patio", 25: "parapet_merlon"}


def get_building_point_iou(ground, prediction):
    """
    Calculate point IOU for buildings
  :param ground: N x 1, numpy.ndarray(int)
  :param prediction: N x 1, numpy.ndarray(int)
  :return:
    metrics: dict: {
                    "label_iou": dict{label: iou (float)},
                    "intersection": dict{label: intersection (float)},
                    "union": dict{label: union (float)
                   }
  """

    label_iou, intersection, union = {}, {}, {}
    # Ignore undetermined
    prediction = np.copy(prediction)
    prediction[ground == 0] = 0

    for i in range(1, len(toplabels)):
        # Calculate intersection and union for ground truth and predicted labels
        intersection_i = np.sum((ground == i) & (prediction == i))
        union_i = np.sum((ground == i) | (prediction == i))

        # If label i is present either on the gt or the pred set
        if union_i > 0:
            intersection[i] = float(intersection_i)
            union[i] = float(union_i)
            label_iou[i] = intersection[i] / union[i]

    metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

    return metrics


def get_building_mesh_iou(ground, prediction, face_area):
    """
    Calculate mesh IOU for buildings
  :param ground: N x 1, numpy.ndarray(int)
  :param prediction: N x 1, numpy.ndarray(int)
  :param face_area: N x 1, numpy.ndarray(float)
  :return:
    metrics: dict: {
                    "label_iou": dict{label: iou (float)},
                    "intersection": dict{label: intersection (float)},
                    "union": dict{label: union (float)
                   }
  """

    label_iou, intersection, union = {}, {}, {}
    # Ignore undetermined
    prediction = np.copy(prediction)
    prediction[ground == 0] = 0

    for i in range(1, len(toplabels)):
        # Calculate binary intersection and union for ground truth and predicted labels
        intersection_i = ((ground == i) & (prediction == i))
        union_i = ((ground == i) | (prediction == i))

        if np.sum(union_i) > 0:
            intersection[i] = np.dot(face_area.T, intersection_i)[0]
            union[i] = np.dot(face_area.T, union_i)[0]
            if union[i] > 0.0:
                label_iou[i] = intersection[i] / union[i]
            else:
                print(len(union[i]))
                label_iou[i] = np.array([0.0])

    metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

    return metrics


def get_shape_iou(buildings_iou):
    """
    Average label IOU and calculate overall shape IOU
  :param buildings_iou: dict: {
                                <model_name> : {
                                                "label_iou": dict{label: iou (float)},
                                                "intersection": dict{label: intersection (float)},
                                                "union": dict{label: union (float)
                                               }
                              }
  :return:
    shape_iou: dict: {
                      "all": avg shape iou,
                      <model_name>: per building shape iou
                     }
  """

    shape_iou = {}

    for building, metrics in buildings_iou.items():
        # Average label iou per shape
        L_s = len(metrics["label_iou"])
        shape_iou[building] = np.sum([v for v in metrics["label_iou"].values()]) / float(L_s)

    # Dataset avg shape iou
    shape_iou['all'] = np.sum([v for v in shape_iou.values()]) / float(len(buildings_iou))

    return shape_iou


def get_part_iou(buildings_iou):
    """
    Average intersection/union and calculate overall part IOU and most frequent part IOU
  :param buildings_iou: dict: {
                              <model_name> : {
                                              "label_iou": dict{label: iou (float)},
                                              "intersection": dict{label: intersection (float)},
                                              "union": dict{label: union (float)
                                             }
                             }
  :return:
    part_iou:  dict: {
                      "all": avg part iou,
                      "fr-part": most frequent labels part iou
                      <label_name>: per label part iou
                     }
  """

    intersection = {i: 0.0 for i in range(1, len(toplabels))}
    union = {i: 0.0 for i in range(1, len(toplabels))}

    for building, metrics in buildings_iou.items():
        for label in metrics["intersection"].keys():
            # Accumulate intersection and union for each label across all shapes
            intersection[label] += metrics["intersection"][label]
            union[label] += metrics["union"][label]

    # Calculate part IOU for each label
    part_iou = {toplabels[key]: intersection[key] / union[key] for key in range(1, len(toplabels))}
    # Avg part IOU
    part_iou["all"] = np.sum([v for v in part_iou.values()]) / float(len(toplabels) - 1)
    # Most frequent labels part IOU
    part_iou["fr-part"] = np.sum([part_iou[freqlabels[key]] for key in freqlabels.keys()]) / float(len(freqlabels))

    return part_iou