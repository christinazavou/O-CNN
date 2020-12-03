from evaluation.iou_calculations import *


def test_get_building_point_iou():
    ground=    np.array([1,1,1,1,2,2,4,4,4,4]).reshape((10,1))
    prediction=np.array([4,1,1,2,1,4,4,4,4,4]).reshape((10,1))
    print(get_building_point_iou(ground,prediction))
    assert True
