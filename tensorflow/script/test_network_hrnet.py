from network_hrnet import front_layer_channeld, branch_channels


def test_front_layer_channeld():
    # print(front_layer_channeld(4, 6, 5))
    # print(front_layer_channeld(4, 7, 5))
    # print(front_layer_channeld(4, 8, 5))
    # print(front_layer_channeld(4, 6, 4))
    # print(front_layer_channeld(4, 5, 4))
    print(front_layer_channeld(64*2, 6, 5))
    print(front_layer_channeld(64*2, 7, 5))
    print(front_layer_channeld(64*2, 8, 5))
    print(front_layer_channeld(64*2, 6, 4))

def test_branch_channels():
    print(branch_channels(128, 1))

def test_me():
    num = 4
    print([[0] * num for i in range(num + 1)])

    print([None] *(num + 1))