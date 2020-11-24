from network_unet import network_unet
from network_hrnet import HRNet


def seg_network(octree, flags, training, reuse=False, pts=None, mask=None):
    if flags.name.lower() == 'unet':
        return network_unet(octree, flags, training, reuse)
    elif flags.name.lower() == 'hrnet':
        return HRNet(flags).network_seg(octree, training, reuse, pts, mask)
    else:
        print('Error, no network: ' + flags.name)
