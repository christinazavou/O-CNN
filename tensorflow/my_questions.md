- why from octree getting the "feature" property gives a shape (3, ?) ?
- why use depth 6 in shape completion data?
- in autoencoder...do points have 4 channels nx,ny,nz,d ?
- is it correct to generate octrees with params ("") for autoencoder ?
- why is there the option to use points in the autoencoder...and use resnet instead of ocnn..and then we use the datasetfactory that calls merge octree on the points ...? is it actually translating the points into octree or not?

- sto test_octree2col:
    - to data_in pou exei shape 1,3,8,1 einai diladi 1 sample, 3 channels, 8 features tou octree(octant) ? 
    - to idx_maps ti einai ?
    - to vi ti einai ?
    - to kernel_size pou einai 3 kathe fora diladi [num_filters, wifht, height] or something else?
    - pos douleuei to octree2col ? examples:
        - input (1,3,8,1), kernel [3,3,3], stride 1, output (3,27,8)
        - input (1,3,8,1), kernel [2,2,2], stride 1, output (3,8,8)
        - input (1,3,8,1), kernel [3,1,1], stride 1, output (3,3,8)
        - input (1,3,8,1), kernel [3,3,1], stride 1, output (3,9,8)
        - input (1,3,8,1), kernel [3,3,3], stride 2, output (3,27,1)
        - input (1,3,8,1), kernel [2,2,2], stride 2, output (3,8,1)
 
- what is octree_search doing ? 

- what is the 'split_label' ?
