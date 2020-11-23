
run_seg_partnet.py:
    graph:

        // from DatasetFactory it reads tfrecords file and gets labels & points
        // and generates the octrees  
        // octree and points are byte strings
        octree, _labels, points = data_iter.get_next()
        
        // from points we get xyz & labels
        pts, label, dc = get_point_info(points, flags_data.mask_ratio)
        
        in hrnet backbone:
        
            // from octree we get features 
            // BUT THIS IS EMPTY WITH SHAPE
            data = octree_property(octree, property_name='feature', dtype=tf.float32,
                                   depth=depth, channel=flags.channel)
            
            // here he calls some convolutions in the feedforward pass (and the creation of branchces), 
            // where he sends the data and the octree as well.
            // the octree is only used for size checks (number of empty notes)
            
        PROBABLY:
        in hrnet header (with points):
            
            // he gets the output of layer of depth 5 (octree features convolved) and upsamples it to get the dimension
            // of octree leaf nodes
            feature, dc = self.points_feat(inputs, octree) 
            
            // then he uses xyz of points to do bilinear interpolation and get a feature vector for each point
            feature = octree_bilinear_v3(pts5, feature, octree, depth=5)
            
        then he finds logits.
        

            
        