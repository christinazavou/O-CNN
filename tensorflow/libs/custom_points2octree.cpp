#include <chrono>
#include <iterator>

#include "math_functions.h"
#include "mesh.h"
#include "octree.h"
#include "points.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("CustomPointsToOctree")
    //    .Input("pts: float")
    //    .Input("normals: float")
    //    .Input("features: float")
    //    .Input("labels: float")
    .Input("in_points: string")
    .Attr("sigma: float=0.01")
    .Attr("clip: float=0.05")
    .Attr("depth: int=6")
    .Attr("full_depth: int=2")
    .Attr("node_dis: bool=False")
    .Attr("node_feature: bool=False")
    .Attr("split_label: bool=False")
    .Attr("adaptive: bool=False")
    .Attr("adp_depth: int=4")
    .Attr("th_normal: float=0.1")
    .Attr("th_distance: float=2.0")
    .Attr("extrapolate: bool=False")
    .Attr("save_pts: bool=False")
    .Attr("key2xyz: bool=False")
    .Output("out_octree: string")
    .Output("out_points: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->MakeShape({1}));
      c->set_output(1, c->MakeShape({1}));
      return Status::OK();
    })
    .Doc(R"doc(Points To Octree operator.)doc");

class CustomPointsToOctreeOp : public OpKernel {
public:
  explicit CustomPointsToOctreeOp(OpKernelConstruction *context)
      : OpKernel(context) {
    //        std::cout<<"gerajgieojh"<<std::endl;

    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("full_depth", &full_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("node_dis", &node_dis_));
    OP_REQUIRES_OK(context, context->GetAttr("node_feature", &node_feature_));
    OP_REQUIRES_OK(context, context->GetAttr("split_label", &split_label_));
    OP_REQUIRES_OK(context, context->GetAttr("adaptive", &adaptive_));
    OP_REQUIRES_OK(context, context->GetAttr("adp_depth", &adp_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("th_distance", &th_distance_));
    OP_REQUIRES_OK(context, context->GetAttr("th_normal", &th_normal_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolate", &extrapolate_));
    OP_REQUIRES_OK(context, context->GetAttr("save_pts", &save_pts_));
    OP_REQUIRES_OK(context, context->GetAttr("key2xyz", &key2xyz_));
    OP_REQUIRES_OK(context, context->GetAttr("sigma", &sigma_));
    OP_REQUIRES_OK(context, context->GetAttr("clip", &clip_));
  }

  void Compute(OpKernelContext *context) override {

    //    auto get_data = [&](vector<float> &vec, int idx) {
    //      const Tensor &data_in = context->input(idx);
    //      const int64 num = data_in.NumElements();
    //      if (num > 0) {
    //        const float *ptr = data_in.flat<float>().data();
    //        vec.assign(ptr, ptr + num);
    //      }
    //    };

    //    // read input data
    //    vector<float> pts, normals, features, labels;
    //    get_data(pts, 0);
    //    get_data(normals, 1);
    //    get_data(features, 2);
    //    get_data(labels, 3);

    //    // set point cloud
    //    Points point_cloud_;
    //    bool succ = false;
    //    if (features.size() > 1)
    //      succ = point_cloud_.set_points(pts, normals, features, labels);
    //    else {
    //      succ = point_cloud_.set_points(pts, normals, {}, labels);
    //    }
    //    CHECK(succ) << "Error occurs when setting points";

    // load the points
    const Tensor &data_in = context->input(0);
    CHECK_EQ(data_in.NumElements(), 1);
    Points point_cloud_;
    point_cloud_.set(data_in.flat<string>()(0).data());

    // check the points
    string msg;
    bool succ = point_cloud_.info().check_format(msg);
    CHECK(succ) << msg;

    // find min and max x,y,z for diagonal computation
    const int npt = point_cloud_.info().pt_num();
    const float* pts=point_cloud_.points();

    vector<float> x, y, z, min_xyz, max_xyz;
    for (int i = 0; i < npt; ++i) {
      x.push_back(pts[i * 3]);
      y.push_back(pts[i * 3 + 1]);
      z.push_back(pts[i * 3 + 2]);
    }
    min_xyz.push_back(*std::min_element(x.begin(), x.end()));
    min_xyz.push_back(*std::min_element(y.begin(), y.end()));
    min_xyz.push_back(*std::min_element(z.begin(), z.end()));

    max_xyz.push_back(*std::max_element(x.begin(), x.end()));
    max_xyz.push_back(*std::max_element(y.begin(), y.end()));
    max_xyz.push_back(*std::max_element(z.begin(), z.end()));

    // compute diagonal
    float diag = sqrt(pow((max_xyz[0] - min_xyz[0]), 2) +
                      pow((max_xyz[1] - min_xyz[1]), 2) +
                      pow((max_xyz[2] - min_xyz[2]), 2));

    // compute gloabal random point translation matrix
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> dis_pt(0.0f, sigma_ * diag);
    vector<float> stddev = {clamp(dis_pt(generator), -1.0f * clip_, clip_)};
    stddev.push_back(clamp(dis_pt(generator), -1.0f * clip_, clip_));
    stddev.push_back(clamp(dis_pt(generator), -1.0f * clip_, clip_));
//    stddev = {0.02, 0.02, -0.02};
    std::cout << "stddev: " << stddev[0] << " " << stddev[1] << " " << stddev[2]
              << std::endl;

    // applay global translation
    point_cloud_.translate(&stddev[0]);

    // init the octree info
    OctreeInfo octree_info_;
    octree_info_.initialize(depth_, full_depth_, node_dis_, node_feature_,
                            split_label_, adaptive_, adp_depth_, th_distance_,
                            th_normal_, key2xyz_, extrapolate_, save_pts_,
                            point_cloud_);

    // build the octree
    Octree octree_;
    octree_.build(octree_info_, point_cloud_);
    const vector<char> &octree_buf = octree_.buffer();
    const vector<char>&point_buf=point_cloud_.get_buffer();
//    vector<float> V;
//    vector<int> F;
//    octree_.octree2mesh(V, F, full_depth_, depth_, true);
//    char file_suffix[64];
//    sprintf(file_suffix, "/media/maria/BigData1/Maria/repos/test_cus_%d.obj",
//            depth_);
//    write_obj(file_suffix, V, F);

    // output
    Tensor *oct_data = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{1}, &oct_data));
    string &oct_str = oct_data->flat<string>()(0);
    oct_str.assign(octree_buf.begin(), octree_buf.end());

    Tensor *pts_data = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape{1}, &pts_data));
    string &pts_str = pts_data->flat<string>()(0);
    pts_str.assign(point_buf.begin(),point_buf.end());

  }

private:
  int depth_;
  int full_depth_;
  bool node_dis_;
  bool node_feature_;
  bool split_label_;
  bool adaptive_;
  int adp_depth_;
  float th_distance_;
  float th_normal_;
  bool extrapolate_;
  bool save_pts_;
  bool key2xyz_;
  float sigma_;
  float clip_;
};

REGISTER_KERNEL_BUILDER(Name("CustomPointsToOctree").Device(DEVICE_CPU),
                        CustomPointsToOctreeOp);
} // namespace tensorflow
