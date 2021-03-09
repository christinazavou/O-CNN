#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "octree_nn.h"
#include "octree_parser.h"
using namespace std;

namespace tensorflow {

REGISTER_OP("CheckOctree")
    .Input("octree: int8")
//    .Attr("depth: int")
    .Doc(R"doc(Print octree properties.)doc");

class CheckOctreeOp : public OpKernel {
 public:
  explicit CheckOctreeOp(OpKernelConstruction* context) : OpKernel(context) {
//    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
  }

  void Compute(OpKernelContext* context) override {
    auto octree_ptr = context->input(0).flat<int8>().data();
    OctreeParser octree;
    octree.set_gpu(octree_ptr);

    int depth = 16;

    cout << "Octree depth:" << octree.info().depth() << endl;
    cout << "batch_size: " << octree.info().batch_size() << endl;
    cout << "full_layer: " << octree.info().full_layer() << endl;
    cout << "adaptive_layer: " << octree.info().adaptive_layer() << endl;
    cout << "threshold_distance: " << octree.info().threshold_distance() << endl;
    cout << "threshold_normal: " << octree.info().threshold_normal() << endl;
    cout << "is_adaptive: " << octree.info().is_adaptive() << endl;
    cout << "has_displace: " << octree.info().has_displace() << endl;
    cout << "nnum: ";
    for (int d = 0; d < depth; ++d) {
      cout << octree.info().node_num(d) << " ";
    }
    cout << endl << "nnum_cum: ";
    for (int d = 0; d < depth; ++d) {
      cout << octree.info().node_num_cum(d) << " ";
    }
    cout << endl << "nnum_nempty: ";
    for (int d = 0; d < depth; ++d) {
      cout << octree.info().node_num_nempty(d) << " ";
    }
    cout << endl << "total_nnum: " << octree.info().total_nnum() << endl;
    cout << "total_nnum_capacity: " << octree.info().total_nnum_capacity() << endl;
    cout << "channel: ";
    for (int j = 0; j < OctreeInfo::kPTypeNum; ++j) {
      cout << octree.info().channel(static_cast<OctreeInfo::PropType>(1 << j)) << " ";
    }
    cout << endl << "locations: ";
    for (int j = 0; j < OctreeInfo::kPTypeNum; ++j) {
      cout << octree.info().locations(static_cast<OctreeInfo::PropType>(1 << j)) << " ";
    }
    cout << endl << "bbox_max: ";
    for (int j = 0; j < 3; ++j) {
      cout << octree.info().bbmax()[j] << " ";
    }
    cout << endl << "bbox_min: ";
    for (int j = 0; j < 3; ++j) {
      cout << octree.info().bbmin()[j] << " ";
    }
    cout << "\nkey2xyz: " << octree.info().is_key2xyz() << endl;
    cout << "sizeof_octree: " << octree.info().sizeof_octree() << endl;
    cout << "===============\n" << endl;

  }

 private:
  int depth_;
};

REGISTER_KERNEL_BUILDER(Name("CheckOctree").Device(DEVICE_GPU),
                        CheckOctreeOp);

}  // namespace tensorflow
