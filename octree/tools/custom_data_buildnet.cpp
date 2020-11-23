//#include <boost/filesystem.hpp>
#include <fstream>
#include <stdlib.h>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"

DEFINE_string(input,kRequired,"","File with paths to point clouds");
DEFINE_bool(colours,kOptional,false,"If the point clouds have colour info or not. Default: false(no colours)");
DEFINE_string(labels,kRequired,"","File with the paths point cloud labels");
//DEFINE_string(output_path, kOptional, "./", "Where to save ocnn point clouds");

using std::cout;
using std::vector;

const vector<int> to_be_ignored={0,32,33};
const int ignore_label=-2;

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("nUsage: custom_points input_file label_dir [colours]");
    return 0;
  }

  Points point_cloud;
  vector<float> points, normals, features, labels;
  float x,y,z,nx,ny,nz,r,g,b,a;
  int l;
  // Set your data in points, normals, features, and labels.
  // The points must not be empty, the labels may be empty,
  // the normals & features must not be empty at the same time.
  //   points: 3 channels, x_1, y_1, z_1, ..., x_n, y_n, z_n
  //   normals: 3 channels, nx_1, ny_1, nz_1, ..., nx_n, ny_n, nz_n
  //   features (such as RGB color): k channels, r_1, g_1, b_1, ..., r_n, g_n, b_n
  //   labels: 1 channels, per-points labels

  vector <string> _all_files;
  std::ifstream f(FLAGS_input);
  string line;

  //Read point files
  if (f.good()){
      while(f>>line)
        _all_files.push_back(line);
      f.close();
  }
  else{
      std::cout<<"Input file could not be opened. Exiting..."<<std::endl;
      exit(EXIT_FAILURE);
  }
  vector <string> _all_labels;
  std::ifstream  fl(FLAGS_labels);
  if (fl.good()){
      while(fl>>line)
        _all_labels.push_back(line);
      fl.close();
  }
  else{
      std::cout<<"Label file could not be opened. Exiting..."<<std::endl;
      exit(EXIT_FAILURE);
  }

// int cnt=0;
  for (int i=0;i<_all_files.size();i++){
    points.clear();normals.clear();features.clear();labels.clear();

    std::cout<<"Processing file: "<<_all_files[i]<<std::endl;
    std::ifstream infile(_all_files[i]);
    std::ifstream lfile(_all_labels[i]);
    string out_path=_all_files[i].substr(0,_all_files[i].rfind('.'));
    if (infile.good()){
        if (_all_files[i].find(".ply",0)!= string::npos){
//            std::cout<<"ply file"<<std::endl;
            infile>>line;
            while(line.find("end_header")){
//                std::cout<<line<<std::endl;
                infile>>line;
            }
        }
        while(infile >>x>>y>>z>>nx>>ny>>nz){
             points.push_back(x);
             points.push_back(y);
             points.push_back(z);
             normals.push_back(nx);
             normals.push_back(ny);
             normals.push_back(nz);

             if (FLAGS_colours){
                 infile>>r>>g>>b>>a;
                 features.push_back(r);
                 features.push_back(g);
                 features.push_back(b);
                 features.push_back(a);
             }

            lfile>>line>>l>>line;
            labels.push_back(l);
//            std::cout<<line<<","<<l<<std::endl;
//            if (std::count(to_be_ignored.begin(), to_be_ignored.end(), l)){
//                labels.push_back(ignore_label);
////                    std::cout<<"ignore label"<<std::endl;                exit(1);
////                    cnt++;
//            }else
//                labels.push_back(l-1);

            //std::cout<<x<<" "<<y<<" "<<z<<" "<<nx<<" "<<ny<<" "<<nz<<" "<<r<<" "<<g<<" "<<b<<" "<<l<<std::endl;
            //exit(0);
        }
        point_cloud.set_points(points, normals, features, labels);
        point_cloud.write_points(out_path +  "_OCNN.points");
//        std::cout<<"# of -2: "<<cnt<<std::endl;
    }
    else{
        std::cout<<"couldnt open file "<<_all_files[i]<<std::endl;
        exit(EXIT_FAILURE);
    }
  }

std::cout<<"Done."<<std::endl;
}

