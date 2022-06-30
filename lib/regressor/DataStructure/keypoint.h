#ifndef keypoint_h_
#define keypoint_h_

#include <Eigen/Dense>
using namespace Eigen;

struct Keypoint {
 public:
   Keypoint();
   ~Keypoint();
   // Ground-truth 3D position in object coordinate system
   Vector3d point3D_obj;
   // Predicted 3D position in camera coordinate system
   Vector3d point3D_cam;
   // Predicted inverse half variance
   Matrix3d inv_half_var;
};
#endif