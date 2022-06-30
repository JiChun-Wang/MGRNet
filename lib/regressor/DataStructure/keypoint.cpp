#include "keypoint.h"

Keypoint::Keypoint() {
  point3D_cam[0] = point3D_cam[1] = point3D_cam[2] = 0.;
  point3D_obj[0] = point3D_obj[1] = point3D_obj[2] = 0.;
  inv_half_var.fill(0.);
  inv_half_var(0, 0) = inv_half_var(1, 1) = inv_half_var(2, 2)  = 1.;
}

Keypoint::~Keypoint() {

}