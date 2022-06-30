#include "edge_vector.h"

EdgeVector::EdgeVector() {
  vec_pred[0] = vec_pred[1] = vec_pred[2] = 0.;
  inv_half_var.fill(0.);
  inv_half_var(0, 0) = inv_half_var(1, 1) = inv_half_var(2, 2)  = 1.;
  start_id = 0;
  end_id = 0;
}

EdgeVector::~EdgeVector() {

}