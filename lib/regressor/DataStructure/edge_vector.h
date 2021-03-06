#ifndef edge_vector_h_
#define edge_vector_h_

#include <Eigen/Dense>
using namespace Eigen;

struct EdgeVector {
 public:
   EdgeVector();
   ~EdgeVector();
   Vector3d vec_pred;
   // Predicted inverse half variance
   Matrix3d inv_half_var;
   // Indices of the source and target end vertices
   int start_id;
   int end_id;
};
#endif