#ifndef _Geometry_KdTree_Hpp_
#define _Geometry_KdTree_Hpp_

#include <memory>
#include <vector>

#include "Core/Typedef.hpp"

#include "nanoflann.hpp"

namespace Geometry {
class KdTree {
public:
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 1>
      TreeType1D;
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 2>
      TreeType2D;
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 3>
      TreeType3D;

protected:
  std::shared_ptr<TreeType1D> tree1D_;
  std::shared_ptr<TreeType2D> tree2D_;
  std::shared_ptr<TreeType3D> tree3D_;

  HostRealMatrix &pointCloud_;

  int dim_;
  int maxLeaf_;

public:
  KdTree(HostRealMatrix &pointCloud, const int dimension, const int maxLeaf = 5)
      : pointCloud_(pointCloud), dim_(dimension), maxLeaf_(maxLeaf) {}

  ~KdTree() {}

  // methods required by Nanoflann
  template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const { return false; }

  inline int kdtree_get_point_count() const { return pointCloud_.extent(0); }

  inline double kdtree_get_pt(const int idx, int dim) const {
    return pointCloud_(idx, dim);
  }

  inline double kdtree_distance(const double *queryPt, const int idx,
                                long long sz) const {
    double distance = 0;
    for (int i = 0; i < dim_; ++i) {
      distance += pow(pointCloud_(idx, i) - queryPt[i], 2.0);
    }
    printf("%f\n", std::sqrt(distance));
    return std::sqrt(distance);
  }

  void generateKDTree() {
    if (dim_ == 1) {
      tree1D_ = std::make_shared<TreeType1D>(
          1, *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf_));
      tree1D_->buildIndex();
    } else if (dim_ == 2) {
      tree2D_ = std::make_shared<TreeType2D>(
          2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf_));
      tree2D_->buildIndex();
    } else if (dim_ == 3) {
      tree3D_ = std::make_shared<TreeType3D>(
          3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf_));
      tree3D_->buildIndex();
    }
  }

  void getIndex(std::vector<int> &idx) {
    idx.resize(pointCloud_.extent(0));

    std::shared_ptr<std::vector<size_t>> p_idx;
    if (dim_ == 1) {
      p_idx = std::make_shared<std::vector<size_t>>(tree1D_->vind);
    } else if (dim_ == 2) {
      p_idx = std::make_shared<std::vector<size_t>>(tree2D_->vind);
    } else if (dim_ == 3) {
      p_idx = std::make_shared<std::vector<size_t>>(tree3D_->vind);
    }
    for (int i = 0; i < pointCloud_.extent(0); i++) {
      idx[i] = (*p_idx)[i];
    }
  }
};
} // namespace Geometry

#endif