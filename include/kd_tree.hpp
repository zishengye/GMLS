#include <memory>
#include <vector>

#include "nanoflann.hpp"
#include "vec3.hpp"

class KDTree {
public:
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KDTree>, KDTree, 1>
      tree_type_1d;
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KDTree>, KDTree, 2>
      tree_type_2d;
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KDTree>, KDTree, 3>
      tree_type_3d;

protected:
  std::shared_ptr<tree_type_1d> _tree_1d;
  std::shared_ptr<tree_type_2d> _tree_2d;
  std::shared_ptr<tree_type_3d> _tree_3d;

  std::shared_ptr<std::vector<vec3>> _point_cloud;

  int _dim;
  int _max_leaf;

public:
  KDTree(std::shared_ptr<std::vector<vec3>> point_cloud, const int dimension)
      : _point_cloud(point_cloud), _dim(dimension), _max_leaf(10) {}

  ~KDTree() {}

  // methods required by Nanoflann
  template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const { return false; }

  inline int kdtree_get_point_count() const { return _point_cloud->size(); }

  inline double kdtree_get_pt(const int idx, int dim) const {
    return (*_point_cloud)[idx][dim];
  }

  inline double kdtree_distance(const double *queryPt, const int idx,
                                long long sz) const {
    vec3 X = (*_point_cloud)[idx];
    double distance = 0;
    for (int i = 0; i < _dim; ++i) {
      distance += (X[i] - queryPt[i]) * (X[i] - queryPt[i]);
    }
    return std::sqrt(distance);
  }

  void generateKDTree() {
    if (_dim == 1) {
      _tree_1d = std::make_shared<tree_type_1d>(
          1, *this, nanoflann::KDTreeSingleIndexAdaptorParams(_max_leaf));
      _tree_1d->buildIndex();
    } else if (_dim == 2) {
      _tree_2d = std::make_shared<tree_type_2d>(
          2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(_max_leaf));
      _tree_2d->buildIndex();
    } else if (_dim == 3) {
      _tree_3d = std::make_shared<tree_type_3d>(
          3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(_max_leaf));
      _tree_3d->buildIndex();
    }
  }

  void getIndex(std::vector<int> &idx) {
    idx.resize(_point_cloud->size());

    std::shared_ptr<std::vector<size_t>> p_idx;
    if (_dim == 1) {
      p_idx = std::make_shared<std::vector<size_t>>(_tree_1d->vind);
    } else if (_dim == 2) {
      p_idx = std::make_shared<std::vector<size_t>>(_tree_2d->vind);
    } else if (_dim == 3) {
      p_idx = std::make_shared<std::vector<size_t>>(_tree_3d->vind);
    }
    for (int i = 0; i < _point_cloud->size(); i++) {
      idx[i] = (*p_idx)[i];
    }
  }
};