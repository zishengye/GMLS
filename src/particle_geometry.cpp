#include "particle_geometry.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <tgmath.h>
#include <vector>

#include <mpi.h>

using namespace std;

static void process_split(int &x, int &y, int &i, int &j, const int size,
                          const int rank) {
  x = sqrt(size);
  while (x > 0) {
    y = size / x;
    if (size == x * y) {
      break;
    } else {
      x--;
    }
  }

  i = rank % x;
  j = rank / y;
}

static void process_split(int &x, int &y, int &z, int &i, int &j, int &k,
                          const int size, const int rank) {
  x = cbrt(size);
  bool splitFound = false;
  while (x > 0 && splitFound == false) {
    y = x;
    while (y > 0 && splitFound == false) {
      z = size / (x * y);
      if (size == (x * y * z)) {
        splitFound = true;
        break;
      } else {
        y--;
      }
    }
    if (splitFound == false) {
      x--;
    }
  }

  i = (rank % (x * y)) % x;
  j = (rank % (x * y)) / x;
  k = rank / (x * y);
}

static int bounding_box_split(vec3 &bounding_box_size,
                              triple<int> &bounding_box_count,
                              vec3 &bounding_box_low, double _spacing,
                              vec3 &domain_bounding_box_low,
                              vec3 &domain_bounding_box_high, vec3 domain_low,
                              vec3 domain_high, triple<int> &domain_count,
                              int x, int y, int i, int j) {
  for (int ite = 0; ite < 2; ite++) {
    bounding_box_count[ite] = bounding_box_size[ite] / _spacing;
  }

  vector<int> count_x, count_y;
  for (int ite = 0; ite < x; ite++) {
    if (bounding_box_count[0] % x > ite) {
      count_x.push_back(bounding_box_count[0] / x + 1);
    } else {
      count_x.push_back(bounding_box_count[0] / x);
    }
  }

  for (int ite = 0; ite < y; ite++) {
    if (bounding_box_count[1] % y > ite) {
      count_y.push_back(bounding_box_count[1] / y + 1);
    } else {
      count_y.push_back(bounding_box_count[1] / y);
    }
  }

  domain_count[0] = count_x[i];
  domain_count[1] = count_y[j];

  double x_start = bounding_box_low[0];
  double y_start = bounding_box_low[1];

  for (int ite = 0; ite < i; ite++) {
    x_start += count_x[ite] * _spacing;
  }
  for (int ite = 0; ite < j; ite++) {
    y_start += count_y[ite] * _spacing;
  }

  double x_end = x_start + count_x[i] * _spacing;
  double y_end = y_start + count_y[j] * _spacing;

  domain_low[0] = x_start;
  domain_low[1] = y_start;
  domain_high[0] = x_end;
  domain_high[1] = y_end;

  domain_bounding_box_low[0] =
      bounding_box_size[0] / x * i + bounding_box_low[0];
  domain_bounding_box_high[1] =
      bounding_box_size[1] / y * j + bounding_box_low[1];
  domain_bounding_box_high[0] =
      bounding_box_size[0] / x * (i + 1) + bounding_box_low[0];
  domain_bounding_box_high[1] =
      bounding_box_size[1] / y * (j + 1) + bounding_box_low[1];
}

void particle_geometry::init(const int _dim, const int _problem_type,
                             const int _refinement_type, double _spacing,
                             double _cutoff_multiplier,
                             string geometry_input_file_name) {
  dim = _dim;
  problem_type = _problem_type;
  refinement_type = _refinement_type;
  _spacing = _spacing;
  cutoff_multiplier = _cutoff_multiplier;
  cutoff_distance = _spacing * cutoff_multiplier;

  if (geometry_input_file_name != "") {
  } else {
    // default setup
    bounding_box_size[0] = 2.0;
    bounding_box_size[1] = 2.0;
    bounding_box_size[2] = 2.0;

    bounding_box[0][0] = -1.0;
    bounding_box[1][0] = 1.0;
    bounding_box[0][1] = -1.0;
    bounding_box[1][1] = 1.0;
    bounding_box[0][2] = -1.0;
    bounding_box[1][2] = 1.0;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (dim == 2) {
      process_split(process_x, process_y, process_i, process_j, size, rank);
    } else if (dim == 3) {
      process_split(process_x, process_y, process_z, process_i, process_j,
                    process_k, size, rank);
    }
  }

  if (dim == 2) {
    bounding_box_split(bounding_box_size, bounding_box_count, bounding_box[0],
                       _spacing, domain_bounding_box[0], domain_bounding_box[1],
                       domain[0], domain[1], domain_count, process_x, process_y,
                       process_i, process_j);
  }

  // prepare data storage
  current_local_work_particle_coord = make_shared<vector<vec3>>(vector<vec3>());
  current_local_work_ghost_particle_coord =
      make_shared<vector<vec3>>(vector<vec3>());
  current_local_gmls_particle_coord = make_shared<vector<vec3>>(vector<vec3>());
  current_local_gmls_ghost_particle_coord =
      make_shared<vector<vec3>>(vector<vec3>());
  current_local_managing_particle_coord =
      make_shared<vector<vec3>>(vector<vec3>());

  current_local_work_particle_index =
      make_shared<vector<long long>>(vector<long long>());
  current_local_work_ghost_particle_index =
      make_shared<vector<int>>(vector<int>());
  current_local_gmls_particle_index = make_shared<vector<int>>(vector<int>());
  current_local_gmls_ghost_particle_index =
      make_shared<vector<int>>(vector<int>());
  current_local_managing_particle_index =
      make_shared<vector<long long>>(vector<long long>());

  local_managing_gap_particle_coord = make_shared<vector<vec3>>(vector<vec3>());

  init_domain_boundary();
}

void particle_geometry::init_rigid_body(rigid_body_manager &mgr) {
  rb_mgr = make_shared<rigid_body_manager>(mgr);
}

void particle_geometry::generate_uniform_particle() {
  generate_rigid_body_surface_particle();
  generate_field_particle();

  index_particle();
}

void particle_geometry::clear_particle() {}

void particle_geometry::mitigate_forward() {}

void particle_geometry::mitigate_backward() {}

void particle_geometry::split_particle() {}

void particle_geometry::refine() {
  if (refinement_type == UNIFORM_REFINE) {
    uniform_refine();
  } else if (refinement_type == ADAPTIVE_REFINE) {
    adaptive_refine();
  }
}

void particle_geometry::init_domain_boundary() {
  if (dim == 3) {
    // six faces as boundary
    // 0 front face
    // 1 right face
    // 2 back face
    // 3 bottom face
    // 4 left face
    // 5 top face
    domain_boundary_type.resize(6);
    if (abs(domain[1][0] - bounding_box[1][0]) < 1e-6) {
      domain_boundary_type[0] = 1;
    } else {
      domain_boundary_type[0] = 0;
    }

    if (abs(domain[1][1] - bounding_box[1][1]) < 1e-6) {
      domain_boundary_type[1] = 1;
    } else {
      domain_boundary_type[1] = 0;
    }

    if (abs(domain[0][0] - bounding_box[0][0]) < 1e-6) {
      domain_boundary_type[2] = 1;
    } else {
      domain_boundary_type[2] = 0;
    }

    if (abs(domain[0][2] - bounding_box[0][2]) < 1e-6) {
      domain_boundary_type[3] = 1;
    } else {
      domain_boundary_type[3] = 0;
    }

    if (abs(domain[0][1] - bounding_box[0][1]) < 1e-6) {
      domain_boundary_type[4] = 1;
    } else {
      domain_boundary_type[4] = 0;
    }

    if (abs(domain[1][2] - bounding_box[1][2]) < 1e-6) {
      domain_boundary_type[5] = 1;
    } else {
      domain_boundary_type[5] = 0;
    }
  }
  if (dim == 2) {
    // four edges as boundary
    // 0 down edge
    // 1 right edge
    // 2 up edge
    // 3 left edge
    domain_boundary_type.resize(4);
    if (abs(domain[0][0] - bounding_box[0][0]) < 1e-6) {
      domain_boundary_type[3] = 1;
    } else {
      domain_boundary_type[3] = 0;
    }

    if (abs(domain[0][1] - bounding_box[0][1]) < 1e-6) {
      domain_boundary_type[0] = 1;
    } else {
      domain_boundary_type[0] = 0;
    }

    if (abs(domain[1][0] - bounding_box[1][0]) < 1e-6) {
      domain_boundary_type[1] = 1;
    } else {
      domain_boundary_type[1] = 0;
    }

    if (abs(domain[1][1] - bounding_box[1][1]) < 1e-6) {
      domain_boundary_type[2] = 1;
    } else {
      domain_boundary_type[2] = 0;
    }
  }
}

void particle_geometry::generate_field_particle() {
  static auto &coord = current_local_managing_particle_coord;

  double pos_x, pos_y, pos_z;
  vec3 normal = vec3(1.0, 0.0, 0.0);
  vec3 boundary_normal;

  if (dim == 2) {
    pos_z = 0.0;
    double vol = spacing * spacing;
    int localIndex = coord->size();

    // down
    if (domain_boundary_type[0] != 0) {
      pos_x = domain[0][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[3] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
      pos_x += 0.5 * spacing;

      while (pos_x < domain[1][0] - 1e-5) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(0.0, 1.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
        pos_x += spacing;
      }

      if (domain_boundary_type[1] != 0) {
        pos_x = domain[1][0];
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(-sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
    }

    // fluid particle
    pos_y = domain[0][1] + spacing / 2.0;
    while (pos_y < domain[1][1] - 1e-5) {
      // left
      if (domain_boundary_type[3] != 0) {
        pos_x = domain[0][0];
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(1.0, 0.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
      }

      pos_x = domain[0][0] + spacing / 2.0;
      while (pos_x < domain[1][0] - 1e-5) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        insert_particle(_pos, 0, spacing, normal, 0, vol);
        pos_x += spacing;
      }

      // right
      if (domain_boundary_type[1] != 0) {
        pos_x = domain[1][0];
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(-1.0, 0.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
      }

      pos_y += spacing;
    }

    // up
    if (domain_boundary_type[2] != 0) {
      pos_x = domain[0][0];
      pos_y = domain[1][1];
      if (domain_boundary_type[3] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
      pos_x += 0.5 * spacing;

      while (pos_x < domain[1][0] - 1e-5) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(0.0, -1.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
        pos_x += spacing;
      }

      pos_x = domain[1][0];
      if (domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(-sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
    }
  }
  if (dim == 3) {
    double vol = spacing * spacing * spacing;
    int localIndex = 0;

    // x-y, z=-z0 face
    if (domain_boundary_type[3] != 0) {
      pos_z = domain[0][2];

      pos_x = domain[0][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[2] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[2] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[1] != 0 && domain_boundary_type[2] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, -sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_x += 0.5 * spacing;
      while (pos_x < domain[1][0] - 1e-5) {
        pos_y = domain[0][1];
        if (domain_boundary_type[4] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_y += 0.5 * spacing;
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, 0.0, 1.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }

        pos_y = domain[1][1];
        if (domain_boundary_type[1] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_x += spacing;
      }

      pos_x = domain[1][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[0] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(-sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, -sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }
    }

    pos_z = domain[0][2] + spacing / 2.0;
    while (pos_z < domain[1][2] - 1e-5) {
      pos_y = domain[0][1];
      pos_x = domain[0][0];
      if (domain_boundary_type[2] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[2] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(1.0, 0.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[1] != 0 && domain_boundary_type[2] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_x += 0.5 * spacing;
      while (pos_x < domain[1][0] - 1e-5) {
        pos_y = domain[0][1];
        if (domain_boundary_type[4] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, 1.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
        }

        pos_y += spacing / 2.0;
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(1.0, 0.0, 0.0);
          insert_particle(_pos, 0, spacing, normal, 0, vol);
          pos_y += spacing;
        }

        pos_y = domain[1][1];
        if (domain_boundary_type[1] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, -1.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
        }

        pos_x += spacing;
      }

      pos_y = domain[0][1];
      pos_x = domain[1][0];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[0] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(-1.0, 0.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_z += spacing;
    }

    // x-y, z=+z0 face
    if (domain_boundary_type[5] != 0) {
      pos_z = domain[1][2];

      pos_x = domain[0][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[2] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[2] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[1] != 0 && domain_boundary_type[2] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, -sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_x += 0.5 * spacing;
      while (pos_x < domain[1][0] - 1e-5) {
        pos_y = domain[0][1];
        if (domain_boundary_type[4] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_y += 0.5 * spacing;
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, 0.0, -1.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }

        pos_y = domain[1][1];
        if (domain_boundary_type[1] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, -sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_x += spacing;
      }

      pos_x = domain[1][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[0] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(-sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, -sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }
    }
  }
}

void particle_geometry::generate_rigid_body_surface_particle() {
  int num_rigid_body = rb_mgr->get_rigid_body_num();
  for (int i = 0; i < num_rigid_body; i++) {
  }
}

void particle_geometry::uniform_refine() {}

void particle_geometry::adaptive_refine() {}

void particle_geometry::insert_particle(const vec3 &_pos, int _particle_type,
                                        const double _spacing,
                                        const vec3 &_normal,
                                        int _adaptive_level, double _volume,
                                        bool _rigid_body_particle,
                                        int _rigid_body_index, vec3 _p_coord) {
  int idx = is_gap_particle(_pos, _spacing, _rigid_body_index);

  if (idx == -2) {
    current_local_managing_particle_coord->push_back(_pos);
    current_local_managing_particle_normal->push_back(_normal);
    current_local_managing_particle_p_coord->push_back(_p_coord);
    current_local_managing_particle_spacing->push_back(_spacing);
    current_local_managing_particle_volume->push_back(_volume);
    current_local_managing_particle_type->push_back(_particle_type);
    current_local_managing_particle_adaptive_level->push_back(_adaptive_level);
    current_local_managing_particle_new_added->push_back(1);
    current_local_managing_particle_attached_rigid_body->push_back(
        _rigid_body_index);
  } else if (idx > -1) {
    local_managing_gap_particle_coord->push_back(_pos);
    local_managing_gap_particle_normal->push_back(_normal);
    local_managing_gap_particle_p_coord->push_back(_p_coord);
    local_managing_gap_particle_volume->push_back(_volume);
    local_managing_gap_particle_spacing->push_back(_spacing);
    local_managing_gap_particle_particle_type->push_back(_particle_type);
    local_managing_gap_particle_adaptive_level->push_back(_adaptive_level);
  }
}

void particle_geometry::split_field_particle() {}

void particle_geometry::split_rigid_body_surface_particle() {}

void particle_geometry::split_gap_particle() {}

int particle_geometry::is_gap_particle(const vec3 &_pos, double _spacing,
                                       int _attached_rigid_body_index) {
  int rigid_body_num = rb_mgr->get_rigid_body_num();
  for (size_t idx = 0; idx < rigid_body_num; idx++) {
    int rigid_body_type = rb_mgr->get_rigid_body_type(idx);
    vec3 rigid_body_pos = rb_mgr->get_position(idx);
    vec3 rigid_body_ori = rb_mgr->get_orientation(idx);
    double rigid_body_size = rb_mgr->get_rigid_body_size(idx);
    switch (rigid_body_type) {
    case 1:
      // circle in 2d, sphere in 3d
      {
        vec3 dis = _pos - rigid_body_pos;
        if (_attached_rigid_body_index >= 0) {
          // this is a particle on the rigid body surface
        } else {
          // this is a fluid particle

          if (dis.mag() < rigid_body_size - 1.5 * _spacing) {
            return -1;
          }
          if (dis.mag() <= rigid_body_size + 0.1 * _spacing) {
            return idx;
          }

          if (dis.mag() < rigid_body_size + 1.5 * _spacing) {
            double min_dis = bounding_box_size[0];
            for (int i = 0; i < rigid_body_surface_particle.size(); i++) {
              vec3 rci = _pos - rigid_body_surface_particle[i];
              if (min_dis > rci.mag()) {
                min_dis = rci.mag();
              }
            }

            if (min_dis < 0.25 * _spacing) {
              // this is a gap particle near the surface of the colloids
              return idx;
            }
          }
        }
      }
      break;

    case 2:
      // square in 2d, cubic in 3d
      {
        if (dim == 2) {
          double half_side_length = rigid_body_size;
          double theta = rigid_body_ori[0];

          vec3 abs_dis = _pos - rigid_body_pos;
          // rotate back
          vec3 dis =
              vec3(cos(theta) * abs_dis[0] + sin(theta) * abs_dis[1],
                   -sin(theta) * abs_dis[0] + cos(theta) * abs_dis[1], 0.0);
          if (_attached_rigid_body_index >= 0) {
            // this is a particle on the rigid body surface
          } else {
            if (abs(dis[0]) < half_side_length - 1.5 * _spacing &&
                abs(dis[1]) < half_side_length - 1.5 * _spacing) {
              return -1;
            }
            if (abs(dis[0]) < half_side_length + 0.1 * _spacing &&
                abs(dis[1]) < half_side_length + 0.1 * _spacing) {
              return idx;
            }
            if (abs(dis[0]) < half_side_length + 1.0 * _spacing &&
                abs(dis[1]) < half_side_length + 1.0 * _spacing) {
              double min_dis = bounding_box_size[0];
              for (int i = 0; i < rigid_body_surface_particle.size(); i++) {
                vec3 rci = _pos - rigid_body_surface_particle[i];
                if (min_dis > rci.mag()) {
                  min_dis = rci.mag();
                }
              }

              if (min_dis < 0.25 * _spacing) {
                // this is a gap particle near the surface of the colloids
                return idx;
              }
            }
          }
        }
        if (dim == 3) {
        }
      }
      break;
    case 3:
      // triangle in 2d, tetrahedron in 3d
      if (dim == 2) {
        double side_length = rigid_body_size;
        double height = (sqrt(3) / 2.0) * side_length;
        double theta = rigid_body_ori[0];

        vec3 translation = vec3(0.0, sqrt(3) / 6.0 * side_length, 0.0);
        vec3 abs_dis = _pos - rigid_body_pos;
        // rotate back
        vec3 dis =
            vec3(cos(theta) * abs_dis[0] + sin(theta) * abs_dis[1],
                 -sin(theta) * abs_dis[0] + cos(theta) * abs_dis[1], 0.0) +
            translation;

        bool possible_gap_particle = false;
        bool gap_particle = false;

        double enlarged_xlim_low = -0.5 * side_length - 0.5 * _spacing;
        double enlarged_xlim_high = 0.5 * side_length + 0.5 * _spacing;
        double enlarged_ylim_low = -0.5 * _spacing;
        double enlarged_ylim_high = height + 0.5 * _spacing;
        double exact_xlim_low = -0.5 * side_length;
        double exact_xlim_high = 0.5 * side_length;
        double exact_ylim_low = 0.0;
        double exact_ylim_high = height;
        if (_attached_rigid_body_index >= 0) {
          // this is a particle on the rigid body surface
        } else {
          if ((dis[0] > enlarged_xlim_low && dis[0] < enlarged_xlim_high) &&
              (dis[1] > enlarged_ylim_low && dis[1] < enlarged_ylim_high)) {
            // this is a possible particle in the gap region of the triangle
            double dis_x = min(abs(dis[0] - exact_xlim_low),
                               abs(dis[0] - exact_xlim_high));
            double dis_y = sqrt(3) * dis_x;
            if (dis[1] < 0 && dis[1] > -0.1 * _spacing) {
              gap_particle = true;
            } else if (dis[1] > 0) {
              if (dis[0] < exact_xlim_low || dis[0] > exact_xlim_high) {
                possible_gap_particle = true;
              } else if (dis[1] < dis_y + 0.1 * _spacing) {
                gap_particle = true;
              } else if (dis[1] < dis_y + 0.5 * _spacing) {
                possible_gap_particle = true;
              }
            }
            possible_gap_particle = true;
          }

          if (gap_particle) {
            return idx;
          } else if (possible_gap_particle) {
            double min_dis = bounding_box_size[0];
            for (int i = 0; i < rigid_body_surface_particle.size(); i++) {
              vec3 rci = _pos - rigid_body_surface_particle[i];
              if (min_dis > rci.mag()) {
                min_dis = rci.mag();
              }
            }

            if (min_dis < 0.25 * _spacing) {
              // this is a gap particle near the surface of the colloids
              return idx;
            }
          }
        }
      }
      if (dim == 3) {
      }
      break;
    }
  }

  return -2;
}

void particle_geometry::index_particle() {
  int local_particle_num = current_local_managing_particle_coord->size();
  current_local_managing_particle_index->resize(local_particle_num);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  vector<int> particle_offset(size + 1);
  vector<int> particle_num(size);
  MPI_Allgather(&local_particle_num, 1, MPI_INT, particle_num.data(), 1,
                MPI_INT, MPI_COMM_WORLD);

  particle_offset[0] = 0;
  for (int i = 0; i < size; i++) {
    particle_offset[i + 1] = particle_offset[i] + particle_num[i];
  }

  vector<long long> &index = *current_local_managing_particle_index;
  for (int i = 0; i < local_particle_num; i++) {
    index[i] = i + particle_offset[rank];
  }
}

void particle_geometry::balance_workload() {
  // use zoltan2 to build a solution to partition
  vector<int> result;
  partitioner.partition(*current_local_managing_particle_index,
                        *current_local_managing_particle_coord, result);

  // use the solution to build the mitigation list
}