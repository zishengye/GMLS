#include "get_input_file.hpp"
#include "gmls_solver.hpp"
#include "rigid_body_surface_particle_hierarchy.hpp"
#include "search_command.hpp"
#include "trilinos_wrapper.hpp"

#include <Compadre_KokkosParser.hpp>

using namespace std;
using namespace Compadre;

int main(int argc, char *argv[]) {
  // get info from input file
  string inputFileName;
  vector<char *> cstrings;
  vector<string> strings;
  SearchCommand<string>(argc, argv, "-input", inputFileName);

  for (int i = 1; i < argc; i++) {
    cstrings.push_back(argv[i]);
  }

  if (!GetInputFile(inputFileName, strings, cstrings)) {
    return -1;
  }

  int inputCommandCount = cstrings.size();
  char **inputCommand = cstrings.data();

  auto kp = KokkosParser(argc, argv, true);

  string rigid_body_input_file_name;
  SearchCommand<string>(inputCommandCount, inputCommand, "-rigid_body_input",
                        rigid_body_input_file_name);

  int dim;
  SearchCommand<int>(inputCommandCount, inputCommand, "-Dim", dim);

  RigidBodyManager rb_mgr;
  rb_mgr.init(rigid_body_input_file_name, dim);

  rigid_body_surface_particle_hierarchy hierarchy;
  hierarchy.init(make_shared<RigidBodyManager>(rb_mgr), dim);

  hierarchy.set_coarse_level_resolution(0.2);

  shared_ptr<vector<Vec3>> coord_ptr;
  shared_ptr<vector<Vec3>> normal_ptr;
  shared_ptr<vector<Vec3>> spacing_ptr;

  hierarchy.get_coarse_level_coordinate(0, coord_ptr);
  hierarchy.get_coarse_level_normal(0, normal_ptr);
  hierarchy.get_coarse_level_spacing(0, spacing_ptr);

  cout << coord_ptr->size() << endl;

  vector<int> refined_particle_index;
  for (int i = 0; i < coord_ptr->size(); i++) {
    hierarchy.find_refined_particle(0, 0, i, refined_particle_index);
    cout << refined_particle_index.size() << endl;
  }

  hierarchy.write_log();

  kp.finalize();

  return 0;
}