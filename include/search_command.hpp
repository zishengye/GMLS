#ifndef _SEARCH_COMMAND_HPP_
#define _SEARCH_COMMAND_HPP_

#include <string>

template <typename T>
int SearchCommand(int argc, char **argv, const std::string &commandName,
                  T &res) {
  int i;
  for (i = 1; i < argc; i++) {
    if (commandName.compare(argv[i]) == 0) {
      break;
    }
  }

  if (i != argc) {
    std::stringstream converter(argv[i + 1]);
    converter >> res;

    return 0;
  } else {
    return 1;
  }
}

#endif