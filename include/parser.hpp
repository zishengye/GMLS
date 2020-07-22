#ifndef _PARSER_HPP_
#define _PARSER_HPP_

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

class parser {
private:
  std::vector<std::string> _strings;
  std::vector<char *> _cstrings;

public:
  parser() {}
  parser(int argc, char *argv[]) { add_command(argc, argv); }

  bool add_input_file(std::string input_file_name) {
    std::ifstream input_file(input_file_name);

    if (input_file.fail()) {
      std::cout << "input file does not exsist" << std::endl;
      return false;
    }

    while (!input_file.eof()) {
      std::string str;
      input_file >> str;
      _strings.push_back(str);
    }

    input_file.close();

    return true;
  }

  void add_command(int argc, char *argv[]) {
    const std::string input_file_str = "-input";
    for (int i = 0; i < argc; i++) {
      if (input_file_str.compare(argv[i]) == 0) {
        add_input_file(static_cast<std::string>(argv[++i]));
      } else {
        _strings.push_back(argv[i]);
      }
    }
  }

  template <typename T> bool search(const std::string &command, T &res) {
    for (auto p_str = _strings.begin(); p_str != _strings.end(); p_str++) {
      if (*p_str == command) {
        std::stringstream converter(*(++p_str));
        converter >> res;

        return true;
      }
    }

    return false;
  }

  char **data() {
    _cstrings.clear();
    for (auto str : _strings) {
      _cstrings.push_back(const_cast<char *>(str.c_str()));
    }

    return _cstrings.data();
  }

  int size() { return _strings.size(); }
};

#endif