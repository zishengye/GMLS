#ifndef _GET_INPUT_FILE_HPP_
#define _GET_INPUT_FILE_HPP_

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static bool GetInputFile(std::string inputFileName,
                         std::vector<std::string> &strings,
                         std::vector<char *> &cstrings) {
  std::ifstream inputFile(inputFileName);

  if (inputFile.fail()) {
    std::cout << "input file does not exsist" << std::endl;
    return false;
  }

  while (!inputFile.eof()) {
    std::string str;
    inputFile >> str;
    strings.push_back(str);
  }

  inputFile.close();

  for (auto &string : strings) {
    cstrings.push_back(&string.front());
  }

  return true;
}

#endif