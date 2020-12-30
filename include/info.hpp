#ifndef _INFO_HPP_
#define _INFO_HPP_

#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <string>
#include <vector>

#include "Compadre_GMLS.hpp"

#include "vec3.hpp"

template <class T> class infoEntity {
private:
  std::map<std::string, T *> __entity;

public:
  infoEntity() {}

  T &Register(std::string entityName) {
    __entity.insert(std::make_pair(entityName, new T));

    return *__entity.at(entityName);
  }

  T &Register(std::string entityName, T *entity) {
    __entity.insert(std::make_pair(entityName, entity));

    return *entity;
  }

  T &GetHandle(std::string entityName) { return *__entity.at(entityName); }
  T *GetPointer(std::string entityName) { return __entity.at(entityName); }

  ~infoEntity() {
    for (auto it = __entity.begin(); it != __entity.end(); it++) {
      delete it->second;
    }
  }
};

typedef infoEntity<Compadre::GMLS *> gmlsInfo;

struct GeneralInfo {
  infoEntity<std::vector<vec3>> vector;
  infoEntity<std::vector<double>> scalar;
  infoEntity<std::vector<int>> index;
};

struct QueueInfo {
  infoEntity<std::list<vec3>> vector;
  infoEntity<std::list<double>> scalar;
  infoEntity<std::list<int>> index;
};

#endif