#include "Ume/Datastore.hh"

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

using sptr = Ume::Datastore::sptr;
using wptr = std::weak_ptr<Ume::Datastore>;
using wlist = std::vector<wptr>;

void flatten(sptr curr, wlist &nodes) {
  nodes.push_back(curr);
  for (auto &c : curr->children_) {
    flatten(c, nodes);
  }
}

TEST_CASE("DS ROOT", "[Datastore]") {
  sptr root = Ume::Datastore::create_root();
  wlist nodes;
  flatten(root, nodes);
  REQUIRE(nodes.size() == 1);
  REQUIRE(nodes[0].use_count() == 1);
  sptr cpy = root->getptr();
  REQUIRE(nodes[0].use_count() == 2);
  cpy.reset();
  REQUIRE(nodes[0].use_count() == 1);
  root.reset();
  REQUIRE(nodes[0].expired());
}

TEST_CASE("DS ROOT+1", "[Datastore]") {
  sptr root = Ume::Datastore::create_root();
  { sptr child1 = Ume::Datastore::create_child(root); }
  wlist nodes;
  flatten(root, nodes);

  REQUIRE(nodes.size() == 2);
  for (auto &n : nodes) {
    REQUIRE(n.use_count() == 1);
  }
  root.reset();
  for (auto &n : nodes) {
    REQUIRE(n.expired());
  }
}

TEST_CASE("DS ROOT+2", "[Datastore]") {
  sptr root = Ume::Datastore::create_root();
  {
    sptr child1 = Ume::Datastore::create_child(root);
    sptr child2 = Ume::Datastore::create_child(root);
  }
  wlist nodes;
  flatten(root, nodes);

  REQUIRE(nodes.size() == 3);
  for (auto &n : nodes) {
    REQUIRE(n.use_count() == 1);
  }
  root.reset();
  for (auto &n : nodes) {
    REQUIRE(n.expired());
  }
}

TEST_CASE("DS ROOT+1+1", "[Datastore]") {
  sptr root = Ume::Datastore::create_root();
  {
    sptr child1 = Ume::Datastore::create_child(root);
    sptr child2 = Ume::Datastore::create_child(child1);
  }
  wlist nodes;
  flatten(root, nodes);

  REQUIRE(nodes.size() == 3);
  for (auto &n : nodes) {
    REQUIRE(n.use_count() == 1);
  }
  root.reset();
  for (auto &n : nodes) {
    REQUIRE(n.expired());
  }
}
