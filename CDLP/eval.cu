/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include "helper.hpp"
#include "template.hu"

#include "pangolin/pangolin.hpp"

#ifndef GRAPH_PREFIX_PATH
#error "define GRAPH_PREFIX_PATH"
#endif

namespace gpu_algorithms_labs_evaluation {

enum Mode { LINEAR = 1, OTHER = 2 };

static int eval(const std::string &path, const Mode &mode) {

  assert(endsWith(path, ".bel") && "graph should be a .bel file");

  pangolin::logger::set_level(pangolin::logger::Level::TRACE);

  timer_start("Reading graph data");
  // create a reader for .bel files
  pangolin::BELReader reader(path);
  // read all edges from the file
  pangolin::EdgeList edgeList = reader.read_all();
  timer_stop(); // reading graph data

  timer_start("building unified memory CSR");
  // build a csr/coo matrix from the edge list
  pangolin::COO<uint32_t> coo = pangolin::COO<uint32_t>::from_edgelist(
    edgeList);
  timer_stop(); // building unified memory CSR

  timer_start("Doing CDLP");
  int iterations = 2;
  std::vector<uint32_t> actual = CDLP(coo.view(),iterations);
  timer_stop();
  return 0;
}
  

TEST_CASE("graph500-scale18-ef16_adj", "") {
  SECTION("LINEAR") { eval(GRAPH_PREFIX_PATH "/graph500-scale18-ef16_adj.bel", LINEAR); }
}

TEST_CASE("amazon0302_adj", "") {
  SECTION("LINEAR") { eval(GRAPH_PREFIX_PATH "/amazon0302_adj.bel", LINEAR); }
}

TEST_CASE("roadNet-CA_adj", "") {
  SECTION("LINEAR") { eval(GRAPH_PREFIX_PATH "/roadNet-CA_adj.bel", LINEAR); }
}


} // namespace gpu_algorithms_labs_evaluation
