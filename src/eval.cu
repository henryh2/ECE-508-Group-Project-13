/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include <fstream>

#include "helper.hpp"
#include "cdlp.hu"

#include "pangolin/pangolin.hpp"

#ifndef GRAPH_PREFIX_PATH
#error "define GRAPH_PREFIX_PATH"
#endif

namespace gpu_algorithms_labs_evaluation {

enum Algorithm { GPU_CDLP = 1, GPU_LCC = 2 };

static std::vector<uint32_t> import_solution(const std::string& sol_path)
{
  std::ifstream sol_file(sol_path, std::ifstream::in);
  std::vector<uint32_t> solution;

  while(sol_file.good())
  {
    uint32_t node, sol;
    sol_file >> node;
    sol_file >> sol;

    solution.push_back(sol);
  }
  sol_file.close();

  return solution;
}

static pangolin::EdgeList import_edgelist(const std::string& edge_path)
{
  std::ifstream input_file(edge_path, std::ifstream::in);
  pangolin::EdgeList edge_list;

  while(input_file.good())
  {
    uint32_t src, dst;
    input_file >> src;
    input_file >> dst;
    
    pangolin::Edge e(src, dst);
    edge_list.push_back(e);
  }
  input_file.close();

  return edge_list;
}

static int eval(int iterations, const std::string& edge_path, const std::string& sol_path, const Algorithm algo) 
{
  pangolin::logger::set_level(pangolin::logger::Level::TRACE);

  // Imports the graph
  timer_start("Reading graph data");
  pangolin::EdgeList edgeList = import_edgelist(edge_path);
  std::vector<uint32_t> expected = import_solution(sol_path);
  timer_stop();

  // Built the graph in CSR format
  timer_start("Building unified memory CSR");
  pangolin::COO<uint32_t> coo = pangolin::COO<uint32_t>::from_edgelist(edgeList);
  timer_stop();

  // Compute the DCLP
  timer_start("Performing CDLP");
  std::vector<uint32_t> actual;
  switch(algo)
  {
    case GPU_CDLP:
      actual = CDLP(coo.view(), iterations);
      break;

    case GPU_LCC:
      break;
  }
  timer_stop();

  // Validate the solution to the algorithm
  timer_start("Validating CDLP");
  REQUIRE(expected.size() == actual.size());
  for(uint32_t i = 1; i < expected.size(); ++i)
    REQUIRE(expected[i] == actual[i]);
  timer_stop();

  return 0;
}
  

TEST_CASE("directed-test", "") 
{
  SECTION("CDLP") { eval(2, GRAPH_PREFIX_PATH "/test-cdlp-directed.e", GRAPH_PREFIX_PATH "/test-cdlp-directed-CDLP", GPU_CDLP); }
  SECTION("LCC") { eval(0, GRAPH_PREFIX_PATH "/test-lcc-directed.e", GRAPH_PREFIX_PATH "/test-lcc-directed-LCC", GPU_LCC); }
}

TEST_CASE("undirected-test", "") 
{
  SECTION("CDLP") { eval(2, GRAPH_PREFIX_PATH "/test-cdlp-undirected.e", GRAPH_PREFIX_PATH "/test-cdlp-undirected-CDLP", GPU_CDLP); }
  SECTION("LCC") { eval(0, GRAPH_PREFIX_PATH "/test-lcc-directed.e", GRAPH_PREFIX_PATH "/test-cdlp-directed-LCC", GPU_LCC); }
}

} // namespace gpu_algorithms_labs_evaluation
