/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include <fstream>

#include "helper.hpp"
#include "cdlp.hu"
#include "lcc.hu"

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

static std::vector<float> import_solution_LCC(const std::string& sol_path)
{
  std::ifstream sol_file(sol_path, std::ifstream::in);
  std::vector<float> solution;

  while(sol_file.good())
  {
    uint32_t node;
    float sol;
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

  pangolin::EdgeList edgeList = import_edgelist(edge_path);

  // Built the graph in CSR format
  timer_start("Building unified memory CSR");
  pangolin::COO<uint32_t> coo = pangolin::COO<uint32_t>::from_edgelist(edgeList);
  timer_stop();

  // Imports the graph
  // timer_start("Reading graph data");
  
  switch(algo) {
    case GPU_CDLP:
    {
      timer_start("Reading graph data");
      std::vector<uint32_t> expected = import_solution(sol_path);
      timer_stop();
      // Compute the CDLP
      timer_start("Performing CDLP");
      std::vector<uint32_t> actual;
      actual = CDLP(coo.view(), iterations);
      timer_stop();
      // Validate the solution to the algorithm
      timer_start("Validating CDLP");
      REQUIRE(expected.size() == actual.size());
      for(uint32_t i = 1; i < expected.size(); ++i)
        REQUIRE(expected[i] == actual[i]);
      timer_stop();

      break;
    }
    case GPU_LCC:
      timer_start("Reading graph data");
      std::vector<float> expected_lcc = import_solution_LCC(sol_path);
      timer_stop();
      // Compute the LCC
      timer_start("Performing LCC");
      std::vector<float> actual_lcc;

      printf("num_rows: %u\n", coo.num_rows());

      printf("Row index:\n");
      for(int i = 0; i < coo.nnz(); i++) {
        printf("%d ", coo.row_ind()[i]);
      }

      printf("\nCol index:\n");
      for(int i = 0; i < coo.nnz(); i++) {
        printf("%d ", coo.col_ind()[i]);
      }

      printf("\nRow ptr:\n");
      for(int i = 0; i <= coo.num_rows(); i++) {
        printf("%d ", coo.row_ptr()[i]);
      }

      // printf("num rows: %d\n", coo.num_rows());
      // actual_lcc = LCC(coo.view(), coo.num_rows());
      actual_lcc = LCC(coo.view(), 11);

      // actual_lcc = LCC(coo.view(), 11);
      // actual_lcc = LCC(coo.view());

      printf("\nActual LCC: \n");
      for(int i = 0; i < actual_lcc.size(); i++) {
        printf("%d: %f\n", i, actual_lcc[i]);
      }

      // Validate the solution to the algorithm
      timer_start("Validating LCC");
      REQUIRE(expected_lcc.size() == actual_lcc.size());
      for(uint32_t i = 1; i < expected_lcc.size(); ++i)
        REQUIRE(expected_lcc[i] == actual_lcc[i]);
      timer_stop();
  }
  
  // timer_stop();

  // // Compute the DCLP
  // timer_start("Performing CDLP");
  // std::vector<uint32_t> actual;
  // switch(algo)
  // {
  //   case GPU_CDLP:
  //     actual = CDLP(coo.view(), iterations);
  //     break;

  //   case GPU_LCC:
  //     actual = LCC(coo.view(), coo.num_rows());
  //     break;
  // }
  // timer_stop();

  // // Validate the solution to the algorithm
  // timer_start("Validating CDLP");
  // REQUIRE(expected.size() == actual.size());
  // for(uint32_t i = 1; i < expected.size(); ++i)
  //   REQUIRE(expected[i] == actual[i]);
  // timer_stop();

  return 0;
}
  

TEST_CASE("directed-test", "") 
{
  // SECTION("CDLP") { eval(2, GRAPH_PREFIX_PATH "/test-cdlp-directed.e", GRAPH_PREFIX_PATH "/test-cdlp-directed-CDLP", GPU_CDLP); }
  // SECTION("LCC") { eval(0, GRAPH_PREFIX_PATH "/test-lcc-directed.e", GRAPH_PREFIX_PATH "/test-lcc-directed-LCC", GPU_LCC); }
}

TEST_CASE("undirected-test", "") 
{
  // SECTION("CDLP") { eval(2, GRAPH_PREFIX_PATH "/test-cdlp-undirected.e", GRAPH_PREFIX_PATH "/test-cdlp-undirected-CDLP", GPU_CDLP); }
  SECTION("LCC") { eval(0, GRAPH_PREFIX_PATH "/test-lcc-undirected.e", GRAPH_PREFIX_PATH "/test-lcc-undirected-LCC", GPU_LCC); }
}

} // namespace gpu_algorithms_labs_evaluation
