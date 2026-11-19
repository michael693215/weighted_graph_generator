// g++ -o exec main.cpp graph.cpp
// uncomment tests to run them
#include "graph.hpp"
#include <cstdint>
#include <iostream>

int main() {
    /* =============== 50 * 50 graph with weights spanning from 1 to 100 =============== */
    /*
    Graph g1(50, 100);
    for (int i = 0; i < 50; ++i) {
        for (std::int64_t j = 0; j < 50; ++j) {
            std::cout << g1.weights[i * 50 + j] << ' '; 
        }
        std::cout << '\n'; 
    }
    */
    /* =============== END TEST =============== */

    /* =============== all the weights will be 2 for this graph =============== */
    /*
    Graph g2(50, 100, 1, 0, 1, 2);
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 50; ++j) {
            std::cout << g2.weights[i * 50 + j] << ' ';
        }
        std::cout << '\n';
    }
    */
    /* =============== END TEST =============== */

    /* =============== obstacle density will be roughly 50% =============== */
    /*
    Graph g3(50, 100, 0, .5f);
    float obstacles = 0.f;
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 50; ++j) {
            if (!g3.weights[i * 50 + j]) {
                obstacles += 1.f / g3.p.size;
            }
        }
    }
    std::cout << "Obstacle density: " << obstacles << '\n';
    */
    /* =============== END TEST =============== */


    /* =============== same seed will produce the same graph =============== */
    /* 
    Graph g4(50, 100);
    Graph g5(50, 100);
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 50; ++j) {
            if (g4.weights[i * 50 + j] != g5.weights[i * 50 + j]) {
                std::cout << "g4: " << g4.weights[i * 50 + j] << ", g5: " << g5.weights[i * 50 + j];
            }
        }
    }
    */
    /* =============== END TEST =============== */
    return 0;
}