#include "graph.hpp"
#include <cstdint>
#include <iostream>

int main() {
    Graph g(50, 100);
    for (std::int64_t i = 0; i < 50; ++i) {
        for (std::int64_t j = 0; j < 50; ++j) {
            std::cout << g.weights[i * 50 + j] << ' ';
        }
        std::cout << '\n'; 
    }
    return 0;
}