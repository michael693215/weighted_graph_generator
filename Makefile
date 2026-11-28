CXX ?= g++
MPICXX ?= mpicxx
CXXFLAGS = -std=c++17 -O3
SRCS = driver.cpp serial_dijkstra.cpp parallel_dijkstra.cpp graph.cpp
TARGET = driver

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    OMP_FLAGS = -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include -L$(OMP_PREFIX)/lib -lomp
else
    OMP_FLAGS = -fopenmp
endif

all:
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o $(TARGET) $(SRCS)

mpi:
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) -DUSE_MPI -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)