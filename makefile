CXX = g++
CONDA_PREFIX ?= $(shell echo $$CONDA_PREFIX)
CXXFLAGS = -std=c++17 -O3 -I$(CONDA_PREFIX)/include/eigen3 -I$(CONDA_PREFIX)/include -I./libigl/include
LDFLAGS = -L$(CONDA_PREFIX)/lib -lgmp -lmpfr

TARGETS = tpms_matan tpms_spatial

all: $(TARGETS)

tpms_matan: TPMS_matan.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

tpms_spatial: TPMS_spatial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)