UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  CXX = g++
  CXXFLAGS = -std=c++17 -O3 -I/opt/homebrew/include/eigen3 -I./libigl/include -I/opt/homebrew/include
  LDFLAGS = -L/opt/homebrew/lib -lgmp -lmpfr
  OMP_CXXFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
  OMP_LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp
else
  CXX = g++
  CXXFLAGS = -std=c++17 -O3 -I/usr/include/eigen3 -I./libigl/include
  LDFLAGS = -lgmp -lmpfr
  OMP_CXXFLAGS = -fopenmp
  OMP_LDFLAGS = -fopenmp
endif

TARGETS = tpms_thickness tpms_boundary tpms_matan tpms_unified tpms_interpolated visualize_inout

all: $(TARGETS)

tpms_thickness: TPMS_Generation_with_thickness.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

tpms_boundary: TPMS_with_boundary_mesh.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

tpms_matan: TPMS_matan.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

tpms_unified: tpms_unified.cpp
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) $< -o $@ $(LDFLAGS) $(OMP_LDFLAGS)

tpms_interpolated: tpms_interpolated.cpp
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) $< -o $@ $(LDFLAGS) $(OMP_LDFLAGS)

visualize_inout: visualize_inout.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)