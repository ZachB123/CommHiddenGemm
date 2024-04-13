CXX = mpicxx
CXXFLAGS =
LDFLAGS =
OMPFLAGS = -fopenmp

hello-hpc-cluster$(EXE): hello-hpc-cluster.cxx
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f hello-hpc-cluster$(EXE)

# eof
