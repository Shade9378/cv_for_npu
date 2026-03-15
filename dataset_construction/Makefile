# -------- compiler --------
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
OPENCV   := `pkg-config --cflags --libs opencv4`

# -------- targets --------
TARGETS := add_overlay batch_maker

# -------- sources --------
ADD_OVERLAY_SRC := add_overlay.cpp
BATCH_MAKER_SRC := batch_maker.cpp

# -------- rules --------
all: $(TARGETS)

add_overlay: $(ADD_OVERLAY_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

batch_maker: $(BATCH_MAKER_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
