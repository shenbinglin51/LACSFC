# Makefile for compiling the enhanced Hilbert processor C++ library

CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -Wall -Wextra
LDFLAGS = -shared

# Target library
TARGET = enhanced_hilbert_processor.so

# Source files
SOURCES = enhanced_hilbert_processor.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build the shared library
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install (copy to appropriate location)
install: $(TARGET)
	@echo "Library compiled successfully: $(TARGET)"

# Test compilation
test: $(TARGET)
	@echo "Testing library..."
	@python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)'); print('Library loaded successfully')"

.PHONY: all clean install test
