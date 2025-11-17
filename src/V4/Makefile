# Makefile for KLT Tracker with OpenACC support
# Optimized for NVIDIA RTX 3080 (Compute Capability 8.6)

# Compiler selection
CC = nvc
CXX = nvc++

# Compiler flags
CFLAGS = -O3 -Wall
CXXFLAGS = -O3 -Wall

# OpenACC flags for RTX 3080 (SM 8.6)

ACC_FLAGS = -acc -gpu=cc86 -Minfo=accel -fast


# Additional optimization flags
OPT_FLAGS = -fast

# Math library
LIBS = -lm

# Directories
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = bin

# Create directories if they don't exist
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# Source files (only example6.c contains main, exclude main.cpp)
C_SOURCES = convolve.c \
            error.c \
            klt.c \
            klt_util.c \
            pnmio.c \
            pyramid.c \
            selectGoodFeatures.c \
            storeFeatures.c \
            trackFeatures.c \
            writeFeatures.c \
            example6.c

# Object files
C_OBJECTS = $(C_SOURCES:%.c=$(OBJ_DIR)/%.o)
OBJECTS = $(C_OBJECTS)

# Header files (for dependency tracking)
HEADERS = base.h \
          convolve.h \
          error.h \
          klt.h \
          klt_util.h \
          pnmio.h \
          pyramid.h

# Target executable
TARGET = $(BIN_DIR)/klt_tracker

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $(ACC_FLAGS) $(OPT_FLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $@"

# Compile C source files with OpenACC
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(ACC_FLAGS) $(OPT_FLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR)/*.o $(TARGET)
	@echo "Clean complete"

# Deep clean (remove all generated directories)
distclean: clean
	@echo "Removing build directories..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Deep clean complete"

# Run the program
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: CXXFLAGS += -g -DDEBUG
debug: ACC_FLAGS += -g
debug: OPT_FLAGS = 
debug: clean $(TARGET)
	@echo "Debug build complete"

# Profile build with OpenACC profiling info
profile: CFLAGS += -pg
profile: CXXFLAGS += -pg
profile: ACC_FLAGS += -Minfo=accel
profile: $(TARGET)
	@echo "Profile build complete"

# Help target
help:
	@echo "KLT Tracker with OpenACC - Makefile Help"
	@echo "========================================="
	@echo "Optimized for NVIDIA RTX 3080 (SM 8.6)"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the project (default)"
	@echo "  clean     - Remove object files and executable"
	@echo "  distclean - Remove all generated directories"
	@echo "  run       - Build and run the program"
	@echo "  debug     - Build with debug symbols"
	@echo "  profile   - Build with profiling enabled"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  NVIDIA HPC SDK (nvc/nvc++ compiler)"
	@echo "  NVIDIA RTX 3080 or compatible GPU"

# Phony targets
.PHONY: all clean distclean run debug profile help

# Print variables (for debugging the Makefile)
print-%:
	@echo $* = $($*)
