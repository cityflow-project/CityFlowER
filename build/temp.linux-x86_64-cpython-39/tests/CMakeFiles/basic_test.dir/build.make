# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /root/anaconda3/envs/myenv/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /root/anaconda3/envs/myenv/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/realCityFlow/realCityFlow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39

# Include any dependencies generated for this target.
include tests/CMakeFiles/basic_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/basic_test.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/basic_test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/basic_test.dir/flags.make

tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o: tests/CMakeFiles/basic_test.dir/flags.make
tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o: /home/realCityFlow/realCityFlow/tests/cpp/basic_test.cpp
tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o: tests/CMakeFiles/basic_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o"
	cd /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o -MF CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o.d -o CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o -c /home/realCityFlow/realCityFlow/tests/cpp/basic_test.cpp

tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/basic_test.dir/cpp/basic_test.cpp.i"
	cd /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/realCityFlow/realCityFlow/tests/cpp/basic_test.cpp > CMakeFiles/basic_test.dir/cpp/basic_test.cpp.i

tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/basic_test.dir/cpp/basic_test.cpp.s"
	cd /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/realCityFlow/realCityFlow/tests/cpp/basic_test.cpp -o CMakeFiles/basic_test.dir/cpp/basic_test.cpp.s

# Object files for target basic_test
basic_test_OBJECTS = \
"CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o"

# External object files for target basic_test
basic_test_EXTERNAL_OBJECTS =

tests/basic_test: tests/CMakeFiles/basic_test.dir/cpp/basic_test.cpp.o
tests/basic_test: tests/CMakeFiles/basic_test.dir/build.make
tests/basic_test: src/libcityflow_lib.a
tests/basic_test: /usr/lib/x86_64-linux-gnu/libgtest.a
tests/basic_test: /usr/lib/x86_64-linux-gnu/libgtest_main.a
tests/basic_test: /home/realCityFlow/realCityFlow/src/libtorch/lib/libtorch_python.so
tests/basic_test: /usr/lib/x86_64-linux-gnu/libgtest.a
tests/basic_test: tests/CMakeFiles/basic_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable basic_test"
	cd /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/basic_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/basic_test.dir/build: tests/basic_test
.PHONY : tests/CMakeFiles/basic_test.dir/build

tests/CMakeFiles/basic_test.dir/clean:
	cd /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests && $(CMAKE_COMMAND) -P CMakeFiles/basic_test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/basic_test.dir/clean

tests/CMakeFiles/basic_test.dir/depend:
	cd /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/realCityFlow/realCityFlow /home/realCityFlow/realCityFlow/tests /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39 /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests /home/realCityFlow/realCityFlow/build/temp.linux-x86_64-cpython-39/tests/CMakeFiles/basic_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/basic_test.dir/depend

