set(MTS_PLUGIN_PREFIX "phasefunctions")

add_plugin(hg hg.cpp)
add_plugin(isotropic isotropic.cpp)
add_plugin(rayleigh rayleigh.cpp)
add_plugin(lut lut.cpp)
add_plugin(blendphase blendphase.cpp)

# Register the test directory
add_tests(${CMAKE_CURRENT_SOURCE_DIR}/tests)
