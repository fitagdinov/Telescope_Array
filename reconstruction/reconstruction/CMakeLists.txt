
cmake_minimum_required(VERSION 3.9)
project(trig)


# header files
GLOB_FILES(sdtrgbk_headers inc/*.h)
file(COPY ${sdtrgbk_headers} DESTINATION ${INCLUDE_OUTPUT_PATH})

# sdtrgbk executable file
BINARY_FROM_SOURCES(SOURCES src/sdtrgbk.cpp
  src/sdtrgbkAnalysis.cpp src/sdtrgbk_util.cpp src/sdinfo_class.cpp
  LIBS sduti dst2k ${BZIP2_LIBRARY} ${ZLIB_LIBRARY} m)

# single source file programs related to trigger
set(additional_trig_prog_sources
  src/cntmsd.cpp
  src/pnotrig.cpp
  src/remhytrig.cpp
  src/remnotrig.cpp
  src/trigp.cpp
  src/trigt.cpp)
foreach(additional_trig_prog_source ${additional_trig_prog_sources})
  BINARY_FROM_SOURCES(SOURCES ${additional_trig_prog_source}
    LIBS sduti dst2k ${BZIP2_LIBRARY} ${ZLIB_LIBRARY} m)
endforeach(additional_trig_prog_source ${additional_trig_prog_sources})

  
# sdtrgbk scripts
GLOB_FILES(sdtrgbk_scripts py/*.py sh/*.sh)
file(COPY ${sdtrgbk_scripts} DESTINATION ${EXECUTABLE_OUTPUT_PATH} USE_SOURCE_PERMISSIONS)
