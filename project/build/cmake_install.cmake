# Install script for directory: /home/emmanuel/projects/musicgen/scripts/torch/proj

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/emmanuel/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/mrnn/scm-1/lua/mrnn" TYPE FILE FILES
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/models/SimpleModel.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/models/FastModel.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/models/Model.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/init.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/main.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/utils/model_utils.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/utils/tools.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/opt.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/modules/LSTM.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/data/DataSet.lua"
    "/home/emmanuel/projects/musicgen/scripts/torch/proj/view.lua"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/emmanuel/projects/musicgen/scripts/torch/proj/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
