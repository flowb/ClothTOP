# 3.9.0 is required for CUDA as a first-class language with Visual Studio generator support
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(ClothTOP)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ClothTOP)

################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if(NOT CMAKE_VS_PLATFORM_NAME)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()
message("${CMAKE_VS_PLATFORM_NAME} architecture in use")

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(FATAL_ERROR "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
endif()

################################################################################
# Global configuration types
################################################################################
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
    CACHE STRING "" FORCE
)

################################################################################
# Global compiler options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_C_FLAGS "")
    set(CMAKE_C_FLAGS_DEBUG "")
    set(CMAKE_C_FLAGS_RELEASE "")
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_DEBUG "")
    set(CMAKE_CXX_FLAGS_RELEASE "")
endif()

################################################################################
# Global linker options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_EXE_LINKER_FLAGS "")
    set(CMAKE_MODULE_LINKER_FLAGS "")
    set(CMAKE_SHARED_LINKER_FLAGS "")
    set(CMAKE_STATIC_LINKER_FLAGS "")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS}")
endif()

################################################################################
# Nuget packages function stub.
################################################################################
function(use_package TARGET PACKAGE VERSION)
    message(WARNING "No implementation of use_package. Create yours. "
                    "Package \"${PACKAGE}\" with version \"${VERSION}\" "
                    "for target \"${TARGET}\" is ignored!")
endfunction()

################################################################################
# Additional Global Settings(add specific info there)
################################################################################
include(CMake/GlobalSettingsInclude.cmake OPTIONAL)

################################################################################
# Use solution folders feature
################################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(ClothTOP C CXX)

################################################################################
# Source groups
################################################################################

set(NVFLEXDIR  $ENV{NVFLEXDIR})
STRING(REGEX REPLACE "\\\\" "/" NVFLEXDIR ${NVFLEXDIR}) 

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${NVFLEXDIR}/include/)
include_directories(${NVFLEXDIR}/core/)
include_directories(${NVFLEXDIR})

set(APP_HEADERS
    "src/CPlusPlus_Common.h"
    "src/ClothTOP.h"
    "src/FlexSystem.h"
)
source_group("Headers" FILES ${APP_HEADERS})

set(APP_SOURCES 
    "src/ClothTOP.cpp"
    "src/FlexSystem.cpp"
    "src/kernel.cu"
)
source_group("Source" FILES ${APP_SOURCES} )

set(FLEX_CORE
    "${NVFLEXDIR}/core/aabbtree.cpp"
    "${NVFLEXDIR}/core/core.cpp"
    "${NVFLEXDIR}/core/extrude.cpp"
    "${NVFLEXDIR}/core/maths.cpp"
    "${NVFLEXDIR}/core/mesh.cpp"
    "${NVFLEXDIR}/core/perlin.cpp"
    "${NVFLEXDIR}/core/pfm.cpp"
    "${NVFLEXDIR}/core/platform.cpp"
    "${NVFLEXDIR}/core/png.cpp"
    "${NVFLEXDIR}/core/sdf.cpp"
    "${NVFLEXDIR}/core/tga.cpp"
    "${NVFLEXDIR}/core/voxelize.cpp"
    "${NVFLEXDIR}/core/aabbtree.h"
    "${NVFLEXDIR}/core/cloth.h"
    "${NVFLEXDIR}/core/convex.h"
    "${NVFLEXDIR}/core/core.h"
    "${NVFLEXDIR}/core/extrude.h"
    "${NVFLEXDIR}/core/mat22.h"
    "${NVFLEXDIR}/core/mat33.h"
    "${NVFLEXDIR}/core/mat44.h"
    "${NVFLEXDIR}/core/maths.h"
    "${NVFLEXDIR}/core/matnn.h"
    "${NVFLEXDIR}/core/mesh.h"
    "${NVFLEXDIR}/core/pfm.h"
    "${NVFLEXDIR}/core/png.h"
    "${NVFLEXDIR}/core/perlin.h"
    "${NVFLEXDIR}/core/platform.h"
    "${NVFLEXDIR}/core/point3.h"
    "${NVFLEXDIR}/core/quat.h"
    "${NVFLEXDIR}/core/sdf.h"
    "${NVFLEXDIR}/core/tga.h"
    "${NVFLEXDIR}/core/types.h"
    "${NVFLEXDIR}/core/vec2.h"
    "${NVFLEXDIR}/core/vec3.h"
    "${NVFLEXDIR}/core/vec4.h"
    "${NVFLEXDIR}/core/voxelize.h"
)
source_group("core" FILES ${FLEX_CORE} )

set(GL 
    "src/GL/glew.c"
    "src/GL/glewinfo.c"
    "src/GL/glew.h"
    "src/GL/wglew.h"
    "src/GL_Extensions.h"
)
source_group("GL" FILES ${GL} )

################################################################################
# Target
################################################################################

find_package(CUDA)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=\"sm_75,compute_75\")
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${APP_HEADERS} ${APP_SOURCES} ${GL} ${FLEX_CORE})

# Compile .cu files
file(GLOB CUDA_FILES "src/" *.cu)
CUDA_COMPILE(CU_O SHARED ${CUDA_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${NVFLEXDIR} ${NVFLEXDIR}/include )
target_link_directories(${PROJECT_NAME} PUBLIC ${NVFLEXDIR}/lib/win64)

target_link_libraries(${PROJECT_NAME} ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} 
                        opengl32.lib 
                        NvFlexDeviceRelease_x64.lib 
                        NvFlexExtReleaseCUDA_x64.lib 
                        NvFlexReleaseCUDA_x64.lib)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)


# The following step will create a post-build event that copies the custom DLL to
# the project Plugins folder.
if (MSVC)
    add_custom_command(TARGET ${PROJECT_NAME}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      "$<TARGET_FILE:ClothTOP>"
                      ${PROJECT_SOURCE_DIR}/Plugins)

    # ## Move Flex DLLs
    # file(GLOB FLEX_DLLS ${NVFLEXDIR}/bin/win64/NvFlexDeviceRelease_x64.dll
    #                     ${NVFLEXDIR}/bin/win64/NvFlexExtReleaseCUDA_x64.dll
    #                     ${NVFLEXDIR}/bin/win64/NvFlexReleaseCUDA_x64.dll)

    # add_custom_command(TARGET ${PROJECT_NAME}
    #                  POST_BUILD
    #                  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #                  ${FLEX_DLLS}
    #                  ${PROJECT_SOURCE_DIR}/Plugins)
endif (MSVC)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)

################################################################################
# Compile definitions
################################################################################
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Debug>:"
        "_DEBUG"
    ">"
    "$<$<CONFIG:Release>:"
        "NDEBUG"
    ">"
    "WIN32;"
    "_WINDOWS;"
    "_USRDLL"
)

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /Od;
            /RTC1;
            /MTd
        >
        $<$<CONFIG:Release>:
            /MT
        >
        /W3;
        /Zi;
        ${DEFAULT_CXX_EXCEPTION_HANDLING};
        /Y-
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:
            /OPT:REF;
            /OPT:ICF
        >
        /DEBUG;
        /SUBSYSTEM:WINDOWS;
        /INCREMENTAL:NO
    )
endif()

################################################################################
# Debugging
################################################################################
set_target_properties(${PROJECT_NAME} PROPERTIES
                      VS_DEBUGGER_COMMAND "C:\\Program Files\\Derivative\\TouchDesigner.2020.28110\\bin\\TouchDesigner.exe"
                      VS_DEBUGGER_COMMAND_ARGUMENTS "..\\$(ProjectName).toe")
    