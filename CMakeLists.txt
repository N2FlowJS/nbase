cmake_minimum_required(VERSION 3.16)
project(libnbase VERSION 1.17.0 LANGUAGES C)

include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NBASE_BUILD_TESTS "Build libnbase test suite" ON)
option(NBASE_BUILD_BENCH "Build libnbase benchmark binary" ON)
option(NBASE_BUILD_APPS "Build nbased server and nbase-cli tools" ON)
option(NBASE_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)
option(NBASE_ENABLE_TSAN "Enable ThreadSanitizer (mutually exclusive with ASan)" OFF)

if(NBASE_ENABLE_SANITIZERS AND NBASE_ENABLE_TSAN)
  message(FATAL_ERROR "NBASE_ENABLE_SANITIZERS and NBASE_ENABLE_TSAN are mutually exclusive")
endif()

find_package(Threads REQUIRED)

set(NBASE_COMPILE_OPTIONS "")
set(NBASE_LINK_OPTIONS "")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  list(APPEND NBASE_COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)
endif()

if(NBASE_ENABLE_SANITIZERS)
  list(APPEND NBASE_COMPILE_OPTIONS -fsanitize=address,undefined -fno-omit-frame-pointer)
  list(APPEND NBASE_LINK_OPTIONS -fsanitize=address,undefined -fno-omit-frame-pointer)
elseif(NBASE_ENABLE_TSAN)
  list(APPEND NBASE_COMPILE_OPTIONS -fsanitize=thread -fno-omit-frame-pointer)
  list(APPEND NBASE_LINK_OPTIONS -fsanitize=thread -fno-omit-frame-pointer)
endif()

add_library(nbase_obj OBJECT
  src/crc32.c
  src/distance.c
  src/store.c
  src/hnsw.c
  src/meta.c
  src/kmeans.c
  src/pq.c
  src/sq8.c
  src/bq.c
  src/bm25.c
  src/lsh.c
  src/wal.c
  src/ivfpq.c
  src/rerank.c
  src/json.c
  src/http_server.c
  src/nbase.c
  src/partitioned.c
  src/clustered.c
)

set_target_properties(nbase_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(nbase_obj PRIVATE _POSIX_C_SOURCE=200809L)
target_compile_options(nbase_obj PRIVATE ${NBASE_COMPILE_OPTIONS})
target_include_directories(nbase_obj
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(nbase_static STATIC $<TARGET_OBJECTS:nbase_obj>)
set_target_properties(nbase_static PROPERTIES OUTPUT_NAME nbase)
target_include_directories(nbase_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nbase_static PRIVATE m Threads::Threads)
target_link_options(nbase_static INTERFACE ${NBASE_LINK_OPTIONS})

add_library(nbase_shared SHARED $<TARGET_OBJECTS:nbase_obj>)
set_target_properties(nbase_shared PROPERTIES
  OUTPUT_NAME nbase
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)
target_include_directories(nbase_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nbase_shared PRIVATE m Threads::Threads)
target_link_options(nbase_shared PRIVATE ${NBASE_LINK_OPTIONS})

if(NBASE_BUILD_TESTS)
  enable_testing()
  add_executable(test_nbase
    tests/test_main.c
    tests/test_distance.c
    tests/test_store.c
    tests/test_hnsw.c
    tests/test_db.c
    tests/test_meta.c
    tests/test_pq.c
    tests/test_partitioned.c
    tests/test_clustered.c
    tests/test_wal.c
    tests/test_ivfpq.c
    tests/test_rerank.c
    tests/test_sq8.c
    tests/test_bq.c
    tests/test_bm25.c
    tests/test_hybrid.c
    tests/test_lsh.c
    tests/test_json.c
    tests/test_server.c
    tests/test_concurrency.c
  )
  target_compile_definitions(test_nbase PRIVATE _POSIX_C_SOURCE=200809L)
  target_compile_options(test_nbase PRIVATE ${NBASE_COMPILE_OPTIONS})
  target_include_directories(test_nbase
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests
  )
  target_link_libraries(test_nbase PRIVATE nbase_static m Threads::Threads)
  target_link_options(test_nbase PRIVATE ${NBASE_LINK_OPTIONS})
  add_test(NAME nbase_suite COMMAND test_nbase)
endif()

if(NBASE_BUILD_BENCH)
  add_executable(bench_nbase
    bench/bench_nbase.c
  )
  target_compile_definitions(bench_nbase PRIVATE _POSIX_C_SOURCE=200809L)
  target_compile_options(bench_nbase PRIVATE ${NBASE_COMPILE_OPTIONS})
  target_include_directories(bench_nbase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(bench_nbase PRIVATE nbase_static m Threads::Threads)
  target_link_options(bench_nbase PRIVATE ${NBASE_LINK_OPTIONS})
endif()

if(NBASE_BUILD_APPS)
  add_executable(nbased apps/nbased.c)
  target_compile_definitions(nbased PRIVATE _POSIX_C_SOURCE=200809L)
  target_compile_options(nbased PRIVATE ${NBASE_COMPILE_OPTIONS})
  target_include_directories(nbased PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(nbased PRIVATE nbase_static m Threads::Threads)
  target_link_options(nbased PRIVATE ${NBASE_LINK_OPTIONS})

  add_executable(nbase_cli apps/nbase_cli.c)
  set_target_properties(nbase_cli PROPERTIES OUTPUT_NAME nbase-cli)
  target_compile_definitions(nbase_cli PRIVATE _POSIX_C_SOURCE=200809L)
  target_compile_options(nbase_cli PRIVATE ${NBASE_COMPILE_OPTIONS})
  target_include_directories(nbase_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(nbase_cli PRIVATE nbase_static m Threads::Threads)
  target_link_options(nbase_cli PRIVATE ${NBASE_LINK_OPTIONS})

  install(TARGETS nbased nbase_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(NBASE_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
else()
  set(NBASE_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  set(NBASE_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  set(NBASE_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libnbase.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libnbase.pc
  @ONLY
)

install(TARGETS nbase_static nbase_shared
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES include/nbase.h include/nbase.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libnbase.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(FILES README.md LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})

set(CPACK_PACKAGE_NAME "nbase")
set(CPACK_PACKAGE_VENDOR "n2flowjs")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pure-C11/C++17 vector, metadata, graph and full-text database engine")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "nbase-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_GENERATOR "TGZ")
include(CPack)
