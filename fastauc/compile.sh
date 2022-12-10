#!/bin/sh
g++ -shared -O3 -march=native -fopenmp -D_GLIBCXX_PARALLEL cpp_auc.cpp -o cpp_auc.so
