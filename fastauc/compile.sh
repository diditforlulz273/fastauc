#!/bin/sh
g++ -shared -O3 -fopenmp -D_GLIBCXX_PARALLEL cpp_auc.cpp -o cpp_auc.so
