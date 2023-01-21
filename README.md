# fastauc

Three fast AUC calculation implementations for python:

python-based is approximately 8X faster than the default sklearn.metrics.roc_auc_score()

Python numba based is approximately 26X faster than the default sklearn.metrics.roc_auc_score()

C++/ctypes based is approximately 37X faster than the default sklearn.metrics.roc_auc_score()

10 000 AUC calculations total time:

![times](times.png) 

NB: run cd fastauc && ./compile.sh to create a binary before you use a C++ version.

See demo.py for speed benchmark and usage examples.