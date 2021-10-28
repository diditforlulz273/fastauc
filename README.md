# fastauc

Two fast AUC calculation implementations for python:

python-based is approximately 5X faster than the default sklearn.metrics.roc_auc_csore()

C++/ctypes based is approximately 22X faster than the default sklearn.metrics.roc_auc_score()

![speedup](speedup.png)

NB: run cd fastauc && ./compile.sh to create a binary before you use a C++ version.

See demo.py for speed benchmark and usage examples.