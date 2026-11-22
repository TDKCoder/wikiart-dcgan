# Source before running any script: `source env.sh`
# Ubuntu 20.04's system libstdc++ is too old for conda's numpy/torch; use conda's copy.
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
