#!/bin/sh

# get tunneling info
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$HOSTNAME # Check environment variables with `printenv` to look for cluster name

# print tunneling instructions jupyter-log
echo -e "
In a separate local terminal, run:
ssh -N -L ${port}:${node}:${port} ${user}@perlmutter-p1.nersc.gov
"

jupyter-notebook --no-browser --port=${port} --ip=${node}
