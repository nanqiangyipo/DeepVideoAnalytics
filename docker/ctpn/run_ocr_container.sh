#!/usr/bin/env bash
nvidia-docker run -p 8888:8888 --name ctpn -it akshayubhat/dva_ctpn /bin/bash -c "git pull && jupyter notebook --no-browser --allow-root"