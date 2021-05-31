#!/bin/bash

# install dependencies
yum update -y
yum install -y python3
yum install -y wget
alias python="python3"
export PYTHONPATH="$PYTHONPATH:/qphar/CDPKit/Python"
pip3 install -r requirements.txt

# download and install cdpkit-installer
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10y8d9fhMyNvy3-i7ncEt19-bJjvSVurX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10y8d9fhMyNvy3-i7ncEt19-bJjvSVurX" -O cdpkit_installer.sh && rm -rf /tmp/cookies.txt
yes | sh cdpkit_installer.sh

# test installation
python test_installation.py
