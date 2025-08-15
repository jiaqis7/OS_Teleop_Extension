#!/bin/bash
set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})

echo "Clearing Caches Script"
echo "Note: This script will delete folders on you system and is not reversible."

CLEAR_PATH=~/.cache/ov/shaders
echo -e "\nClearing shader cache... ${CLEAR_PATH}"
read -p "Are you sure (Y/[N])? " -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf ${CLEAR_PATH}
    echo -e "\nClearing shader cache DONE."
fi

CLEAR_PATH=~/.cache/ov/texturecache
echo -e "\nClearing texturecache... ${CLEAR_PATH}"
read -p "Are you sure (Y/[N])? " -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf ${CLEAR_PATH}
    echo -e "\nClearing texturecache DONE."
fi

CLEAR_PATH=~/.cache/ov/Kit/106.5
echo -e "\nClearing Kit cache... ${CLEAR_PATH}"
read -p "Are you sure (Y/[N])? " -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf ${CLEAR_PATH}
    echo -e "\nClearing Kit cache DONE."
fi

echo
read -n 1 -s -r -p "Press any key to continue"
