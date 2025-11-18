set shell := ["powershell"]

default:
    @just --list

flame:
    sudo run cargo flamegraph -o flamegraph.html --image-width 1800 --open --flamechart
