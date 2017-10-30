#!/bin/bash
for value in {2..8}
do
	python cw1_2b.py $1 -hl $value
done
