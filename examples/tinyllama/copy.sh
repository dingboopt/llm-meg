#!/bin/bash

# 假设我们有一个数组
#array=(element1 element2 element3)

# 使用 seq 生成索引范围
for i in `seq 0 $2`; do
    cp $1.idx $1_$i.idx
	cp $1.bin $1_$i.bin
	echo "processed $1_$i"
done
