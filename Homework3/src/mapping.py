#!/usr/bin/env python
# coding: utf-8
import sys
input_fname = sys.argv[1]
output_fname = sys.argv[2]

convert_mapping = {}
with open( input_fname, encoding='big5hkscs') as f:
    for line in f:
        value = line.split(" ")[0]
        list_zhuyin = [ w[0] for w in line.split(" ")[1].replace("\n", "").split("/")]
        list_zhuyin = list(set(list_zhuyin))
        convert_mapping[value] = [value]
        for z in list_zhuyin:
            try:
                convert_mapping[z].append(value)
            except KeyError:
                convert_mapping[z] = [value]

with open( output_fname, "w", encoding='big5hkscs' ) as f:
    for i in convert_mapping:
        f.write( "{}\t{}\n".format( i, " ".join( convert_mapping[i] )))
