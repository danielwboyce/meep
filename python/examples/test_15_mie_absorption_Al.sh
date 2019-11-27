#!/bin/sh

python3 test_mie_absorption_Al.py -res 15 -courant 0.3 -until1 20 -until2 200|tee 'res_15_courant_0.3_until1_20_until2_200.out.txt'
python3 test_mie_absorption_Al.py -res 15 -courant 0.3 -until1 30 -until2 300|tee 'res_15_courant_0.3_until1_30_until2_300.out.txt'
python3 test_mie_absorption_Al.py -res 15 -courant 0.3 -until1 40 -until2 400|tee 'res_15_courant_0.3_until1_40_until2_400.out.txt'
python3 test_mie_absorption_Al.py -res 15 -courant 0.3 -until1 50 -until2 500|tee 'res_15_courant_0.3_until1_50_until2_500.out.txt'

