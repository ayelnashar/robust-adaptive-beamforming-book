"""Compare a regenerated results_summary.json with the values printed in Chapter 7 (Tables 7.3-7.6 and the
numbers quoted in Sections 7.7.2-7.7.12). Timing (Table 7.1) depends on the machine and is not compared.

    python -m ch7rdfisac.verify [figures]       # folder that holds results_summary.json
"""
import json
import sys
from pathlib import Path

EXPECTED = json.loads(r"""{
"table_7_3": {
"A|rdf_as_isac|0": 10.87,
"A|projection|0": 10.49,
"A|socp|0": 9.88,
"A|socp_radius|0": 10.48,
"A|socp_online|0": 7.78,
"A|socp_radius_online|0": 9.25,
"A|fixed|0": 10.36,
"A|true_cov|0": 11.21,
"A|bound|0": 20.11,
"A|rdf_as_isac|1": 10.86,
"A|projection|1": 10.49,
"A|socp|1": 10.17,
"A|socp_radius|1": 10.59,
"A|socp_online|1": 8.12,
"A|socp_radius_online|1": 9.35,
"A|fixed|1": 10.36,
"A|true_cov|1": 11.22,
"A|bound|1": 21.0,
"A|rdf_as_isac|2": 10.33,
"A|projection|2": 10.12,
"A|socp|2": 10.39,
"A|socp_radius|2": 10.29,
"A|socp_online|2": 10.11,
"A|socp_radius_online|2": 10.08,
"A|fixed|2": 10.1,
"A|true_cov|2": 10.17,
"A|bound|2": 29.0,
"A|rdf_as_isac|3": 9.0,
"A|projection|3": 8.96,
"A|socp|3": 9.1,
"A|socp_radius|3": 9.08,
"A|socp_online|3": 8.71,
"A|socp_radius_online|3": 8.86,
"A|fixed|3": 8.94,
"A|true_cov|3": 9.05,
"A|bound|3": 29.13,
"B|rdf_as_isac|0": 9.75,
"B|projection|0": 9.16,
"B|socp|0": 0.77,
"B|socp_radius|0": 6.67,
"B|socp_online|0": -1.08,
"B|socp_radius_online|0": 5.81,
"B|fixed|0": 9.03,
"B|true_cov|0": 9.98,
"B|bound|0": 20.18,
"B|rdf_as_isac|1": 9.61,
"B|projection|1": 9.25,
"B|socp|1": 1.86,
"B|socp_radius|1": 6.88,
"B|socp_online|1": 0.3,
"B|socp_radius_online|1": 5.77,
"B|fixed|1": 9.13,
"B|true_cov|1": 10.28,
"B|bound|1": 21.04,
"B|rdf_as_isac|2": 11.16,
"B|projection|2": 11.03,
"B|socp|2": 13.29,
"B|socp_radius|2": 12.39,
"B|socp_online|2": 12.8,
"B|socp_radius_online|2": 12.13,
"B|fixed|2": 11.01,
"B|true_cov|2": 10.97,
"B|bound|2": 29.11,
"B|rdf_as_isac|3": 9.83,
"B|projection|3": 9.87,
"B|socp|3": 12.11,
"B|socp_radius|3": 11.17,
"B|socp_online|3": 11.6,
"B|socp_radius_online|3": 10.92,
"B|fixed|3": 9.85,
"B|true_cov|3": 9.85,
"B|bound|3": 29.1,
"C|rdf_as_isac|0": 11.37,
"C|projection|0": 10.88,
"C|socp|0": 10.3,
"C|socp_radius|0": 10.93,
"C|socp_online|0": 8.1,
"C|socp_radius_online|0": 9.61,
"C|fixed|0": 10.74,
"C|true_cov|0": 11.73,
"C|bound|0": 20.19,
"C|rdf_as_isac|1": 11.33,
"C|projection|1": 10.88,
"C|socp|1": 10.6,
"C|socp_radius|1": 11.04,
"C|socp_online|1": 8.43,
"C|socp_radius_online|1": 9.71,
"C|fixed|1": 10.74,
"C|true_cov|1": 11.74,
"C|bound|1": 21.18,
"C|rdf_as_isac|2": 9.48,
"C|projection|2": 9.37,
"C|socp|2": 9.25,
"C|socp_radius|2": 9.39,
"C|socp_online|2": 8.93,
"C|socp_radius_online|2": 9.16,
"C|fixed|2": 9.35,
"C|true_cov|2": 9.2,
"C|bound|2": 28.82,
"C|rdf_as_isac|3": 7.9,
"C|projection|3": 8.21,
"C|socp|3": 8.39,
"C|socp_radius|3": 8.21,
"C|socp_online|3": 8.07,
"C|socp_radius_online|3": 8.03,
"C|fixed|3": 8.19,
"C|true_cov|3": 8.36,
"C|bound|3": 28.84,
"D|rdf_as_isac|0": 10.36,
"D|projection|0": 9.84,
"D|socp|0": 3.52,
"D|socp_radius|0": 7.95,
"D|socp_online|0": 1.47,
"D|socp_radius_online|0": 7.01,
"D|fixed|0": 9.7,
"D|true_cov|0": 10.72,
"D|bound|0": 20.24,
"D|rdf_as_isac|1": 10.29,
"D|projection|1": 9.95,
"D|socp|1": 3.51,
"D|socp_radius|1": 8.03,
"D|socp_online|1": 1.72,
"D|socp_radius_online|1": 6.87,
"D|fixed|1": 9.81,
"D|true_cov|1": 10.98,
"D|bound|1": 21.17,
"D|rdf_as_isac|2": 10.45,
"D|projection|2": 10.28,
"D|socp|2": 11.46,
"D|socp_radius|2": 11.34,
"D|socp_online|2": 11.13,
"D|socp_radius_online|2": 11.09,
"D|fixed|2": 10.26,
"D|true_cov|2": 10.08,
"D|bound|2": 28.91,
"D|rdf_as_isac|3": 8.87,
"D|projection|3": 9.11,
"D|socp|3": 10.0,
"D|socp_radius|3": 9.86,
"D|socp_online|3": 9.95,
"D|socp_radius_online|3": 9.69,
"D|fixed|3": 9.08,
"D|true_cov|3": 9.15,
"D|bound|3": 28.94,
"F|rdf_as_isac|0": 9.37,
"F|projection|0": -17.24,
"F|socp|0": 3.85,
"F|socp_radius|0": 3.44,
"F|socp_online|0": 2.24,
"F|socp_radius_online|0": 2.77,
"F|fixed|0": -17.01,
"F|true_cov|0": 9.85,
"F|bound|0": 19.88,
"F|rdf_as_isac|1": 9.04,
"F|projection|1": -16.28,
"F|socp|1": 4.43,
"F|socp_radius|1": 3.61,
"F|socp_online|1": 3.09,
"F|socp_radius_online|1": 2.95,
"F|fixed|1": -16.89,
"F|true_cov|1": 9.46,
"F|bound|1": 20.06,
"F|rdf_as_isac|2": 9.42,
"F|projection|2": -9.36,
"F|socp|2": 10.61,
"F|socp_radius|2": 8.8,
"F|socp_online|2": 10.11,
"F|socp_radius_online|2": 8.45,
"F|fixed|2": -8.49,
"F|true_cov|2": 9.49,
"F|bound|2": 27.47,
"F|rdf_as_isac|3": 7.26,
"F|projection|3": -9.61,
"F|socp|3": 8.86,
"F|socp_radius|3": 6.73,
"F|socp_online|3": 8.31,
"F|socp_radius_online|3": 6.37,
"F|fixed|3": -9.67,
"F|true_cov|3": 7.37,
"F|bound|3": 19.15,
"G|rdf_as_isac|0": 7.19,
"G|projection|0": -18.59,
"G|socp|0": -3.87,
"G|socp_radius|0": -9.45,
"G|socp_online|0": -4.8,
"G|socp_radius_online|0": -4.34,
"G|fixed|0": -18.59,
"G|true_cov|0": 10.52,
"G|bound|0": 18.95,
"G|rdf_as_isac|1": 6.08,
"G|projection|1": -17.63,
"G|socp|1": -3.4,
"G|socp_radius|1": -9.02,
"G|socp_online|1": -3.63,
"G|socp_radius_online|1": -4.07,
"G|fixed|1": -18.47,
"G|true_cov|1": 9.38,
"G|bound|1": 21.15,
"G|rdf_as_isac|2": 7.65,
"G|projection|2": -10.68,
"G|socp|2": 5.17,
"G|socp_radius|2": -1.87,
"G|socp_online|2": 5.44,
"G|socp_radius_online|2": 2.78,
"G|fixed|2": -10.06,
"G|true_cov|2": 8.38,
"G|bound|2": 25.34,
"G|rdf_as_isac|3": 5.53,
"G|projection|3": -10.94,
"G|socp|3": 2.12,
"G|socp_radius|3": -3.16,
"G|socp_online|3": 3.16,
"G|socp_radius_online|3": 1.14,
"G|fixed|3": -11.24,
"G|true_cov|3": 6.31,
"G|bound|3": 18.03
},
"table_7_4": {
"A|projection|user": [
0.35,
0.27,
0.42
],
"A|socp_radius|user": [
0.36,
0.16,
0.56
],
"A|socp_radius_online|user": [
1.29,
0.84,
1.73
],
"A|true_cov|user": [
-0.35,
-0.45,
-0.26
],
"A|projection|target": [
0.13,
0.06,
0.2
],
"A|socp_radius|target": [
-0.01,
-0.13,
0.12
],
"A|socp_radius_online|target": [
0.13,
-0.04,
0.31
],
"A|true_cov|target": [
0.03,
-0.05,
0.11
],
"B|projection|user": [
0.5,
-0.01,
1.01
],
"B|socp_radius|user": [
3.68,
1.55,
5.8
],
"B|socp_radius_online|user": [
4.49,
2.63,
6.35
],
"B|true_cov|user": [
-0.18,
-0.82,
0.45
],
"B|projection|target": [
0.05,
-0.29,
0.39
],
"B|socp_radius|target": [
-1.56,
-2.52,
-0.6
],
"B|socp_radius_online|target": [
-1.27,
-2.25,
-0.28
],
"B|true_cov|target": [
-0.03,
-0.43,
0.37
],
"C|projection|user": [
0.43,
0.32,
0.54
],
"C|socp_radius|user": [
0.41,
0.18,
0.64
],
"C|socp_radius_online|user": [
1.41,
0.91,
1.92
],
"C|true_cov|user": [
-0.37,
-0.5,
-0.23
],
"C|projection|target": [
0.06,
-0.16,
0.29
],
"C|socp_radius|target": [
0.08,
-0.23,
0.38
],
"C|socp_radius_online|target": [
0.18,
-0.14,
0.5
],
"C|true_cov|target": [
0.01,
-0.26,
0.28
],
"D|projection|user": [
0.47,
0.06,
0.88
],
"D|socp_radius|user": [
2.69,
1.2,
4.19
],
"D|socp_radius_online|user": [
3.72,
2.36,
5.08
],
"D|true_cov|user": [
-0.29,
-0.8,
0.23
],
"D|projection|target": [
0.07,
-0.27,
0.41
],
"D|socp_radius|target": [
-1.07,
-1.94,
-0.2
],
"D|socp_radius_online|target": [
-0.84,
-1.72,
0.05
],
"D|true_cov|target": [
0.01,
-0.37,
0.39
],
"F|projection|user": [
25.68,
25.0,
26.36
],
"F|socp_radius|user": [
6.19,
5.54,
6.85
],
"F|socp_radius_online|user": [
6.79,
6.26,
7.31
],
"F|true_cov|user": [
-0.5,
-0.63,
-0.37
],
"F|projection|target": [
17.91,
16.92,
18.9
],
"F|socp_radius|target": [
0.49,
-0.09,
1.07
],
"F|socp_radius_online|target": [
0.84,
0.27,
1.4
],
"F|true_cov|target": [
-0.1,
-0.21,
0.01
],
"G|projection|user": [
23.86,
23.12,
24.6
],
"G|socp_radius|user": [
15.39,
14.95,
15.83
],
"G|socp_radius_online|user": [
10.4,
9.76,
11.04
],
"G|true_cov|user": [
-3.51,
-3.76,
-3.26
],
"G|projection|target": [
17.49,
15.93,
19.04
],
"G|socp_radius|target": [
9.12,
8.32,
9.93
],
"G|socp_radius_online|target": [
4.5,
3.92,
5.08
],
"G|true_cov|target": [
-0.88,
-0.97,
-0.79
]
},
"table_7_5": {
"array|(16, 0)|rdf_as_isac": [
7.7,
8.6
],
"array|(16, 0)|projection": [
3.7,
7.2
],
"array|(16, 0)|socp": [
5.6,
9.5
],
"array|(16, 0)|socp_radius": [
5.1,
8.4
],
"array|(16, 0)|fixed": [
3.6,
7.1
],
"array|(16, 0)|true_cov": [
9.4,
9.1
],
"array|(16, 0)|bound": [
16.1,
22.8
],
"array|(16, 3)|rdf_as_isac": [
-19.2,
-11.0
],
"array|(16, 3)|projection": [
-20.9,
-13.3
],
"array|(16, 3)|socp": [
-7.2,
2.0
],
"array|(16, 3)|socp_radius": [
-19.2,
-11.1
],
"array|(16, 3)|fixed": [
-20.9,
-13.3
],
"array|(16, 3)|true_cov": [
-19.2,
-11.1
],
"array|(16, 3)|bound": [
5.7,
16.0
],
"array|(32, 0)|rdf_as_isac": [
11.0,
8.7
],
"array|(32, 0)|projection": [
8.1,
8.5
],
"array|(32, 0)|socp": [
4.5,
9.9
],
"array|(32, 0)|socp_radius": [
8.6,
9.8
],
"array|(32, 0)|fixed": [
8.1,
8.5
],
"array|(32, 0)|true_cov": [
10.9,
9.0
],
"array|(32, 0)|bound": [
21.4,
28.4
],
"array|(32, 3)|rdf_as_isac": [
2.9,
7.0
],
"array|(32, 3)|projection": [
-15.4,
-8.0
],
"array|(32, 3)|socp": [
3.4,
10.3
],
"array|(32, 3)|socp_radius": [
-0.5,
5.7
],
"array|(32, 3)|fixed": [
-15.4,
-8.0
],
"array|(32, 3)|true_cov": [
4.1,
7.1
],
"array|(32, 3)|bound": [
17.3,
25.4
],
"array|(64, 0)|rdf_as_isac": [
11.9,
8.7
],
"array|(64, 0)|projection": [
9.4,
8.4
],
"array|(64, 0)|socp": [
-1.4,
11.0
],
"array|(64, 0)|socp_radius": [
9.2,
10.2
],
"array|(64, 0)|fixed": [
9.3,
8.4
],
"array|(64, 0)|true_cov": [
12.1,
8.8
],
"array|(64, 0)|bound": [
24.9,
31.7
],
"array|(64, 3)|rdf_as_isac": [
9.1,
9.4
],
"array|(64, 3)|projection": [
-15.9,
-8.5
],
"array|(64, 3)|socp": [
0.4,
11.5
],
"array|(64, 3)|socp_radius": [
3.9,
8.8
],
"array|(64, 3)|fixed": [
-16.0,
-8.7
],
"array|(64, 3)|true_cov": [
9.8,
10.0
],
"array|(64, 3)|bound": [
21.4,
29.4
],
"array|(128, 0)|rdf_as_isac": [
11.8,
9.1
],
"array|(128, 0)|projection": [
10.0,
8.7
],
"array|(128, 0)|socp": [
-6.7,
7.9
],
"array|(128, 0)|socp_radius": [
6.8,
11.2
],
"array|(128, 0)|fixed": [
9.9,
8.7
],
"array|(128, 0)|true_cov": [
12.0,
8.5
],
"array|(128, 0)|bound": [
28.3,
35.4
],
"array|(128, 3)|rdf_as_isac": [
11.6,
8.7
],
"array|(128, 3)|projection": [
-7.8,
-0.7
],
"array|(128, 3)|socp": [
-7.8,
11.4
],
"array|(128, 3)|socp_radius": [
7.8,
10.7
],
"array|(128, 3)|fixed": [
-8.3,
-1.2
],
"array|(128, 3)|true_cov": [
11.7,
9.4
],
"array|(128, 3)|bound": [
27.6,
34.8
],
"load|(2, 0)|rdf_as_isac": [
11.3,
8.7
],
"load|(2, 0)|projection": [
8.9,
9.2
],
"load|(2, 0)|socp": [
-2.1,
12.4
],
"load|(2, 0)|socp_radius": [
8.6,
10.7
],
"load|(2, 0)|fixed": [
8.8,
9.2
],
"load|(2, 0)|true_cov": [
11.3,
9.6
],
"load|(2, 0)|bound": [
24.1,
31.6
],
"load|(2, 3)|rdf_as_isac": [
2.9,
8.4
],
"load|(2, 3)|projection": [
-17.1,
-9.0
],
"load|(2, 3)|socp": [
-3.5,
12.1
],
"load|(2, 3)|socp_radius": [
-0.3,
8.2
],
"load|(2, 3)|fixed": [
-17.3,
-9.2
],
"load|(2, 3)|true_cov": [
6.4,
9.7
],
"load|(2, 3)|bound": [
19.4,
29.9
],
"load|(3, 0)|rdf_as_isac": [
9.1,
8.2
],
"load|(3, 0)|projection": [
8.7,
8.1
],
"load|(3, 0)|socp": [
1.5,
11.1
],
"load|(3, 0)|socp_radius": [
7.9,
9.1
],
"load|(3, 0)|fixed": [
8.6,
8.2
],
"load|(3, 0)|true_cov": [
9.3,
8.2
],
"load|(3, 0)|bound": [
23.9,
31.8
],
"load|(3, 3)|rdf_as_isac": [
3.5,
4.1
],
"load|(3, 3)|projection": [
-17.9,
-9.8
],
"load|(3, 3)|socp": [
2.6,
9.5
],
"load|(3, 3)|socp_radius": [
-2.5,
2.4
],
"load|(3, 3)|fixed": [
-17.8,
-10.0
],
"load|(3, 3)|true_cov": [
3.8,
4.0
],
"load|(3, 3)|bound": [
22.1,
28.9
],
"load|(4, 0)|rdf_as_isac": [
8.2,
6.3
],
"load|(4, 0)|projection": [
7.5,
6.0
],
"load|(4, 0)|socp": [
1.7,
8.8
],
"load|(4, 0)|socp_radius": [
6.7,
6.6
],
"load|(4, 0)|fixed": [
7.3,
6.0
],
"load|(4, 0)|true_cov": [
8.5,
6.1
],
"load|(4, 0)|bound": [
23.7,
30.9
],
"load|(4, 3)|rdf_as_isac": [
-4.9,
-1.0
],
"load|(4, 3)|projection": [
-17.9,
-11.1
],
"load|(4, 3)|socp": [
0.8,
7.1
],
"load|(4, 3)|socp_radius": [
-7.4,
-2.3
],
"load|(4, 3)|fixed": [
-17.9,
-11.2
],
"load|(4, 3)|true_cov": [
-0.5,
0.9
],
"load|(4, 3)|bound": [
20.6,
28.5
]
},
"table_7_6": {
"D|default": [
10.8,
9.47
],
"F|default": [
9.09,
7.83
],
"G|default": [
5.75,
6.35
],
"D|win_az=6": [
10.8,
9.47
],
"F|win_az=6": [
9.09,
7.83
],
"G|win_az=6": [
5.75,
6.35
],
"D|win_az=12": [
10.8,
9.47
],
"F|win_az=12": [
9.1,
7.83
],
"G|win_az=12": [
5.75,
6.35
],
"D|eta=0.94": [
10.8,
9.48
],
"F|eta=0.94": [
9.08,
7.83
],
"G|eta=0.94": [
5.69,
6.35
],
"D|eta=0.98": [
10.82,
9.45
],
"F|eta=0.98": [
9.03,
7.81
],
"G|eta=0.98": [
5.3,
5.81
],
"D|T=5": [
10.79,
9.46
],
"F|T=5": [
8.98,
7.82
],
"G|T=5": [
5.86,
6.3
],
"D|T=20": [
10.81,
9.47
],
"F|T=20": [
9.25,
7.86
],
"G|T=20": [
2.79,
5.08
],
"D|tol=3": [
10.8,
9.47
],
"F|tol=3": [
9.09,
7.83
],
"G|tol=3": [
5.69,
6.36
],
"D|tol=10": [
10.8,
9.47
],
"F|tol=10": [
9.09,
7.83
],
"G|tol=10": [
5.84,
6.35
],
"D|beta_v=0.1": [
10.81,
9.47
],
"F|beta_v=0.1": [
9.22,
7.86
],
"G|beta_v=0.1": [
2.7,
5.18
],
"D|beta_v=0.5": [
10.79,
9.46
],
"F|beta_v=0.5": [
8.98,
7.81
],
"G|beta_v=0.5": [
6.01,
6.39
],
"D|alpha_p=0.3": [
10.81,
9.47
],
"F|alpha_p=0.3": [
9.1,
7.83
],
"G|alpha_p=0.3": [
5.44,
6.24
],
"D|alpha_p=1": [
10.8,
9.47
],
"F|alpha_p=1": [
8.98,
7.81
],
"G|alpha_p=1": [
5.65,
6.3
],
"D|nu_mult=2": [
10.81,
9.36
],
"F|nu_mult=2": [
9.12,
7.69
],
"G|nu_mult=2": [
5.37,
6.56
],
"D|nu_mult=4": [
10.82,
9.54
],
"F|nu_mult=4": [
8.29,
7.65
],
"G|nu_mult=4": [
0.25,
2.54
]
}
}""")


def _tol(x, extra=0.0):
    s = repr(float(x)); d = len(s.split('.')[1]) if '.' in s and 'e' not in s else 0
    return 0.5*10**(-d) + 1e-9 + extra


def text_checks(N):
    """(label, regenerated value, chapter value, tolerance). Differences of two rounded means get 0.011."""
    a, l, sw, cr = N['array'], N['load'], N['sweep'], N['cross']
    u = lambda d, m: d[m]['user']; p1 = lambda d, m: d[m]['per'][0]
    pv = N['main_paired_avg']['heldout']; ck = N['checks']; det = N['detection_by_pfa']['1e-04']
    hm = N['main_means']['heldout']
    D = 0.011
    out = [
        ('7.7.2 exact projection, F (iterative)', N['exact_projection_gain']['F']['iterative'], 25.68, 0),
        ('7.7.2 exact projection, F (exact)', N['exact_projection_gain']['F']['exact'], 25.92, 0),
        ('7.7.2 exact projection, G (iterative)', N['exact_projection_gain']['G']['iterative'], 23.86, 0),
        ('7.7.2 exact projection, G (exact)', N['exact_projection_gain']['G']['exact'], 24.33, 0),
        ('7.7.2 SDP and SOCP agree within 0.006 dB', N['sdp_socp_maxdiff'] <= 0.006, True, None),
        ('7.7.2 naive LCMV, A-D scenes with a user below 0 dB', N['users_below_0dB_AD']['naive'], 114, None),
        ('7.7.2 number of A-D scenes', N['n_AD'], 120, None),
        ('7.7.3 base-radius SOCP, users below 0 dB', N['users_below_0dB_AD']['socp'], 29, None),
        ('7.7.3 matched-radius SOCP, users below 0 dB', N['users_below_0dB_AD']['socp_radius'], 10, None),
        ('7.7.3 RDF-AS-ISAC, projection, fixed beam: users below 0 dB',
         [N['users_below_0dB_AD'][k] for k in ('rdf_as_isac', 'projection', 'fixed')], [0, 0, 0], None),
        ('7.7.3 fixed beam vs RDF-AS-ISAC, held-out A-D user gain (min)', min(pv[s]['fixed']['user']['mean'] for s in 'ABCD'), 0.48, 0),
        ('7.7.3 fixed beam vs RDF-AS-ISAC, held-out A-D user gain (max)', max(pv[s]['fixed']['user']['mean'] for s in 'ABCD'), 0.64, 0),
        ('7.7.3 online base-radius SOCP, held-out G user gain', pv['G']['socp_online']['user']['mean'], 9.95, 0),
        ('7.7.3 VU-MBWC mean user SINR, held-out F', round(sum(hm['F']['vu_mbwc'][:2])/2, 2), 5.09, D),
        ('7.7.3 VU-MBWC mean user SINR, held-out G', round(sum(hm['G']['vu_mbwc'][:2])/2, 2), -1.90, D),
        ('7.7.4 benign, M = 16: gap to true covariance', u(a['(16, 0)'], 'true_cov') - u(a['(16, 0)'], 'rdf_as_isac'), 1.70, D),
        ('7.7.4 benign, M >= 32: largest gap to true covariance', max(u(a[f'({m}, 0)'], 'true_cov') - u(a[f'({m}, 0)'], 'rdf_as_isac') for m in (32, 64, 128)), 0.24, D),
        ('7.7.4 benign: smallest gain over projection', min(u(a[f'({m}, 0)'], 'rdf_as_isac') - u(a[f'({m}, 0)'], 'projection') for m in (16, 32, 64, 128)), 1.76, D),
        ('7.7.4 benign: largest gain over projection', max(u(a[f'({m}, 0)'], 'rdf_as_isac') - u(a[f'({m}, 0)'], 'projection') for m in (16, 32, 64, 128)), 3.92, D),
        ('7.7.4 strong, M = 16: true covariance', u(a['(16, 3)'], 'true_cov'), -19.2, 0.051),
    ]
    for m_, g in zip((32, 64, 128), (1.27, 0.72, 0.11)):
        out.append((f'7.7.4 strong, M = {m_}: gap to true covariance', u(a[f'({m_}, 3)'], 'true_cov') - u(a[f'({m_}, 3)'], 'rdf_as_isac'), g, D))
    for m_, g in zip((32, 64, 128), (3.30, 5.26, 3.79)):
        out.append((f'7.7.4 strong, M = {m_}: gain over matched-radius SOCP', u(a[f'({m_}, 3)'], 'rdf_as_isac') - u(a[f'({m_}, 3)'], 'socp_radius'), g, D))
    out += [
        ('7.7.4 strong, M = 32: base-radius SOCP above RDF-AS-ISAC', u(a['(32, 3)'], 'socp') - u(a['(32, 3)'], 'rdf_as_isac'), 0.54, D),
        ('7.7.4 benign, M = 64: base-radius SOCP', u(a['(64, 0)'], 'socp'), -1.37, 0),
        ('7.7.4 benign, M = 128: base-radius SOCP', u(a['(128, 0)'], 'socp'), -6.69, 0),
        ('7.7.4 load, benign: largest gap to true covariance', max(u(l[f'({k}, 0)'], 'true_cov') - u(l[f'({k}, 0)'], 'rdf_as_isac') for k in (2, 3, 4)), 0.30, D),
    ]
    for k, g in zip((2, 3, 4), (3.56, 0.32, 4.37)):
        out.append((f'7.7.4 load U = T = {k}, strong: gap to true covariance', u(l[f'({k}, 3)'], 'true_cov') - u(l[f'({k}, 3)'], 'rdf_as_isac'), g, D))
    out += [
        ('7.7.4 U = T = 4, strong: true covariance', u(l['(4, 3)'], 'true_cov'), -0.51, 0),
        ('7.7.4 U = T = 4, strong: RDF-AS-ISAC', u(l['(4, 3)'], 'rdf_as_isac'), -4.88, 0),
        ('7.7.4 U = T = 4, strong: base-radius SOCP', u(l['(4, 3)'], 'socp'), 0.85, 0),
    ]
    s_, i_, mm = sw['separation'], sw['inr'], sw['mismatch']
    gap = lambda d, k: p1(d[k], 'true_cov') - p1(d[k], 'rdf_as_isac')
    out += [
        ('7.7.5 separation >= 10 deg: largest gap to true covariance', max(gap(s_, k) for k in ('10', '14', '20')), 0.33, D),
        ('7.7.5 separation 7 deg: gap', gap(s_, '7'), 1.52, D), ('7.7.5 separation 5 deg: gap', gap(s_, '5'), 3.82, D),
        ('7.7.5 separation 3 deg: true covariance', p1(s_['3'], 'true_cov'), -15.1, 0.051),
        ('7.7.5 projection and fixed beam stay below -7.7 dB', max(max(p1(s_[k], 'projection'), p1(s_[k], 'fixed')) for k in s_) < -7.7, True, None),
        ('7.7.5 INR >= 15 dB: largest gap to true covariance', max(gap(i_, k) for k in ('15', '20', '25', '30')), 0.25, D),
        ('7.7.5 INR 5 dB: gap', gap(i_, '5'), 2.49, D), ('7.7.5 INR 10 dB: gap', gap(i_, '10'), 7.21, D),
        ('7.7.5 INR 10 dB: below matched-radius SOCP', p1(i_['10'], 'socp_radius') - p1(i_['10'], 'rdf_as_isac'), 2.53, D),
        ('7.7.5 INR 10 dB: below projection', p1(i_['10'], 'projection') - p1(i_['10'], 'rdf_as_isac'), 0.4, 0.06),
        ('7.7.5 INR 10 dB: below fixed beam', p1(i_['10'], 'fixed') - p1(i_['10'], 'rdf_as_isac'), 0.4, 0.06),
        ('7.7.5 mismatch: RDF-AS-ISAC lowest mean user SINR', min(u(mm[k], 'rdf_as_isac') for k in mm), 10.76, 0),
        ('7.7.5 mismatch: RDF-AS-ISAC highest mean user SINR', max(u(mm[k], 'rdf_as_isac') for k in mm), 10.89, 0),
        ('7.7.5 mismatch 0 deg: matched-radius SOCP', u(mm['0'], 'socp_radius'), 10.66, 0),
        ('7.7.5 mismatch 5 deg: matched-radius SOCP', u(mm['5'], 'socp_radius'), 5.94, 0),
        ('7.7.5 mismatch 0 deg: base-radius SOCP', u(mm['0'], 'socp'), 10.28, 0),
        ('7.7.5 mismatch 5 deg: base-radius SOCP', u(mm['5'], 'socp'), -5.48, 0),
        ('7.7.5 mismatch 5 deg: RDF-AS-ISAC target SNR', mm['5']['rdf_as_isac']['target'], 7.04, 0),
        ('7.7.5 mismatch 5 deg: projection target SNR', mm['5']['projection']['target'], 8.06, 0),
        ('7.7.6 crossing: lowest user 1 SINR', min(p1(cr[k], 'rdf_as_isac') for k in cr), 10.55, 0),
        ('7.7.6 crossing: highest user 1 SINR', max(p1(cr[k], 'rdf_as_isac') for k in cr), 11.68, 0),
        ('7.7.6 crossing: largest gap to true covariance', max(p1(cr[k], 'true_cov') - p1(cr[k], 'rdf_as_isac') for k in cr), 0.23, D),
        ('7.7.6 crossing: lowest target 1 SNR', min(cr[k]['rdf_as_isac']['per'][2] for k in cr), 9.94, 0),
        ('7.7.6 crossing: highest target 1 SNR', max(cr[k]['rdf_as_isac']['per'][2] for k in cr), 10.74, 0),
        ('7.7.9 scenes', ck['n_scenes'], 180, None), ('7.7.9 output weights', ck['n_weights'], 54000, None),
        ('7.7.9 smallest margin (1.0e-12)', round(ck['min_margin'], 13), 1.0e-12, 0.05e-12),
        ('7.7.9 largest certified gap (6.7e-8)', ck['max_cert'], 6.7e-8, 0.05e-8),
        ('7.7.9 final gap to conic solve (7.6e-9)', ck['max_gap'], 7.6e-9, 0.05e-9),
        ('7.7.9 median fixed-point steps', ck['median_iter'], 61, None),
        ('7.7.9 refreshes at the 300-step cap (%)', round(100*ck['frac_iter_cap'], 2), 0.04, 0),
        ('7.7.10 alpha = 0: user', N['alpha']['0']['user'], 5.93, 0), ('7.7.10 alpha = 1: user', N['alpha']['1']['user'], 13.96, 0),
        ('7.7.10 alpha = 0: target', N['alpha']['0']['target'], 14.64, 0), ('7.7.10 alpha = 1: target', N['alpha']['1']['target'], 4.08, 0),
    ]
    for k, v in zip(('rdf_as_isac', 'projection', 'socp', 'socp_radius', 'vu_mbwc'), (41, 49, 23, 34, 14)):
        out.append((f'7.7.11 coverage {k} (of 49)', N['coverage'][k], v, None))
    for k, vals in zip(('rdf_as_isac', 'projection', 'socp_radius'), ((0.44, 0.36, 0.39), (0.46, 0.36, 0.40), (0.59, 0.42, 0.48))):
        for sname, v in zip(('sw0', 'sw1', 'sw3'), vals):
            out.append((f'7.7.12 worst-target Pd at Pfa = 1e-4, {k}, {sname}', det[k][sname]['worst'], v, 0))
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    folder = Path(argv[0] if argv else 'figures'); N = json.loads((folder/'results_summary.json').read_text())
    bad = []; n = 0
    E = EXPECTED; ref = N['main_means']['reference']; pv = N['main_paired_avg']['heldout']
    for key, v in E['table_7_3'].items():
        sc, m, i = key.split('|'); n += 1; got = ref[sc][m][int(i)]
        if abs(got - v) > _tol(v): bad.append(('Table 7.3 ' + key, got, v))
    for key, v in E['table_7_4'].items():
        sc, m, w = key.split('|'); c = pv[sc][m][w]; got = [c['mean'], c['lo'], c['hi']]; n += 1
        if any(abs(g - x) > 0.0051 for g, x in zip(got, v)): bad.append(('Table 7.4 ' + key, got, v))
    for key, v in E['table_7_5'].items():
        grp, k, m = key.split('|'); c = N[grp][k][m]; got = [c['user'], c['target']]; n += 1
        if any(abs(g - x) > 0.0501 for g, x in zip(got, v)): bad.append(('Table 7.5 ' + key, got, v))
    for key, v in E['table_7_6'].items():
        sc, k = key.split('|'); c = N['sens'][sc][k]; got = [c['user'], c['target']]; n += 1
        if any(abs(g - x) > 0.0051 for g, x in zip(got, v)): bad.append(('Table 7.6 ' + key, got, v))
    for lab, got, v, tol in text_checks(N):
        n += 1
        ok = (got == v) if tol is None else (abs(got - v) <= (tol if tol else _tol(v)))
        if not ok: bad.append((lab, got, v))
    for b in bad: print('MISMATCH', b)
    print(f'{n - len(bad)} of {n} chapter values reproduced')
    return len(bad)


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
