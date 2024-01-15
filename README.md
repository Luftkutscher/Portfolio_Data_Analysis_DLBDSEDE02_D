# Portfolio_Data_Analysis_DLBDSEDE02_D

Twitter Topic Modelling using TF-IDF and LSA, based on NLTK and Gensim.

The results of the topic modelling of a test run were as follows (please find the diagrams in the root directory of this repository):

~~~

These are the 10 most used #hashtags:

zurich: 31
oliviarodrigo: 17
putin: 16
switzerland: 10
mev: 10
fashion: 9
gc: 8
sports: 6
email: 5
lowcode: 4

These are the 10 most active users:

username
ST0P_PUTIN         30
mehmet2023a        25
ZurichCH           21
paxluxfashion      21
NaokoMogi7575      16
BrunoLauper        15
MoaddebSepideh     13
HinkleHoliday      12
business           10
Tanzeelilyas444     9
Name: count, dtype: int64

Topic Coherence (# Topics vs. UMass Value):

1: -14.251354723880821
2: -14.365256277062944
3: -13.000781857981039
4: -14.251980014707254
5: -14.315272933751064
6: -10.47583039613834
7: -11.914073992647795
8: -11.640511133998718
9: -13.489726281390777
10: -10.94735530601038
11: -13.734449731897508
12: -12.428948681380668
13: -12.127075990519225
14: -12.708152725017564
15: -14.161696167347014
16: -13.191535125196644
17: -13.443106924681354
18: -13.53691845509027
19: -13.669739934103063
20: -12.970313644128606
21: -13.771908982366583
22: -13.731745220617961
23: -14.379863686384182
24: -14.59437436763472
25: -14.729950697679078
26: -15.271715061198218
27: -13.420191940440398
28: -13.990417879575523
29: -13.863857716363265
30: -13.783714416783381
31: -13.113036677232795
32: -13.925777897617362
33: -14.008667174675043
34: -14.341614308475105
35: -14.016373355723077
36: -14.706657720330533
37: -14.152391966495037
38: -14.360771339737553
39: -14.422858110478538
40: -14.689169082637218
41: -15.018394637592392
42: -14.859655445932432
43: -14.626617388062702
44: -14.572561919453943
45: -14.5102518021677
46: -14.42596877117339
47: -14.415898554406054
48: -14.118638002306946
49: -14.942695623388426
50: -14.324863462352528
51: -14.364380311902096
52: -14.645655833494452
53: -14.602477363819595
54: -13.94541413651249
55: -15.342734597051384
56: -14.73701440619684
57: -14.916481579543044
58: -15.007484708072425
59: -14.810767357007629
60: -14.997882964668934
61: -14.767452417603078
62: -14.855307523468984
63: -15.497941404066484
64: -15.080359620031814
65: -14.791386251678555
66: -15.434201528559564
67: -15.233850588238418
68: -14.876405241850902
69: -15.090933242138165
70: -14.78597311532144
71: -15.168694017676678
72: -14.881637737367978
73: -15.19258268111454
74: -14.930025419087752
75: -15.15328501971678
76: -14.916007061888365
77: -15.542710539192722
78: -15.245220651231085
79: -15.178397051985081
80: -15.246330492975625
81: -15.138361112057535
82: -15.388905425440932
83: -15.46279633598466
84: -14.900094569558458
85: -15.019325073119617
86: -14.844684589285947
87: -15.20022026884999
88: -15.244411624237138
89: -15.090564661291431
90: -15.504460934481543
91: -15.292492689332281
92: -15.27304617340146
93: -15.279758452135535
94: -15.663371491161113
95: -15.285639495213953
96: -15.326351454539838
97: -15.485422418936542
98: -15.284902538296388
99: -15.838924436337754
100: -15.438930666411043

These are the Top 6 discussed topics:

[(0,
  '-0.485*"airdrop" + -0.420*"live" + -0.350*"link" + -0.348*"news" + -0.346*"amazing"'),
 (1,
  '0.864*"city" + 0.171*"think" + 0.142*"dream" + 0.141*"dislike" + 0.140*"overrated"'),
 (2,
  '-0.543*"biggest" + -0.535*"pump" + -0.480*"time" + -0.402*"day" + -0.141*"days"'),
 (3,
  '0.462*"amazing" + 0.456*"news" + 0.386*"crypto" + -0.381*"link" + -0.318*"winter"'),
 (4,
  '-0.451*"gift" + -0.438*"claim" + -0.409*"happy" + -0.365*"year" + -0.319*"new"'),
 (5,
  '0.514*"floor" + 0.511*"standing" + 0.336*"june" + 0.314*"may" + 0.187*"london"')]


~~~

# Quickstart

1) Download the provided Jupyter notebook: analysis_twitter_dataset_final.ipynb
2) Download the provided X (former Twitter) dataset and copy it to the same folder as you downloaded the Jupyter notebook: tweets_2477.json (you may use retrieve_twitter_dataset_sanitized.ipynb to download your own dataset, just add your secret and change the query)
3) Install the required libraries (if you are not sure, you can just run the program and see which libs are missing in the error message)
4) Download the NTLK dataset  either manually or by uncommenting # nltk.download() in analysis_twitter_dataset_final.ipynb -> https://www.nltk.org/data.html
5) Run the notebook and enjoy the result!

Sidenote: the umass.png (which is the result of the topic coherence calculation) is not saved automatically, but was copied from the Jupyter notebook output after the test run. It can be saved easily with the command plt.savefig(), if you wish to do so.
