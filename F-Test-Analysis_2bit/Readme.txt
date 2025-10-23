Requirements: Python3 + numpy +scipy

Commandline script: python3 RetroAnalysis_MaskedAES.py N \sigma^2
N: number of traces
\sigma^2: random noise variance

Considering AES has 16 S-boxes, $N$ should be larger than $2^16$. Although our paper used $N=10^6$ and $\sigma^2=16$, running this experiment requires a lot of memory (plus longer execution time). In most cases, it is sufficient to use:

python3 RetroAnalysis_MaskedAES.py 300000 16

which costs around 15 mins on our PC. If users prefer an even "lighter" test case:

python3 RetroAnalysis_MaskedAES.py 100000 4

produces the same result within 5 minutes. 


"TVLA_normal.txt": Standard CRI fixed-versus-random test on the plaintext (with fixed key), a leak is confirmed if two times tests both produced $p-value<10^{-6}$

"TVLA_key.txt": Fixed-versus-random test on the key (with fixed plaintext), a leak is confirmed if two times tests both produced $p-value<10^{-6}$

Both tests above rely on the selected "constant" and therefore do not showcase the same result in each run; however, the closure always remains the same as stated in our paper.

"Ftest_key.txt": provides the degree analysis and our Algorithm 1's output. In most cases stays exactly the same as in our paper.