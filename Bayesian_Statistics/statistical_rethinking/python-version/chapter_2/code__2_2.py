import scipy.stats as stats


'''
    * The counts of "water" W and "land" L are distbuted binomially, with probability p of "water" on each
    toss.
'''

# The likelihood of the data - six Ws in 9 tosses - under any value of p
# k = number of successes
# n = number of trials
# p = probability of success
print(stats.binom.pmf(k=6, n=9, p=0.5))
