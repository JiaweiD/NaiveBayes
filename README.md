# NaiveBayes
The theory of NaiveBayse, program realized in python, an example in Machine Learning in Action.

The Bayesian decision theory chooses the best class through minimizing the Bayes risk. In order to get the minimum risk, we should obtain posterior probabilty P(c|x).

Based on Bayes Theorem,
                    P(c|x) = P(x,c)/P(x)
                           = P(c)P(x|c)/P(x)
P(c) is the prior probability, P(x|c) is the likelihood.

We can use Maximum Likelihood Estimation to estimate P(x|c). The likelihood function is:
                    P(Dc|θc) = πP(x|θc)     x∈Dc
While product can cause underflow, we can use log-likelihood:
                    LL(θc) = log(P(Dc|θc))
                           = Σlog(P(x|θc))     x∈Dc
                           
Since P(x|c) is joint probability, in order to avoid this, the naive Bayes classifier uses attribute conditional independence assumption. And the conditional probability formula can be rewritten:
                    P(c|x) = P(c)P(x|c)/P(x)
                           = P(c)/P(x)*Σlog(P(xi|c))     i=1,2,...d
d is the number of attributes. 

Since P(x) is always the same for every class, the naive Bayes classifier expression will be:
                    hnb(x) = argmaxP(c)*Σlog(P(xi|c))     i=1,2,...d   c∈|y|

As for training dataset:
                    P(c) = |Dc|/|D|
                    P(xi|c) = |Dc,xi|/|Dc|
P(xi|c) above is for discrete attributes, while for continuous attributes, we need to consider probability density function. An example of normal distribution(p(xi|c)~N(μi,δi^2)) is as follows:
                    P(xi|c) = 1/sqrt(2π)δi*exp(-(xi-μi)^2/2δi^2)
