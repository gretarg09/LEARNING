

## Notes from video

* min 32:34 - Idea of time varient coefficient.
* min 34:50 - Gaussian processes add an expressive power to the model that is needed in order to get these models effective.
            They made the model time aware.
* min 38:00 - Instead of adding a single coefficient to descripe your marketing efficiently over time you have a coefficient that is 
            allowed to change over time. There are some informations that you borrow from the past and you bring them into the future. Thats 
            what the gaussian processes (gps) can actually do.
* min 38:47 - Strong priors on how specific groups of marketing channels should behave. the constrains on the specific gaussian processes 
            and time varient coefficient you can start adding structure by saying, these coefficients are allowed to vary over time but
            coefficients that account for similar channels they need to vary with correlated structure, that correlated structure is the 
            hierarchical using processes.
* min 45:55 - Most important aspect of the model was the interactions between upper funnel channels and lower funnel channels in a 
            causal graph. What is going to become more and more important in the future is modeling specific interaction term between 
            variables in a causal sense. In order to do that you need to implement multiple regression models, hierarchy of regression
            models each one of them is related in a specific way.


1. Create a baseline static MMM
2. Allow parameters to evolve over time.
3. Add seasonal effects to the model.

Research graphs:
ad spend in total time series vs revenue time series. I probably need to scale both timeseries to get it working on the same plot.t


## Notes from articles

The Bayesian approach offers a principled way of describing, dealing with, and reducing our uncertainty based upon data.

[FROM reducing customer acquisition ....] We are currently still working with the HelloFresh team on the topic. For example,
Bayesian MMM’s can be used to optimally and automatically set budgets across media channels, rather than just informing those budgets.
And those budgets can be used not only to maximize customer acquisitions but also to drive further learning and marketing experimentation
where we have remaining uncertainty.

In brief, Gaussian processes allow for scalable and flexible modeling of temporal dependencies.


[Video -- Bayesian marketing mix models](https://www.youtube.com/watch?v=xVx91prC81g&t=1034s)
[Video -- Solving Real-World business problems with Bayesian Modeling](https://www.youtube.com/watch?v=twpZhNqVExc) - Here pymc lab is using hello fresh as an example of real world problem.
)
The hello fresh articles, it was a cooperation between pymc labs and hello fresh.
[article .. bayesian media mix modeling using pymc3 for fun and profic](https://engineering.hellofresh.com/bayesian-media-mix-modeling-using-pymc3-for-fun-and-profit-2bd4667504e6)
[article -- Bayesian media mix modeling for marketing optimization](https://www.pymc-labs.com/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
[article -- Reducing customer acquisition costs how we helped optimizing hello fresh marketing budget](https://www.pymc-labs.com/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
[article -- ](https://www.pymc-labs.com/blog-posts/modelling-changes-marketing-effectiveness-over-time/)
