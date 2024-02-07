# Robust regression

Data points with with high residuals are not treated equally when calculating, they are "down weighted", less important.
But how do we come up with a system to down weight  the less important residuals?. Thats where Huber weighting comes in.
A key component of the Huber weighting is the tuning constant (lets call it `k`). `k` is usually calculated from data, from the variance  
of the residuals.

Thats the general idea behind Huber regression. For more informations see the links below.

[Robust Regression with Huber Weighting](https://www.google.com/search?sca_esv=567555228&sxsrf=AM9HkKm87O67TSyyMfS2xicsAfBLGrZ2OA:1695378277563&q=huber+regression+explained&tbm=vid&source=lnms&sa=X&ved=2ahUKEwiLwfmfgL6BAxUNSPEDHbUcCw0Q0pQJegQICBAB&biw=1916&bih=1056&dpr=1#fpstate=ive&vld=cid:19d8334e,vid:0drbiDPCuYQ,st:0)
[Huber loss](https://www.cantorsparadise.com/huber-loss-why-is-it-like-how-it-is-dcbe47936473)
[Robust regression for machine learning in python](https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/#:~:text=Huber%20Regression&text=The%20%E2%80%9Cepsilon%E2%80%9D%20argument%20controls%20what,model%20more%20robust%20to%20outliers.) 
[Regression in the face of messy outliers](https://towardsdatascience.com/regression-in-the-face-of-messy-outliers-try-huber-regressor-3a54ddc12516)
