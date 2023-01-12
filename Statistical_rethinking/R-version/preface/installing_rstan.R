#!/usr/bin/Rscript

# run the next line if you already have rstan installed
# remove.packages(c("StanHeaders", "rstan"))

install_package <- function(){
    install.packages("rstan", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}

verify_installation <- function(){
    example(stan_model, package = "rstan", run.dontrun = TRUE)
}

#install_package()
verify_installation()
# For more info check out https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started.
# The package is installed to ~/.lib/R/library/rstan 
# In order to install the package I needed to run the script as sudo.
