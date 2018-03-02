NMGS Reloaded
=============

This is a modification of the original C code for the paper
[Linking statistical and ecological theory: Hubbell's unified
neutral theory of biodiversity as a hierarchical Dirichlet
process](http://arxiv.org/abs/1410.4038) by Keith Harris et al.

I wanted to have a multi-threaded implementation that could handle large
individual counts, such as we might have in microbial systems. So I
added OpenMP support in places where I thought it would be most
useful. Obviously, parallelizing the Gibbs sampler is not straightforward,
but there were nevertheless quite a few array operations that could be
sped up. The sampling step itself is trivially parallel, so that part
scales very well with the number of cores. Do do this, I had to move
some of the memory allocation into the local functions, which means
that there is a bit more overhead overall, but shouldn't be too much
in comparison with the calculation of random numbers from complicated
distributions (gamma, beta, etc.).

I did not touch the Python, Matlab, and R code, and I have no idea
whether that depends on the C code.

Peace and respect to the original authors of NGMS.

