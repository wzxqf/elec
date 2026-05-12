# Eberhart和Shi - 2000 - Comparing inertia weights and constriction factors in particle swarm optimization

## Metadata

- source_pdf: 参考文献/Eberhart和Shi - 2000 - Comparing inertia weights and constriction factors in particle swarm optimization.pdf
- extraction_method: pymupdf
- extraction_status: partial
- title: 
- doi: 

## Abstract



## Body

Comparing Inertia Weights and Constriction Factors in Particle Swarm
Optimization
R. C. Eberhart
Purdue School of Engineering and Technology
799 West Michigan Street
Indianapolis, IN 46202 USA
EberhartBengr. iupui.edu
AbstractThe performance
of
particle
swarm
optimization using an inertia weight is compared with
performance
using
a
constriction
factor.
Five
benchmark functions are used for the comparison. It is
concluded that the best approach is to use the
constriction factor while limiting the maximum velocity
Vmax to the dynamic range of the variable Xmax on
each dimension. This approach provides performance
on the benchmark functions superior to any other
published results known by the authors.
1 Introduction
Particle swarm optimization (PSO) is an evolutionary
computation technique motivated by the simulation of
social behavior.
PSO was developed by Kennedy and
Eberhart (Kennedy and Eberhart 1995; Eberhart, Simpson,
and Dobbins 1996).
PSO is similar to a genetic algorithm (GA) in that the
system is initialized with a population of random solutions.
It is unlike a GA, however, in that each potential solution is
also assigned a randomized velocity, and the potential
solutions, called particles, are then "flown" through the
problem space.
The authors have published several papers that describe
research with a version of the particle swarm algorithm that
incorporates what we call an inertia weight (Shi and
Eberhart 1998).
Equations (1) and (2) describe the velocity and position
update equations with the inertia weight included. Equation
(1) calculates a new velocity for each particle (potential
solution) based on its previous velocity, the particle's
location at which the best fitness so far has been achieved,
and the population global (or local neighborhood, in the
neighborhood version of the algorithm) location at which
the best fitness so far has been achieved. Equation (2)
updates each particle's position in solution hyperspace. The
two random numbers are independently generated. The use
of the inertia weight w, which typically decreases linearly
from about 0.9 to 0.4 during a run, has provided improved
performance in a number of applications.
Y. Shi
EDS Indianapolis Technology Center
12400 North Meridian Street
Cannel, IN 46032 USA
Yuhui.Shi@EDS.com
Recent work done by Clerc ( 1999) indicates that use of a
constriction fcrctor may be necessary to insure convergence
of the particle swarm algorithm. A detailed discussion of
the constriction factor is beyond the scope of this paper, but
a simplified method of incorporating it appears in equation
(3), where K is a function of ci and c2 as reflected in
equation (4).
, where p = c, + c 2 , 9 > 4
(4)
K =
2 Experimental Approach
For comparison, five non-linear benchmark functions are
used here.
The first function is the Sphere function
described by equation (5):
(5)
i=l
where x = [xi, x2, ..., X n ] is an n-dimensional real-valued
vector. The second function is the Rosenbrock function
described by equation (6):
i=l
The third function is the generalized Rastrigrin function
described by equation (7):
I 1
fi
( x ) =
(x; - 10 cos(2m; ) + 10)
(7)
i=l
0-7803-6375-2/00/$10.00 02000 IEEE.
Authorized licensed use limited to: CHANGSHA UNIV OF SCIENCE AND TECHNOLOGY. Downloaded on April 30,2026 at 14:08:03 UTC from IEEE Xplore. Restrictions apply.

The fourth function is the generalized Griewank function
described by equation (8):
I1
I'
X .
f3
(x) = -
x; - n
cos(')
+ 1
4000 j = ,
i=l
J;
1.538
The fifth function is Schaffer's f6 function which is
described by Equation (9):
(sin ,/=I2
- 0.5
(1.0+0.001(x2 +-v2))2
fs (x) = 0.5 -
(9)
In all cases, the population size was set to 30, and the
maximum number of iterations was set to 10,000. Each of
these values is somewhat arbitrary. The maximum number
of iterations was set higher than ever found necessary in
previous applications, and at the upper limit of one of the
author's (RE'S) patience.
In all cases for which the inertia weight was used, it was
set to 0.9 at the beginning of the run, and made to decrease
linearly to 0.4 at the maximum number of iterations. Inertia
weight cases used a Vmux set to the maximum range Xmux.
Each of the two (p-x) terms was multiplied by an
acceleration constant of 2.0 (times a random number
between 0 and 1).
In all cases for which Clerc's constriction method was
used, cp was set to 4.1 and the constant multiplier K is thus
0.729.
This resulted in the previous velocity being
multiplied by 0.729, and each of the two (p-x) terms being
multiplied by 0.729 * 2.05 = 1.49445 (times a random
number between 0 and 1). In the initial comparisons, Vmux
was set to 100,000, since it was believed that Vmux isn't
even needed when Clerc's constriction approach is used.
Note that this is functionally equivalent to using equation
(1) with w = .729 and cl = c2 = 1.49445. These new
parameter values are, however, related (see equation (4)).
In all cases (intertia weight and Clerc's constriction
methods) particles were allowed to fly outside of the region
defined by Xmux (see the remarks about flying outside of
the Solar System in Section 4.1).
3 Initial Results
3.1 Introduction
The surprising result of these comparisons is that it appears
that setting Vmux = Xmax significantly improves results
when using the constriction approach. In fact, the bottom
line appears to be that the most consistent way to obtain
good results (and almost always the fastest way) is to use
the constriction approach with Vmax = X m u . But we are
getting ahead of ourselves. Let us first examine the initial
comparisons. Each version of each approach (inertia weight
and constriction factor) was run 20 times for each test
function.
3.2 Spherical Function
The spherical function was always run with 30 dimensions,
the most usually reported in the literature, a value for Xmux
of 100, and was run until an error less than 0.01 was
obtained. Table 1 gives the number of iterations required to
reach this error value using the inertia weight method; Table
2 gives iterations needed using the constriction method.
Table 1: Spherical function iterations using inertia weight
I
1561 I
1537 I 1530 I
I
1615 I
1506 I 1523 I
1553 I
1576 I
The average number of iterations using the inertia weight
is thus 1537.8; the range of values is 130 (from 1485 to
1615), about 8.5 percent of the average.
Table 2: Spherical function iterations using constriction
factor and Vmux=lOOOOO
I 575 I552 I
550 I 559 I
I
I
I
I
542 I 550 I
I
560 I
The average number of iterations using the constriction
factor with Vmax = 100000 is thus 552.05; the range of
values is 96 (from 503 to 599), about 17.4 percent of the
average. The constriction method thus yields faster results,
with a higher range/average quotient.
3.3 Rosenbrock Function
The Rosenbrock function was always run with 30
dimensions, the most usually reported in the literature, a
value for Xmux of 30, and was run until an error less than
100 was obtained. Table 3 gives the number of iterations
required to reach this error value using the inertia weight
method; Table 4 contains iterations needed using the
constriction method.
Table 3: Rosenbrock function iterations using inertia weight
The average number of iterations using the inertia weight
is thus 3517.35; the range of values is 1640 (from 2866 to
4506), about 46.6 percent of the average.
Authorized licensed use limited to: CHANGSHA UNIV OF SCIENCE AND TECHNOLOGY. Downloaded on April 30,2026 at 14:08:03 UTC from IEEE Xplore. Restrictions apply.

Table 4: Rosenbrock function iterations using constriction
factor and Vmux= 100000
I518
I 1122 I655 I 1289 I651
1023 I 700
I 4541
301 1
276 1
I 475
1 796 I 4793 I 2361
The average number of iterations using the constriction
factor with Vmax = 100000 is 1424.1; the range of values is
43 18 (from 475 to 4793), about 303 percent of the average.
The constriction method thus again yields faster results,
with a higher rangelaverage quotient.
3.4 Rastrigrin Function
The Rastrigrin function was always run with 30 dimensions,
the most usually reported in the literature, a value for Xmax
of 5.12, and was run until an error of less than 100 was
obtained.
It was with this test that problems started to
appear with the constriction method (with Vmux = 100000).
Table 5 gives the number of iterations required to reach the
error value of 100 using the inertia weight method. Using
the constriction method, the results for which are in Table 6,
convergence to the specified error value was not achieved in
one of the 20 runs; that run was terminated after 10000
iterations with an error of about 125.
Table 5: Rastrigrin function iterations using inertia weight
1392 I
1122 I
1283 I
1523 I
I
1431 I
I 1367 I
I
I
10000'
10000*
The average number of iterations using the inertia weight
is thus 1320.9; the range of values is 961 (from 743 to
1704), about 73 percent of the average.
10000*
Table 6: Rastrigrin function iterations using constriction
factor and Vmax= 100000
I285
I439
1 283
I329
I258
I248
42 1
I316
I474
I 5000
I 299
The average number of iterations using the constriction
factor with Vmax=lOOOOO is 943 for 19 of the 20 runs. The
target error was not reached within 10000 iterations for one
run. For the 19 successful runs, the range of values is 6823
(from 233 to 7056), about 724 percent of the average. The
constriction method yielded wide variance with this
function, and it is not clear that it would be a better choice
than the inertia weight method for the Rastrigrin test
function.
3.5 Griewank Function
The Griewank function was always run with 30 dimensions,
the most usually reported in the literature, a value for Xmax
of 600, and was initially run until an error of less than 0.05
was obtained. Even more problems were found with the
constriction method (with Vmax=100000) on this function
than on the last one.
Table 7 contains the number of
iterations required to reach the error value of 0.05 with the
inertia weight method.
Table 7: Griewank function iterations using inertia weight,
error = 0.05
The average number of iterations using the inertia weight
(to an error of .05) is thus 2900.5; the range of values is
1335 (from 2556 to 3891), about 46 percent of the average.
Using the constriction factor with Vmax=lOOOOO, in
three runs the full 10000 iterations were run without
achieving the error value of .05. It was decided to relax the
error level to . 1 and try again. Table 8 shows the number of
iterations required to reach the error value of 0.1 using the
constriction method.
Table 8: Griewank fbnction iterations using constriction
factor, error=O. 1, Vmax=10OUOU
I 10000' I 483
I415
I431
I 444
The average number of iterations using the constriction
factor (Vmax=lOOOUO, error=O.l) is 437 for 17 of the 20
runs.
The target error was not achieved within 10000
iterations for three runs. For the 17 successful runs, the
range of values is 279 (from 384 to 663), about 64 percent
of the average.
For comparison, Table 9 contains the
number of iterations required to reach the error value of
0.10 with the inertia weight method.
Authorized licensed use limited to: CHANGSHA UNIV OF SCIENCE AND TECHNOLOGY. Downloaded on April 30,2026 at 14:08:03 UTC from IEEE Xplore. Restrictions apply.

Table 9: Griewank hnction iterations using inertia weight,
errorz0.10
I2718
I2728
t%
1 2664
12947
I 2827
I
5 13
The average number of iterations using the inertia weight
(to an error of 0.10) is 2757.7; the range of values is 397
(from 2638 to 3035), about 14 percent of the average. Even
though the constriction method converged on the desired
error value much more quickly 85 percent of the time, it did
not converge 15 percent of the time, and the inertia weight
method appears more reliable for the Griewank test
fimction.
3.6 Schaffer's f6 Function
Schaffer's f6 function was always run with two dimensions,
as it is defined, a value of Xmux of 100, and was run until an
error value of .00001 was achieved. Table 10 gives the
number of iterations required to reach this error value using
the inertia weight method; Table 1 I contains iterations
needed using the constriction method.
Table 10: Schaffer's f6 hnction iterations using inertia
weight
I422
I449
I627
I 569
I748
53 1
52 1
5 12
The average number of iterations using the inertia weight
is thus 5 12.35; the range of values is 409 (from 339 to 748),
about 80 percent of the average.
Table 1 1: Schaffer's f6 function iterations using constriction
factor, Vmax- 100000
I 832
I 843
I310
I741
I 608
I 156
I239
I 139
The average number of iterations using the constriction
factor is 430.55; the range of values is 794 (from 105 to
899), about 184 percent of the average. The constriction
method
thus
yields
faster
results,
with
a higher
range/average quotient.
4 Observations and Improvements
4.1 Observations
One of the authors (RE) watched the particles "fly" for all of
the runs reported above. It was observed that the variance
using the constriction method and a Vmax of I00000 was
much greater than when using the inertia weight method. In
fact, the scale of the area used to observe the particles had to
be increased by 10 times on both the x and y scales (100
times in area) in order that the constriction method particles
not fly off-screen. It was like watching spacecraft explore
the Milky Way Galaxy in order to find a target known to be
in the Solar System.
If you know that the target is in the Solar System, it
makes sense to limit the distance that can be covered in one
time step to the largest dimension of (distance across) the
system. In our case, we call this maximum distance Xmax.
Note that if we limit our maximum velocity to Xmux, we are
not limiting our exploration to the Solar System; our
spacecraft
particles
can still overshoot the system,
sometimes by a wide range. But we are limiting our search
to at least some reasonable vicinity of the system. It is, of
course, generally assumed that the optimum we are seeking
is somewhere within this dynamic range defined by Xmux.
4.2 Constriction Method with Vmax=Xmax
It was therefore decided to try the constriction method on all
of the test functions configured as before except to set
Vmux=Xmux. The results are presented in Tables 12-16, in
the same order of test functions as above. The results were
surprisingly better.
Table 12: Spherical function iterations using constriction
factor and Vmax=Xmux=IOO
I
I
511 I
I
I
I
I
I
For the spherical function, the average number of
iterations is 529.65; the range is 78 (from 495-573), about
15 percent of the average. This represents an improvement
over the results with Vmux=100000, both in number of
iterations and range.
Table 13: Rosenbrock function iterations using constriction
factor and Vmux=Xmux=30
I
I
k!&
I
I
Authorized licensed use limited to: CHANGSHA UNIV OF SCIENCE AND TECHNOLOGY. Downloaded on April 30,2026 at 14:08:03 UTC from IEEE Xplore. Restrictions apply.

The average number of iterations for the Rosenbrock
function is 668.75; the range is 992 (from 402 to 1394),
about 148 percent of the average.
This represents a
significant
improvement
over
the
results
with
Vmax=100000, both in number of iterations required and
the range.
Table 14: Rastrigrin function iterations using constriction
factor and Vmax=Xmax=5. I2
I
I
I
I
I
I
I
I
I
For the Rastrigrin function, the average number of
iterations is just 213.45; the range is 175 (from 161 to 336),
about 82 percent of the average. This is a very significant
improvement over the results with Vmax=100000, both in
number of iterations required and the range. Note that the
error value was achieved in each of the 20 runs, a result not
obtained with the larger Vmux.
Table 15: Griewank function iterations using constriction
factor, erroI-0.1, Vmax=Xmux=600
I
I
I
I
365 I 294
I
For the Griewank function, the average number of
iterations is 3 12.6; the range is 84 (from 282 to 366), about
27 percent of the average.
This is another significant
improvement over the results with Vmax=lOOOOO, both in
number of iterations required and the range. Note that the
error value was achieved in each of the 20 runs, a result not
obtained with the larger Vmux.
Table 16: Schaffer f6 function iterations using constriction
factor, Vmux=Xmax=lOO
I
I
I
I
257 1
The average for the Schaffer f6 function is thus 532.4;
the range is 1952 (from 94 to 2046), about 367 percent of
the average. Here we have the only case for which the
restricted Vmux did not yield better results. However, if the
one "outlier" value of 2046 is removed, the average for the
remaining 19 runs is about 453, not far from the value
obtained for the larger V m m .
5 Discussion and Conclusion
Comparing the PSO algorithm with inertia
weight
represented by equations (1) and (2) with the algorithm with
constriction factor represented by equations (3) and (4), we
see that equations (1) and (2) are equivalent to equations (3)
and (4) if the inertia weight w is set to be K, and C I and c2
meet
the conditions p = c, +c2, p > 4 .
The PSO
algorithm with the constriction factor can be considered as a
special case of the algorithm with inertia weight since the
three parameters are connected through equation (4).
From the experimental results, it can be concluded that
the best approach to use with particle swarm optimization as
a "rule of thumb" is to utilize the constriction factor
approach while limiting Vmclx to Xmux. or utilize the inertia
weight approach while selecting w, cl, and c2 according to
equation (4). This is easy and straightforward, particularly
since the particle swarm paradigm as currently implemented
requires the specification of Xmax (the Solar System
dimension) anyway. The results here also indicate that
improved performance can be obtained by carefully
selecting the inertia weight w,
C I , and c2. A method to
dynamically adapt the inertia weight is under investigation.
Acknowledgments
The cooperation and comments of Jim Kennedy are
appreciated and acknowledged. Portions of this paper are
adapted from a chapter of Swarm Intelligence: Collective
Adaptation, to be published in about October 2000 by
Morgan KaufmaM Publishers; their permission to use this
material is gratefully acknowledged.
