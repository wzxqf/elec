# Deb - 2000 - An efficient constraint handling method for genetic algorithms

## Metadata

- source_pdf: 参考文献/Deb - 2000 - An efficient constraint handling method for genetic algorithms.pdf
- extraction_method: pymupdf
- extraction_status: success
- title: 
- doi: 

## Abstract

Many real-world search and optimization problems involve inequality and/or equality constraints and are thus posed as constrained
optimization problems. In trying to solve constrained optimization problems using genetic algorithms (GAs) or classical optimization
methods, penalty function methods have been the most popular approach, because of their simplicity and ease of implementation.
However, since the penalty function approach is generic and applicable to any type of constraint (linear or nonlinear), their performance is not always satisfactory. Thus, researchers have developed sophisticated penalty functions speci®c to the problem at hand and
the search algorithm used for optimization. However, the most dicult aspect of the penalty function approach is to ®nd appropriate
penalty parameters needed to guide the search towards the constrained optimum. In this paper, GA's population-based approach and
ability to make pair-wise comparison in tournament selection operator are exploited to devise a penalty function approach that does
not require any penalty parameter. Careful comparisons among feasible and infeasible solutions are made so as to provide a search
direction towards the feasible region. Once sucient feasible solutions are found, a niching method (along with a controlled mutation
operator) is used to maintain diversity among feasible solutions. This allows a real-parameter GA's crossover operator to continuously
®nd better feasible solutions, gradually leading the search near the true optimum solution. GAs with this constraint handling approach
have been tested on nine problems commonly used in the literature, including an engineering design problem. In all cases, the proposed
approach has been able to repeatedly ®nd solutions closer to the true optimum solution than that reported earlier. Ó 2000 Elsevier
Science S.A. All rights reserved.

## Body

1. Introduction
Many search and optimization problems in science and engineering involve a number of constraints
which the optimal solution must satisfy. A constrained optimization problem is usually written as a
nonlinear programming (NLP) problem of the following type:
Minimize
f ~x
Subject to
gj~x P 0;
j  1; . . . ; J;
hk~x  0;
k  1; . . . ; K;
xl
i 6 xi 6 xu
i ;
i  1; . . . ; n:
1
In the above NLP problem, there are n variables (that is,~x is a vector of size n), J greater-than-equal-to type
inequality constraints, and K equality constraints. The function f ~x is the objective function, gj~x is the jth
inequality constraints, and hk~x is the kth equality constraints. The ith variable varies in the range xl
i; xu
i .
www.elsevier.com/locate/cma
Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338
E-mail address: deb@iitk.ac.in (K. Deb).
0045-7825/00/$ - see front matter Ó 2000 Elsevier Science S.A. All rights reserved.
PII: S 0 0 4 5 - 7 8 2 5 ( 9 9 ) 0 0 3 8 9 - 8

Constraint handling methods used in classical optimization algorithms can be classi®ed into two groups:
(i) generic methods that do not exploit the mathematical structure (whether linear or nonlinear) of the
constraint, and (ii) speci®c methods that are only applicable to a special type of constraints. Generic
methods, such as the penalty function method, the Lagrange multiplier method, and the complex search
method [1,2] are popular, because each one of them can be easily applied to any problem without much
change in the algorithm. But since these methods are generic, the performance of these methods in most
cases is not satisfactory. However, speci®c methods, such as the cutting plane method, the reduced gradient
method, and the gradient projection method [1,2], are applicable either to problems having convex feasible
regions only or to problems having a few variables, because of increased computational burden with large
number of variables.
Since genetic algorithms (GAs) are generic search methods, most applications of GAs to constraint
optimization problems have used the penalty function approach of handling constraints. The penalty
function approach involves a number of penalty parameters which must be set right in any problem to
obtain feasible solutions. This dependency of GA's performance on penalty parameters has led researchers
to devise sophisticated penalty function approaches such as multi-level penalty functions [3], dynamic
penalty functions [4], and penalty functions involving temperature-based evolution of penalty parameters
with repair operators [5]. All these approaches require extensive experimentation for setting up appropriate
parameters needed to de®ne the penalty function. Michalewicz [6] describes the diculties in each method
and compares the performance of these algorithms on a number of test problems. In a similar study,
Michalewicz and Schoenauer [7] concluded that the static penalty function method (without any sophistication) is a more robust approach than the sophisticated methods. This is because one such sophisticated
method may work well on some problems but may not work so well in another problem.
In this paper, we develop a constraint handling method based on the penalty function approach which
does not require any penalty parameter. The pair-wise comparison used in tournament selection is exploited
to make sure that (i) when two feasible solutions are compared, the one with better objective function value
is chosen, (ii) when one feasible and one infeasible solutions are compared, the feasible solution is chosen,
and (iii) when two infeasible solutions are compared, the one with smaller constraint violation is chosen.
This approach is only applicable to population-based search methods such as GAs or other evolutionary
computation methods. Although at least one other constraint handling method satisfying above three
criteria was suggested earlier [8] it involved penalty parameters which again must be set right for proper
working of the algorithm.
In the rest of the paper, we ®rst show that the performance of a binary-coded GA using the static penalty
function method on an engineering design problem largely depends on the chosen penalty parameter.
Thereafter, we describe the proposed constraint handling method and present the performance of realparameter GAs on nine test problems, including the same engineering design problem. The results are also
compared with best-known solutions obtained using earlier GA implementations or using classical optimization methods.
2. Constraint handling in GAs
In most applications of GAs to constrained optimization problems, the penalty function method has
been used. In the penalty function method for handling inequality constraints in minimization problems,
the ®tness function F ~x is de®ned as the sum of the objective function f ~x and a penalty term which
depends on the constraint violation hgj~xi:
F ~x  f ~x 
X
J
j1
Rjhgj~xi2;
2
where h i denotes the absolute value of the operand, if the operand is negative and returns a value zero,
otherwise. The parameter Rj is the penalty parameter of the jth inequality constraint. The purpose of a
penalty parameter Rj is to make the constraint violation gj~x of the same order of magnitude as the
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

objective function value f ~x. Equality constraints are usually handled by converting them into inequality
constraints as follows: 1
gkJ~x  d ÿ jhk~xj P 0;
where d is a small positive value. This increases the total number of inequality constraints to m  J  K and
the term J in Eq. (2) can then be replaced by m to include all inequality and equality constraints. Thus, there
are total of m penalty parameters Rj which must be set right in a penalty function approach.
In order to reduce the number of penalty parameters, often the constraints are normalized and only one
penalty parameter R is used [1]. In any case, there are two problems associated with this static penalty
function approach:
1. The optimal solution of F ~x depends on penalty parameters Rj (or R). Users usually have to try different values of Rj (or R) to ®nd what value would steer the search towards the feasible region. This requires extensive experimentation to ®nd any reasonable solution. This problem is so severe that some
researchers have used different values of Rj (or R) depending on the level of constraint violation [3],
and some have used sophisticated temperature-based evolution of penalty parameters through generations [5] involving a few parameters describing the rate of evolution.
2. The inclusion of the penalty term distorts the objective function [1]. For small values of Rj (or R), the
distortion is small, but the optimum of F ~x may not be near the true constrained optimum. On the other
hand, if a large Rj (or R) is used, the optimum of F ~x is closer to the true constrained optimum, but the
distortion may be so severe that F ~x may have arti®cial locally optimal solutions. This primarily happens due to interactions among multiple constraints. To avoid such locally optimal solutions, classical
penalty function approach works in sequences, where in every sequence the penalty parameters are increased in steps and the current sequence of optimization begins from the optimized solution found in
the previous sequence. This way a controlled search is possible and locally optimal solutions can be
avoided. However, most classical methods use gradient-based search methods and usually have dif®culty
in solving discrete search space problems and to problems having a large number of variables. Although
GAs do not use gradient information, they are not free from the distortion effect caused due to the addition of the penalty term with the objective function. However, GAs are comparatively less sensitive to
distorted function landscapes due to the stochasticity in their operators.
In order to investigate the eect of the penalty parameter Rj (or R) on the performance of GAs, we
consider a well-studied welded beam design problem [2]. The resulting optimization problem has four
design variables ~x  h; `; t; b and ®ve inequality constraints:
Minimize
fw~x  1:10471h2`  0:04811tb14:0  `
Subject to
g1~x  13; 600 ÿ s~x P 0;
g2~x  30; 000 ÿ r~x P 0;
g3~x  b ÿ h P 0;
g4~x  Pc~x ÿ 6; 000 P 0;
g5~x  0:25 ÿ d~x P 0;
0:125 6 h 6 10; 0:1 6 `; t; b 6 10:
3
The terms s~x, r~x, Pc~x, and d~x are given below:
s~x 

s0~x2  s00~x2  `s0~xs00~x=

0:25`2  h  t2
q
r
r~x  504 000
t2b
1 It is important to note that this transformation makes the resulting inequality constraint function nondierentiable, thereby causing
diculty to many classical search and optimization algorithms to use this transformation. In those cases, an equality constraint is
converted into two inequality constraints hk~x 6 d and hk~x P ÿ d.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

Pc~x  64746:0221 ÿ 0:0282346ttb3;
d~x  2:1952
t3b
where
s0~x  6000
p
h`
s00~x 
600014  0:5`

0:25`2  h  t2
q
2 0:707h``2=12  0:25h  t2
n
o :
The optimized solution reported in the literature [2] is h  0:2444, `  6:2187, t  8:2915, and
b  0:2444 with a function value equal to f   2:38116. Binary GAs are applied on this problem in an
earlier study [9] and the solution~x  0:2489; 6:1730; 8:1789; 0:2533 with f  2:43 (within 2% of the above
best solution) was obtained with a population size of 100. However, it was observed that the performance
of GAs largely dependent on the chosen penalty parameter values.
In order to get more insights on the working of GAs, we apply binary GAs with tournament selection
without replacement and single-point crossover operator with pc  0:9 on this problem. In the tournament
selection, two solutions are picked at random from the population and are compared based on their ®tness
(F ~x) values. The better solution is chosen and kept in an intermediate population. This process is continued till all N population slots are ®lled. This operation is usually performed systematically, so the best
solution in a population always get exactly two copies in the intermediate population. Each variable is
coded in 10 bits, so that total string length is 40. A population size of 80 is used and GAs with 50 different
initial populations are run. GAs are run till 500 generations. All constraints are normalized (for example,
the ®rst constraint is normalized as 1 ÿ s~x=13 600 P 0, and so on) and a single penalty parameter R is
used. Table 1 shows the performance of binary GAs for different penalty parameter values.
For each case, the best, median, 2 and worst values of 50 optimized objective function values are also
shown in the table. With R  1, though three out of 50 runs have found a solution within 10% of the bestknown solution, 13 GA runs have not been able to ®nd a single feasible solution in 40 080 function
evaluations. This happens because with small R there is not much pressure for the solutions to become
feasible. With large penalty parameters, the pressure for solutions to become feasible is more and all 50 runs
found feasible solutions. However, because of larger emphasis of solutions to become feasible, when a
particular solution becomes feasible it has a large selective advantage over other solutions (which are infeasible) in the population. If new and different feasible solutions are not created, GAs would overemphasize this sole feasible solution and soon prematurely converge near this solution. This has exactly
happened in GA runs with larger R values, where the best solution obtained is, in most cases, more than
50% away (in terms of function values) from the true constrained optimum.
Similar experiences have been reported by other researchers in applying GAs with penalty function
approach to constrained optimization problems. Thus, if penalty function method is to be used, the user
usually have to take many runs or `adjust' the penalty parameters to get a solution within an acceptable
limit. In a later section, we shall revisit this welded beam design problem and show how the proposed
constrained handling method ®nds solutions very close to the true optimum reliably and without the need
of using any penalty parameter.
Michalewicz [6] and later Michalewicz and Schoenauer [7] have discussed dierent constraint handling
methods used in GAs. They have classi®ed most of the evolutionary constraint handling methods into ®ve
categories:
2 The optimized objective function values (of 50 runs) are arranged in ascending order and the 25th value in the list is called the
median optimized function value.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

1. Methods based on preserving feasibility of solutions;
2. Methods based on penalty functions;
3. Methods making distinction between feasible and infeasible solutions;
4. Methods based on decoders;
5. Hybrid methods.
The methods under the ®rst category explicitly use the knowledge of the structure of the constraints and use
a search operator that maintains the feasibility of solutions. Second class of methods uses penalty functions
of various kinds, including dynamic penalty approaches where penalty parameter are adapted dynamically
over time. The third class of constraint handling methods uses dierent search operators for handling
infeasible and feasible solutions. The fourth class of methods uses an indirect representation scheme which
carries instructions for constructing a feasible solution. In the ®fth category, evolutionary methods are
combined with heuristic rules or classical constrained search methods. Michalewicz and Schoenauer [7]
have compared dierent algorithms on a number of test problems and observed that each method works
well on some classes of problems whereas does not work well on other problems. Owing to this inconsistency in the performance of dierent methods, they suggested to use the static penalty function method,
similar to that given in Eq. (2). Recently, a two-phase evolutionary programming (EP) method is developed
[10]. In the ®rst phase, a standard EP technique with a number of strategy parameters which were evolved
during the optimization process was used. With the solution obtained in the ®rst phase, a neural network
method was used in the second phase to improve the solution. The performance of the second phase depends on how close a solution to the true optimal solution is found in the ®rst phase. The approach involves
too many dierent procedures with many control parameters and it is unclear which procedure and parameter settings are important. Moreover, out of the six test problems used in the study, ®ve were twovariable problems having at most two constraints. It is unclear how this rather highly sophisticated method
will scale up its performance to more complex problems.
In Section 3, we present a dierent yet simple penalty function approach which does not require any penalty
parameter, thereby making the approach applicable to a wide variety of constrained optimization problems.
3. Proposed constraint handling method
The proposed method belongs to both second and third categories of constraint handling methods
described by Michalewicz and Schoenauer [7]. Although a penalty term is added to the objective function to
penalize infeasible solutions, the method diers from the way the penalty term is de®ned in conventional
methods and in earlier GA implementations.
The method proposes to use a tournament selection operator, where two solutions are compared at a
time, and the following criteria are always enforced [11]:
1. Any feasible solution is preferred to any infeasible solution.
2. Among two feasible solutions, the one having better objective function value is preferred.
3. Among two infeasible solutions, the one having smaller constraint violation is preferred.
Although there exist a number of other implementations [6,8,12] where criteria similar to the above are
imposed in their constraint handling approaches, all of these implementations used dierent measures of
constraint violations which still needed a penalty parameter for each constraint.
Table 1
Number of runs (out of 50 runs) converged within % of the best-known solution using binary GAs with dierent penalty parameter
values on the welded beam design problem
R
Infeasible
Optimized fw~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
2.41324
7.62465
483.50177
3.14206
4.33457
7.45453
3.38227
5.97060
10.65891
3.72929
5.87715
9.42353
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

Recall that penalty parameters are needed to make the constraint violation values of the same order as
the objective function value. In the proposed method, penalty parameters are not needed because in any of
the above three scenarios, solutions are never compared in terms of both objective function and constraint
violation information. Of the three tournament cases mentioned above, in the ®rst case, neither objective
function value nor the constraint violation information is used, simply the feasible solution is preferred. In
the second case, solutions are compared in terms of objective function values alone and in the third case,
solutions are compared in terms of the constraint violation information alone. Moreover, the idea of
comparing infeasible solutions only in terms of constraint violation has a practical implication. In order to
evaluate any solution (say a particular solution of the welded beam problem discussed earlier), it is a usual
practice to ®rst check the feasibility of the solution. If the solution is infeasible (that is, at least one constraint is violated), the designer will never bother to compute its objective function value (such as the cost of
the design). It does not make sense to compute the objective function value of an infeasible solution, because the solution simply cannot be implemented in practice.
Motivated by these arguments, we devise the following ®tness function, where infeasible solutions are
compared based on only their constraint violation:
F ~x 
f ~x
if gj~x P 0
8j  1; 2; . . . ; m;
fmax  Pm
j1hgj~xi
otherwise:
4
The parameter fmax is the objective function value of the worst feasible solution in the population. Thus,
the ®tness of an infeasible solution not only depends on the amount of constraint violation, but also on the
population of solutions at hand. However, the ®tness of a feasible solution is always ®xed and is equal to its
objective function value.
We shall ®rst illustrate this constraint handling technique on a single-variable constrained minimization
problem and later show its eect on contours of a two-dimensional problem. In Fig. 1, the ®tness function
F ~x (thick solid line in infeasible region and dashed line in feasible region) are shown. The unconstrained
minimum solution is not feasible here. It is important to note that F ~x  f ~x in the feasible region and there
is a gradual 3 increase in ®tness for infeasible solutions away from the constraint boundary. Under the
tournament selection operator mentioned earlier, there will be selective pressure for infeasible solutions to
Fig. 1. The proposed constraint handling scheme is illustrated. Six solid circles are solutions in a GA population.
3 Although, in some cases, it is apparent that the above strategy may face trouble where constraint violations may not increase
monotonically from the constraint boundary inside the infeasible region [13], this may not be a problem to GAs. Since the above
strategy guarantees that the ®tness of any feasible solution is better than ®tness of all infeasible solutions in a population, once a
feasible solution is found, such nonlinearity in constraint violations may not matter much. However, this needs a closer look which we
plan to investigate in a future study.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

come closer and inside the feasible region. The ®gure also shows how the ®tness value of six population
members (shown by solid bullets) will be evaluated. It is interesting to note how the ®tness of infeasible solutions depends on the worst feasible solution. If no feasible solution exists in a population, fmax is set to zero.
It is important to reiterate that since solutions are not compared in terms of both objective function
value and constraint violation information, there is no need of any explicit penalty parameter in the
proposed method. This is a major advantage of the proposed method over earlier penalty function implementations using GAs. However, to avoid any bias from any particular constraint, all constraints are
normalized (a usual practice in constrained optimization [1]) and Eq. (4) is used. It is important to note that
such a constraint handling scheme without the need of a penalty parameter is possible because GAs use a
population of solutions in every iteration and a pair-wise comparison of solutions is possible using the
tournament selection operator. For the same reason, such schemes cannot be used with classical pointby-point search and optimization methods.
The proposed constraint handling technique is better illustrated in Figs. 2 and 3, where ®tness function is
shown by drawing contours of the following NLP problem:
Minimize
f x; y  x ÿ 0:82  y ÿ 0:32
Subject to
g1x; y  1 ÿ x ÿ 0:22  y ÿ 0:52=0:16 P 0;
g2x; y  x  0:52  y ÿ 0:52=0:81 ÿ 1 P 0:
5
The contours have higher function values as they move out of the point x; y  0:8; 0:3. Fig. 2 shows
the contour plot of the objective function f x; y and the crescent shaped (nonconvex) feasible region
formed by g1x; y and g2x; y constraint functions. Assuming that the worst feasible solution in a population lie at 0:35; 0:85 (the point marked by a o in the ®gure), the corresponding fmax  0:505. Fig. 3 shows
the contour plot of the ®tness function F x; y (calculated using Eq. (4)). It is interesting to note that the
contours do not get changed inside the feasible region, whereas they become parallel to the constraint
surface outside the feasible region. Thus, when most solutions in a population are infeasible, the search
forces solutions to come closer to feasible region. Once sucient solutions exist inside the feasible region,
the search gets directed by the eect of the objective function alone. In the case of multiple disconnected
feasible regions, the ®tness function has a number of such attractors, one corresponding to each feasible
region. When solutions come inside feasible regions, the selection operator mainly works with the true
objective function value and helps to focus the search in the correct (global) feasible region.
Fig. 2. Contour plot of the objective function f x; y and the feasible search space are shown. Contours are plotted at f x; y values
0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, and 1.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

We have realized that the proposed method is somewhat similar to PS [8] method, which involves penalty
parameters. Thus, like other penalty function approaches, PS method is also sensitive to penalty parameters. Moreover, the PS method may sometime create arti®cial local optima, as discussed in the following.
Consider the same single-variable function shown in Fig. 1. The calculation procedure of the ®tness
function in PS method is illustrated in Fig. 4.
The major dierence between the PS method and the proposed method is that in the PS method the
objective function value is considered in calculating the ®tness of infeasible solutions. In the PS method,
the penalized function value 4 f ~x  R P
jhgj~xi is raised by an amount k (shown in the ®gure) to make the
Fig. 4. Powell and Skolnick's constraint handling scheme is illustrated. Six solid circles are solutions in a GA population.
4 In Powell and Skolnick's study, the square of constraint violation was used. Although, this changes relative importance of
constraint violation with respect to the objective function value in PS method, it does not matter in the proposed approach, because of
the use of tournament selection.
Fig. 3. Contour plot of the ®tness function F x; y at a particular generation is shown. Contours are plotted at F x; y values 0.1, 0.2,
0.3, 0.4, 0.5, 0.75, 1, 2, 3, and 4.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

®tness of the best infeasible solution equal to the ®tness of the worst feasible solution. Fig. 4 shows that, in
certain situations, the resulting ®tness function (shown by a long dashed line) may have an arti®cial
minimum in the infeasible region. When the feasible region is narrow, there may not be many feasible
solutions present in a population. In such a case, GAs with this constraint handling method may get
trapped into this arti®cial local optimum. It is worth mentioning that the effect of this arti®cial local optimum can get reduced if a large enough penalty parameter R is used. This dependency of a constraint
handling method on the penalty parameter is not desirable (and the meaning of `large penalty parameter' is
subjective to the problem at hand) and has often led researchers to rerun an optimization algorithm with
different values of penalty parameters.
3.1. Binary versus real-coded GAs
The results of the welded beam design problem presented in Section 2 are all achieved with binary GAs,
where all variables are coded in binary strings. It is intuitive that the feasible region in constrained optimization problems may be of any shape (convex or concave and connected or disjointed). In real-parameter
constrained optimization using GAs, schemata specifying contiguous regions in the search space (such as
110      may be considered to be more important than schemata specifying discrete regions in the
search space (such as 1  10     , in general. In a binary GA under a single-point crossover operator,
all common schemata corresponding to both parent strings are preserved in both children strings. Since,
any arbitrary contiguous region in the search space cannot be represented by a single Holland's schema and
since the feasible search space can usually be of any arbitrary shape, it is expected that the single-point
crossover operator used in binary GAs may not always be able to create feasible children solutions from
two feasible parent solutions. Moreover, in most cases, such problems have feasible region which is a tiny
fraction of the entire search space. Thus, once feasible parent solutions are found, a controlled crossover
operator is desired in order to (hopefully) create children solutions which are also feasible.
The ¯oating-point representation of variables in a GA and a search operator that respects contiguous
regions in the search space may be able to eliminate the above two diculties associated with binary coding
and single-point crossover. In this paper, we use real-coded GAs with simulated binary crossover (SBX)
operator [14] and a parameter-based mutation operator [15], for this purpose. SBX operator is particularly
suitable here, because the spread of children solutions around parent solutions can be controlled using a
distribution index gc (see Appendix A). With this operator any arbitrary contiguous region can be searched,
provided there is enough diversity maintained among the feasible parent solutions. Let us illustrate this
aspect with the help of Fig. 3. Note that the constrained optimum is at the lower half of the crescent-shaped
feasible region (on g1x; y constraint). Although a population may contain solutions representing both the
lower and the upper half of the feasible region, solutions in the lower half are more important, though the
representative solutions in the lower half may have inferior objective function values compared to those in
the upper half. In such cases, the representative solutions of the lower half must be restored in the population, in the hope of ®nding better solutions by the action of the crossover operator. Thus, maintaining
diversity among feasible solutions is an important task, which will allow a crossover operator to constantly
®nd better feasible solutions.
There are a number of ways diversity can be maintained in a population. Among them, niching methods
[16] and use of mutation [17] are popular ones. In this paper, we use either or both of the above methods of
maintaining diversity among the feasible solutions. A simple niching strategy is implemented in the tournament selection operator. When comparing two feasible solutions (i and j), a normalized Euclidean distance dij is measured between them. If this distance is smaller than a critical distance d, the solutions are
compared with their objective function values. Otherwise, they are not compared and another solution j is
checked. If a speci®c number (nf) of feasible solutions are checked and none is found to qualify within the
critical distance, the ith solution is declared as winner. The normalized Euclidean distance is calculated as
follows:
dij 

n
X
n
k1
xi
k ÿ xj
k
xu
k ÿ xl
k

!2
v
u
u
t
6
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

This way, the solutions that are far away from each other are not compared and diversity among feasible
solutions can be maintained.
3.2. Evolutionary strategies versus real-coded GAs
Evolutionary strategies (ESs) are evolutionary optimization methods which work on ¯oating-point
numbers directly [18,19]. The main dierence in the working principles of an ES and a real-coded GA is that
in ES mutation operator is the main search operator. ES also uses a block truncation selection operator,
which is dierent from the tournament selection operator. Moreover, an ES uses two dierent populations
(parent and children populations) with children population size about an order of magnitude larger than
that of the parent population size. It is highlighted earlier that the population approach and the ability to
compare solutions pairwise are two essential features of the proposed constraint handling method. Although an ES uses a population approach, it usually does not make a pairwise comparison of solutions.
Although a tournament selection scheme can be introduced in an ES, it remains an open question as to how
such an ES will work in general.
Moreover, there exists a plethora of other implementations of GAs such as multi-modal GAs, multiobjective GAs, and others, which have been successfully implemented with real-coded GAs [20]. We believe
that the constraint handling strategy suggested in this study can also be easily incorporated along with
various other kinds of existing real-coded GAs.
Thus, for the sake of simplicity in implementation, we have tested the constraint handling strategy with
real-coded GAs, instead with an ES framework. We are currently working on implementing the proposed
constraint handling method with an ES framework and results comparing real-coded GAs and ESs will be
reported at a later date.
4. Results
In this section, we apply GAs with the proposed constraint handling method to nine dierent constrained optimization problems that have been studied in the literature.
In all problems, we run GAs 50 times from dierent initial populations. Fixing the correct population
size in a problem is an important factor for proper working of a GA. Previous population sizing considerations [21,22] based on schema processing suggested that the population size should increase with the
problem size. Although the correct population size should also depend on the underlying signal-to-noise in
a problem, here we follow a simple procedure of calculating the population size: N  10n, where n is the
number of variables in a problem. In all problems, we use binary tournament selection operator without
replacement. We use a crossover probability of 0.9. When binary-coded GAs are used, the single-point
crossover operator is used. When real-coded GAs are used, simulated binary crossover (SBX) is used [14].
The SBX procedure is described brie¯y in Appendix A. When mutation is used, the bit-wise mutation
operator is used for binary GAs and a parameter-based mutation is used for real-coded GAs. This procedure is also described in Appendix A. Wherever niching is used, we have used d  0:1 and nf  0:25N.
4.1. Test problem 1
To investigate the ecacy of the proposed constraint handling method, we ®rst choose a two-dimensional constrained minimization problem:
Minimize
f1~x  x2
1  x2 ÿ 112  x1  x2
2 ÿ 72
Subject to
g1~x  4:84 ÿ x1 ÿ 0:052 ÿ x2 ÿ 2:52 P 0;
g2~x  x2
1  x2 ÿ 2:52 ÿ 4:84 P 0;
0 6 x1 6 6; 0 6 x2 6 6:
7
The unconstrained objective function f1x1; x2 has a minimum solution at (3,2) with a function value
equal to zero. However, due to the presence of constraints, this solution is no more feasible and the
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

constrained optimum solution is x  2:246826; 2:381865 with a function value equal to f 
1  13:59085.
The feasible region is a narrow crescent-shaped region (approximately 0.7% of the total search space) with
the optimum solution lying on the ®rst constraint, as shown in Fig. 5.
Niching and mutation operators are not used here. We have run GAs till 50 generations. Powell and
Skolnick's [8] constraint handling method (PS) is implemented with the real-coded GAs and with tournament selection and the SBX operator. With a penalty parameter R  1 for both constraints, the performance of GAs is tabulated in Table 2. The table shows that 11 out of 50 runs cannot ®nd a single feasible
solution with Powell and Skolnick's method with R  1, whereas the proposed method (TS-R) ®nds a
feasible solution every time. Moreover, 58% runs have found a solution within 1% of the true optimum
solution. The dependency of PS method on the penalty parameter R is also clear from the table.
In order to investigate the performance of the binary GA on this problem, binary GAs with the proposed
constraint handling method (TS-B) is applied next. Each variable is coded in 20 bits. Binary GAs ®nd
solutions within 1% and 50% of the optimum solution in only 2 and 13 out of 50 runs, respectively. Although, all 50 GA runs are able to ®nd feasible solutions, the performance (best 13.59658, median 37.90495,
and worst 244.11616) is not as good as that of real-coded GAs.
In runs where PS method did not ®nd a feasible solution, GAs have converged to an arti®cially created
minimum solution in the infeasible region. We show the proceedings of one such run in Fig. 5 with R  1.
The initial population of 50 random solutions show that initially solutions exist all over the search space (no
Fig. 5. Population history at initial generation (marked with open circles), at generation 10 (marked with ) and at generation 50
(marked with open boxes) using PS method (R  1) on test problem 1. The population converges to a wrong, infeasible solution.
Table 2
Number of runs (out of 50 runs) converged within % of the optimum solution for real-coded GAs with two constraint handling
techniques ± PS method with dierent R values and the proposed method (TS-R) ± on test problem 1
Method
Infeasible
Optimized f1~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
PS (R  0:01)
13.58958
24.07437
114.69033
PS (R  1)
13.59108
16.35284
172.81369
TS-R
13.59085
13.61673
117.02971
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

solution is feasible in the initial population). After 10 generations, a real-coded GA with Powell and
Skolnick's constraint handling strategy (with R  1) could not drive the solutions towards the narrow
feasible region. Instead, the solutions get stuck at a solution ~x  2:891103; 2:11839 (with a function value
equal to 0.41708), which is closer to the unconstrained minimum at (3,2) (albeit infeasible). The reason for
such suboptimal convergence is discussed earlier in Fig. 4. When an identical real-coded GA but with the
proposed constraint handling strategy (TS-R) is applied to the identical initial populations of 50 solutions
(rest all parameter settings are also the same as in the Powell and Skolnick's case), the GA distributes well
its population around and inside the feasible region (Fig. 6) after 10 generations. Finally, GAs converge
near to the true optimum solution at ~x  2:243636; 2:342702 with a function value equal to 13.66464
(within 0.54% of the true optimum solution).
Fig. 6. Population history at initial generation (marked with open circles), at generation 10 (marked with ) and at generation 50
(marked with open boxes) using the proposed scheme on test problem 1. The population converges to a solution very close to the true
constrained optimum solution on a constraint boundary.
Fig. 7. Comparison of the proposed (TS) and Powell and Skolnick's (PS) methods for constraint handling in terms of average number
of feasible solutions found in 50 GA runs on test problem 1.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

The number of feasible solutions found in each generation in all 50 runs are noted and their average is
plotted in Fig. 7. In the initial generation, there are not many feasible solutions (about 0.7%). Thereafter,
the number of feasible solutions increase rapidly for both binary and real-coded GAs with the proposed
constraint handling scheme. At around generation 25, more than 90% population members are feasible,
whereas GAs with Powell and Skolnick's constraint handling strategy the initial rate of feasible solution
discovery is also slower and GAs have found less than 50% of their population members in the feasible
region.
Although binary GAs have found slightly more solutions in the feasible region that that found by realcoded GAs in this problem, Fig. 8 shows that the average Euclidean distance among feasible solutions for
the binary GAs is smaller than that for the real-coded GAs. This means that real-coded GAs is able to
spread solutions better, thereby allowing their search operators to ®nd better solutions. This is the reason
why real-coded GAs has performed better than binary GAs. In the following, we compare these GAs to a
more complicated test problem.
4.2. Test problem 2
This problem is a minimization problem with ®ve variables and 38 inequality constraints [23,24]:
Minimize
f2~x  0:1365 ÿ 5:84310ÿ7y17  1:1710ÿ4y14  2:35810ÿ5y13  1:50210ÿ6y16
0:0321y12  0:004324y5  1:010ÿ4c15=c16  37:48y2=c12
Subject to
g1~x  1:5x2 ÿ x3 P 0;
g2~x  y1~x ÿ 213:1 P 0;
g3~x  405:23 ÿ y1~x P 0;
gj2~x  yj~x ÿ aj P 0;
j  2; . . . ; 17;
gj18~x  bj~x ÿ yj~x P 0;
j  2; . . . ; 17;
g36~x  y4~x ÿ 0:28=0:72y5~x P 0;
g37~x  21 ÿ 3496:0y2~x=c12~x P 0;
g38~x  62212:0=c17~x ÿ 110:6 ÿ y1~x P 0;
704:4148 6 x1 6 906:3855;
68:6 6 x2 6 288:88;
0 6 x3 6 134:75;
193 6 x4 6 287:0966;
25 6 x5 6 84:1988:
8
The terms yj~x and cj~x, and parameters aj and bj are given in Appendix B. The best solution reported
in [23] and in [24] is
Fig. 8. Comparison of the proposed (TS) and Powell and Skolnick's (PS) methods for constraint handling in terms of average normalized Euclidean distance among feasible solutions in 50 GA runs on test problem 1.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

~x  705:1803; 68:60005; 102:90001; 282:324999; 37:5850413;
f 
2  ÿ1:90513:
At this solution, none of the 38 constraints is active (an inequality constraint is active at any solution if
the constraint violation is zero at that solution). Thus, this solution lies inside the feasible region. 5 This
function is particularly chosen to test the proposed constraint handling method on a problem having a large
number of constraints.
Table 3 shows the performance of real-coded GAs with the proposed constraint handling scheme with a
population size 10  5 or 50. Powell and Skolnick's (PS) constraint handling method depends on the the
penalty parameter used. For a large penalty parameter, PS method is similar in performance to the proposed method (TS-R). However, for small penalty parameter values, PS method does not perform well. The
proposed method of this study (TS) does not require any penalty parameter. The performance of GAs with
the proposed method (TS-R) improves with niching and further with the mutation operator. With mutation, all 50 runs have found solutions better than the best solution reported earlier.
However, binary GAs with the proposed scheme (TS-B) cannot ®nd feasible solutions in 9 runs and the
best run found a solution within about 13% of the best-known solution. Six runs have found feasible solutions having an objective function value more than 150% of that of the best-known solution. The best,
median, and worst function values are ÿ1:66316, ÿ1:20484, and ÿ0:73044, respectively.
Fig. 9 shows the average of the total normalized Euclidean distance of all feasible solutions in each
iteration. It is clear that with the presence of niching, the average Euclidean distance of feasible solutions
increases, meaning that there is more diversity present among the feasible solutions. With the introduction
of mutation, this diversity further increases and GAs perform the best. Once again, this ®gure shows that
real-coded GAs with PS (R  1) and binary GAs with the proposed scheme have not been able to ®nd and
distribute solutions well in the feasible region.
It is also interesting to note that the best solutions obtained with real-coded GAs (TS-R) is better than
that reported in [23,24]. The solution here is
~x  707:337769; 68:600273; 102:900146; 282:024841; 84:198792;
f2  ÿ1:91460;
which is about 0.5% better in the objective function value than that reported earlier. The main dierence
between this solution and that reported earlier is in the value of x5. At this solution, ®ve constraints (g1, g2,
g34, g35, and g38) are active with constraint values less than 10ÿ3. The ratio of the best f2~x obtained in a GA
5 However, we shall see later in this section that this solution is not the true optimal solution. The solution obtained in this study is
better than this solution and makes 5 of 38 constraints active.
Table 3
Number of runs (out of 50 runs) converged within % of the best-known solution using real-coded GAs with the proposed constraint
handling scheme (TS-R) and using PS-a method with different penalty parameters R  10a on test problem 2.
Method
Mutation
Niching
Infeasible
Optimized f2~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
Best
Median
Worst
PS-0
No
No
ÿ1:86365
ÿ1:69507
ÿ1:35910
PS-2
No
No
ÿ1:89845
ÿ1:65156
ÿ1:00969
PS-6
No
No
ÿ1:91319
ÿ1:65763
ÿ1:11550
TS-R
No
No
ÿ1:91319
ÿ1:65763
ÿ1:11550
TS-R
No
Yes
ÿ1:91410
ÿ1:85504
ÿ1:30643
TS-R
Yes
Yes
ÿ1:91460
ÿ1:91457
ÿ1:91454
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

generation and the best-known f2~x (that is, f 
2  ÿ1:90513) is calculated for all 50 runs and their average
is plotted in Fig. 10 for dierent GA implementations.
Since, f 
2 is negative, for any suboptimal solution, the ratio f ~x=f 
2 would be smaller than one. When
this ratio is close to one, it is clear that the best-known solution x is found. The ®gure shows how realcoded GAs with the Powell and Skolnick's (PS) constraint handling method with R  1 get stuck at
suboptimal solutions. The average value of f ~x where GAs converge in 50 runs is even less than 20% of f 
2 .
However, real-coded GAs with the proposed constraint handling scheme ®nds this ratio greater than 0.8.
This ratio further increases to more than 0.9 with niching alone. The ®gure also shows that for GAs with
niching and mutation the ratio is little better than 1.0, indicating that better solutions than that reported
earlier have been obtained in this study.
Because of the dependency of the performance of Powell and Skolnick's (PS) method on the penalty
parameter, we do not apply this method in the subsequent test problems and only present the results for
GAs with the proposed constraint handling method. Since binary GAs with the proposed constraint
handling scheme also do not perform well on both the above constrained optimization problems (mainly
Fig. 10. Average ratio of the best f~x found by GAs to f 
2 is plotted versus generation number on test problem 2.
Fig. 9. Average normalized Euclidean distance of feasible solutions versus generation number on test problem 2.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

due to its inability to maintain diverse solutions in the feasible region), we also do not apply binary GAs to
subsequent test problems.
4.3. Test problem 3
The problem is a minimization problem having 13 variables and nine inequality constraints [6]:
Minimize
f3~x  5 P4
i1 xi ÿ 5 P4
i1 x2
i ÿ P13
i5 xi
Subject to
g1~x  2x1  2x2  x10  x11 6 10;
g2~x  2x1  2x3  x10  x12 6 10;
g3~x  2x2  2x3  x11  x12 6 10;
g4~x  ÿ8x1  x10 6 0;
g5~x  ÿ8x2  x11 6 0;
g6~x  ÿ8x3  x12 6 0;
g7~x  ÿ2x4 ÿ x5  x10 6 0;
g8~x  ÿ2x6 ÿ x7  x11 6 0;
g9~x  ÿ2x8 ÿ x9  x12 6 0;
0 6 xi 6 1;
i  1; . . . ; 9;
0 6 xi 6 100;
i  10; 11; 12;
0 6 x13 6 1:
9
The optimal solution to this problem is
~x  1; 1; 1; 1; 1; 1; 1; 1; 1; 3; 3; 3; 1;
f 
3  ÿ15:
At this optimal solution, six constraints (all except g4, g5, and g6) are active. This is a relatively easy
problem with the objective function and constraints being linear or quadratic. Michalewicz [6] reported that
all constraint handling methods used to solve this problem have found the optimal solution. Not surprisingly, all methods tried here have also found the true optimal solution many times, as depicted in
Table 4. However, it is important to note that here no eort has been spent to exploit the structure of the
constraints, whereas in the other study [6] special closed operators (in addition to standard GA operators)
are applied on linear constraints to satisfy them. Although a similar approach can also be used with the
proposed method, we do not consider the special cases here (because such operators can only be used to a
special class of constraints), instead present a generic strategy for solving constraint optimization problems.
GA parameters are set as before. Since there are 13 variables, a population size of (10  13) or 130 is
used. With the presence of niching, the performance of GAs becomes better and 38 out of 50 runs have
found solutions within 1% from the true optimum. With the presence of niching and mutation, the performance of GAs is even better.
Average normalized Euclidean distance of feasible solutions are plotted in Fig. 11 and average ratio of
the best ®tness obtained by GAs to the best-known objective function value f 
3 is plotted in Fig. 12. Figures
show how diversity among feasible solutions is restored in GAs with niching and mutation. The latter ®gure
also shows the suboptimal convergence of GAs without niching in some runs.
Table 4
Number of runs (out of 50 runs) converged within % of the best-known solution using real-coded GAs with the proposed constraint
handling scheme on test problem 3
Mutation
Niching
Infeasible
Optimized f3~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
No
No
ÿ15:000
ÿ15:000
ÿ9:603
No
Yes
ÿ15:000
ÿ15:000
ÿ10:959
Yes
Yes
ÿ15:000
ÿ15:000
ÿ13:000
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

4.4. Test problem 4
This problem has eight variables and six inequality constraints [6]:
Minimize
f4~x  x1  x2  x3
Subject to
g1~x  1 ÿ 0:0025x4  x6 P 0;
g2~x  1 ÿ 0:0025x5  x7 ÿ x4 P 0;
g3~x  1 ÿ 0:01x8 ÿ x5 P 0;
g4~x  x1x6 ÿ 833:33252x4 ÿ 100x1  83333:333 P 0;
g5~x  x2x7 ÿ 1250x5 ÿ x2x4  1250x4 P 0;
g6~x  x3x8 ÿ x3x5  2500x5 ÿ 1 250 000 P 0;
100 6 x1 6 10 000;
1000 6 x2; x3 6 10 000;
10 6 xi 6 1000;
i  4; . . . ; 8:
10
Fig. 11. Average normalized Euclidean distance of feasible solutions for dierent real-coded GAs with the proposed constraint
handling scheme is plotted versus generation number on test problem 3.
Fig. 12. Average f ~x=f 
3 obtained by dierent real-coded GAs with the proposed constraint handling scheme is plotted versus
generation number on test problem 3.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

The optimum solution is
~x  579:3167; 1359:943; 5110:071; 182:0174; 295:5985; 217:9799; 286:4162; 395:5979;
f 
4  7049:330923:
All six constraints are active at this solution.
Table 5 shows the performance of GAs with dierent constraint handling methods. Michalewicz [6]
experienced that this problem is dicult to solve. Out of seven methods tried in that study, three found
solutions somewhat closer to the true optimum. The best solution obtained by any method used in that
study had an objective function value equal to 7377:976, which is about 4.66% worse than the true optimal
objective function value. A population size of 70 was used and ¯oating-point GAs with a number specialized crossover and mutation operators were run for 5000 generations, totaling 350 070 function evaluations. As mentioned earlier, in this study, we have used a dierent real-coded GA with SBX operator and
we have consistently found solutions very close to the true optimum with 80 080 function evaluations
(population size 80, maximum generations 1000). However, the best solution obtained by GAs with niching
and mutation and with a maximum of 320 080 function evaluations (population size 80, maximum generations 4000) has a function value equal to 7060:221, which is only about 0.15% more than the true optimal objective function value. Thus, GAs with the proposed constraint handling method has been able to
®nd better solutions than that found by any method used in [6]. Moreover, the median solution found in
GAs with niching and mutation is even better than the best solution found in [13].
Figs. 13and14showtheeectofniching ontheaverageEuclideandistanceamong feasible solutionsandthe
average proportion of feasible solutions in the population of 50 GA runs. The former ®gure shows that niching
Table 5
Number of runs (out of 50 runs) converged within % of the best-known solution using real-coded GAs with the proposed constraint
handling scheme on test problem 4
Mutation
Niching
Infeasible
Optimized f4~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
Maximum generation  1000
No
No
7063.377 8319.211 13738.276
No
Yes
7065.742 8274.830 10925.165
Maximum generation  4000
Yes
Yes
7060.221 7220.026 10230.834
Fig. 13. Average Euclidean distance of feasible solutions in 50 runs of real-coded GAs with the proposed constraint handling scheme
on test problem 4.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

helps to maintain diversity in the population. When mutation operator is added, the diversity among feasible
solutions is better and is maintained for longer generations. The latter ®gure shows that initially no solution
was feasible. With generations, more number of feasible solutions are continuously found. Since niching helps
to maintain diversity in feasible solutions, more feasible solutions are also found with generations.
4.5. Test problem 5
This problem has seven variables and four nonlinear constraints [6]:
Minimize
f5~x  x1 ÿ 102  5x2 ÿ 122  x4
3  3x4 ÿ 11:02  10x6
5  7x2
6  x4
7 ÿ 4x6x7
ÿ10x6 ÿ 8x7
Subject to
g1~x  127 ÿ 2x2
1 ÿ 3x4
2 ÿ x3 ÿ 4x2
4 ÿ 5x5 P 0;
g2~x  282 ÿ 7x1 ÿ 3x2 ÿ 10x2
3 ÿ x4  x5 P 0;
g3~x  196 ÿ 23x1 ÿ x2
2 ÿ 6x2
6  8x7 P 0;
g4~x  ÿ4x2
1 ÿ x2
2  3x1x2 ÿ 2x2
3 ÿ 5x6  11x7 P 0;
ÿ10 6 xi 6 10;
i  1; . . . ; 7:
11
The optimal solution is
~x  2:330499; 1:951372; ÿ0:4775414; 4:365726; ÿ0:6244870; 1:038131; 1:594227;
f 
5  680:6300573:
At this solution, constraints g1 and g4 are active. Michalewicz [6] reported that the feasible region for this
problem occupies only about 0.5% of the search space.
Table 6 presents the performance of GAs with the proposed constraint handling method with a population size of 10  7 or 70. In this problem also, niching seems to have done better. In the ®rst case, when
GAs are run without niching and mutation, all GA runs get stuck to a solution closer to the true optimum
solution at around 577 generations. Thus, increasing the generation number to 5000 does not alter GA's
performance. However, when niching is introduced among feasible solutions, diversity of solutions is
maintained and GAs with SBX operator can ®nd better solutions. For space restrictions, we do not present
generation-wise plots for this and subsequent test problems.
The best result reported in [6] is with penalty function approach in which the penalty parameters are
changed with generation. With a total of 350 070 function evaluations, the best, median, and worst obFig. 14. Average proportion of feasible solutions in the population obtained by 50 runs of real-coded GAs with the proposed constraint handling scheme on test problem 4.
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

jective function values of 10 runs were 680.642, 680.718, and 680.955, respectively. Table 6 shows that
50 GA runs with the proposed constrained handling method have found best, median, and worst solutions
as 680.634, 680.642, 680.651, respectively with an identical number of function evaluations. These solutions
are much closer to the true optimum solution than that found by the best algorithm in [6].
4.6. Test problem 6
This problem has ®ve variables and six inequality constraints [7,23]:
Minimize
f6~x  5:3578547x2
3  0:8356891x1x5  37:293239x1 ÿ 40792:141
Subject to
g1~x  85:334407  0:0056858x2x5  0:0006262x1x4 ÿ 0:0022053x3x5 P 0;
g2~x  85:334407  0:0056858x2x5  0:0006262x1x4 ÿ 0:0022053x3x5 6 92;
g3~x  80:51249  0:0071317x2x5  0:0029955x1x2  0:0021813x2
3 P 90;
g4~x  80:51249  0:0071317x2x5  0:0029955x1x2  0:0021813x2
3 6 110;
g5~x  9:300961  0:0047026x3x5  0:0012547x1x3  0:0019085x3x4 P 20;
g6~x  9:300961  0:0047026x3x5  0:0012547x1x3  0:0019085x3x4 6 25;
78 6 x1 6 102;
33 6 x2 6 45;
27 6 xi 6 45;
i  3; 4; 5:
12
The best-known optimum solution [23] is
~x  78:0; 33:0; 29:995; 45:0; 36:776;
f 
6  ÿ30665:5:
At this solution, constraints g2 and g5 are active. The best-known GA solution to this problem obtained
elsewhere [3] using a multi-level penalty function method is
~xGA  80:49; 35:07; 32:05; 40:33; 33:34;
f GA
 ÿ30005:7;
which is about 2.15% worse than the best-known optimum solution.
Table 7 presents the performance of GAs with the proposed constraint handling method with a population size 10  5 or 50. Once again, it is found that the presence of niching improves the performance of
GAs. When GAs are run longer, the solution improves in the presence of niching. GAs without niching and
mutation could not improve the solution much with more generations, but GAs with niching continuously
improve the solution with generations. The presence of niching and mutation ®nds the best solution. The
important aspect is that 47 of 50 runs have found solutions within 1% of the best-known solution. It is also
interesting to note that all GAs used here have found solutions better than that reported earlier [3], solved
using binary GAs with a multi-level penalty function method.
Table 6
Number of runs (out of 50 runs) converged within % of the best-known solution using GAs with the proposed constraint handling
scheme on test problem 5
Mutation
Niching
Optimized f5~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
Best
Median
Worst
Maximum generation  1000
No
No
680.800720
683.076843
705.861145
No
Yes
680.659424
681.525635
687.188599
Maximum generation  5000
No
Yes
680.660339
681.487427
684.845764
Yes
Yes
680.634460
680.641724
680.650879
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

4.7. Test problem 7
This problem has ®ve variables and three equality constraints [6]:
Minimize
f7~x  expx1x2x3x4x5;
Subject to
h1~x  x2
1  x2
2  x2
3  x2
4  x2
5  10;
h2~x  x2x3 ÿ 5x4x5  0;
h3~x  x3
1  x3
2  ÿ1;
ÿ2:3 6 xi 6 2:3;
i  1; 2;
ÿ3:2 6 xi 6 3:2;
i  3; 4; 5:
13
The optimal solution to this problem is as follows:
~x  ÿ1:717143; 1:595709; 1:827247; ÿ0:7636413; ÿ0:7636450;
f 
7  0:053950:
Equality constraints are handled by converting them as inequality constraints as d ÿ jhk~xj P 0 for all k,
as mentioned earlier. In this problem, d is set to 10ÿ3, in order to allow some room for the search algorithm
to work on. Table 8 shows the performance of GAs with a maximum of 350 050 function evaluations
(population size 50, maximum generations 7000). Although niching alone could not improve performance
much, along with mutation 19 out of 50 runs have found a solution within 1% of the optimal objective
function value.
4.8. Test problem 8
This problem has 10 variables and eight constraints [6]:
Minimize
f8~x  x2
1  x2
2  x1x2 ÿ 14x1 ÿ 16x2  x3 ÿ 102  4x4 ÿ 52  x5 ÿ 32
2x6 ÿ 12  5x2
7  7x8 ÿ 112  2x9 ÿ 102  x10 ÿ 72  45
Subject to
g1~x  105 ÿ 4x1 ÿ 5x2  3x7 ÿ 9x8 P 0;
g2~x  ÿ10x1  8x2  17x7 ÿ 2x8 P 0;
g3~x  8x1 ÿ 2x2 ÿ 5x9  2x10  12 P 0;
g4~x  ÿ3x1 ÿ 22 ÿ 4x2 ÿ 32 ÿ 2x2
3  7x4  120 P 0;
g5~x  ÿ5x2
1 ÿ 8x2 ÿ x3 ÿ 62  2x4  40 P 0;
g6~x  ÿx2
1 ÿ 2x2 ÿ 22  2x1x2 ÿ 14x5  6x6 P 0;
g7~x  ÿ0:5x1 ÿ 82 ÿ 2x2 ÿ 42 ÿ 3x2
5  x6  30 P 0;
g8~x  3x1 ÿ 6x2 ÿ 12x9 ÿ 82  7x10 P 0;
ÿ10 6 x1 6 10;
i  1; . . . ; 10:
14
Table 7
Number of runs (out of 50 runs) converged within % of the best-known solution using real-coded GAs with the proposed constraint
handling scheme on test problem 6
Mutation
Niching
Optimized f6~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
Best
Median
Worst
Maximum generation  1000
No
No
ÿ30614:814
ÿ30196:404
ÿ29606:451
No
Yes
ÿ30646:469
ÿ30279:744
ÿ29794:441
Maximum generation  5000
No
No
ÿ30614:814
ÿ30196:404
ÿ29606:596
No
Yes
ÿ30651:865
ÿ30376:906
ÿ29913:635
Yes
Yes
ÿ30665:537
ÿ30665:535
ÿ29846:654
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

The optimum solution to this problem is as follows:
~x  2:171996; 2:363683; 8:773926; 5:095984; 0:9906548; 1:430574; 1:321644; 9:828726; 8:280092;
8:375927;
f 
8  24:3062091:
The ®rst six constraints are active at this solution.
Table 9 shows the performance of GAs with the proposed constraint handling scheme with a population
size 10  10 or 100. In this problem, GAs with and without niching performed equally well. However, GA's
performance improves drastically with mutation, which provided the necessary diversity among the feasible
solutions. This problem was also solved by Michalewicz [6] by using dierent constraint handling techniques. The best reported method had its best, median, and worst objective function values as 24.690,
29.258, and 36.060, respectively, in 350 070 function evaluations. This was achieved with a multi-level
penalty function approach. With a similar maximum number of function evaluations, GAs with the proposed constraint handling method have found better solutions (best: 24.372, median: 24.409, and worst:
25.075). The best solution is within 0.27% of the optimal objective function value. Most interestingly, 41 out
of 50 runs have found a solution having objective function value within 1% (or f ~x smaller than 24.549) of
the optimal objective function value.
4.9. Welded beam design problem revisited
We shall now apply the proposed method to solve the welded beam design problem discussed earlier. GA
parameter values same as that used earlier are also used here. Table 10 presents the performance of GAs
with a population size 80. Real-coded GAs without niching is good enough to ®nd a solution within 2.6% of
the best objective function value. However, with the introduction of niching, 28 runs out of 50 runs have
found a solution within 1% of the optimal objective function value and this has been achieved with only a
maximum of 40 080 function evaluations. When more number of function evaluations are allowed, real
GAs with the proposed constraint handling technique and mutation operator perform much better ± all
50 runs have found a solution within 0.1% (to be exact) of the true optimal objective function value. This
means that with the proposed GAs, one run is enough to ®nd a satisfactory solution close to the true
Table 9
Number of runs (out of 50 runs) converged within % of the best-known solution using real-coded GAs with the proposed constraint
handling scheme on test problem 8
Mutation
Niching
Infeasible
Optimized f8~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
Maximum generation  1000
No
No
24.81711
27.85520
42.47685
No
Yes
24.87747
26.73401
50.40042
Maximum generation  3500
Yes
Yes
24.37248
24.40940
25.07530
Table 8
Number of runs (out of 50 runs) converged within % of the best-known solution using real-coded GAs with the proposed constraint
handling scheme on test problem 7
Mutation
Niching
Infeasible
Optimized f7~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
No
No
0.053950
0.365497
0.990920
No
Yes
0.053950
0.363742
0.847147
Yes
Yes
0.053950
0.241289
0.507761
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

optimal solution. In handling such complex constrained optimization problems, any user would like to use
such an ecient yet robust optimization algorithm.
When binary GAs (each variable is coded in 10 bits) with (or without) niching are applied, no solution
within 50% of the best-known solution is found. With niching on, the best, median, and worst objective
function values of optimized solutions are found to be 3.82098, 8.89996, and 14.29893, respectively. Clearly,
the real-coded GA implementation with SBX operator is better able to ®nd near-optimum solutions than
the binary GAs.
Fig. 15 shows the performance of various GAs in terms of ®nding a solution closer to the true optimum
solution. Average ratio of the best objective function value obtained by GAs to the best-known objective
function value of 50 GA runs is plotted with generation number. The ®gure shows that binary GAs prematurely converge to suboptimal solutions, whereas real-coded GAs with niching (and with or without
mutation) ®nd solutions very close to the true optimal solution.
4.10. Summary of results
Here, we summarize the best GA results obtained in this paper (Table 11) and compare that with the best
reported results in earlier studies. It is found here that the previously reported solution (marked with a ) of
test problem 2 is not the true optimum. The solution obtained here is better than this previously known best
solution. In all problems marked by a #, a better solution than that obtained by a previous GA implementation is obtained. In all other cases, the best solution of this study matches that of the previous GA
studies.
Fig. 15. Average f ~x=f 
w obtained by dierent GAs with the proposed constraint handling scheme is plotted versus generation number
on the welded beam design problem.
Table 10
Number of runs (out of 50 runs) converged within % of the best-known solution using binary GAs (TS-B) and real-coded GAs (TS-R)
with the proposed constraint handling scheme on the welded beam design problem
Method
Mutation
Niching
Optimized fw~x
6 1%
6 2%
6 5%
6 10%
6 20%
6 50%
> 50%
Best
Median
Worst
Maximum generations  500
TS-R
No
No
2.44271
3.83412
7.44425
TS-R
No
Yes
2.38119
2.39289
2.64583
Maximum generations  4000
TS-R
No
Yes
2.38119
2.39203
2.64583
TS-R
Yes
Yes
2.38145
2.38263
2.38355
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

For test problems 3±8, earlier methods recorded the best, median, and worst values for 10 GA runs only.
However, the corresponding values for GAs with the proposed method have been presented for 50 runs. In
some test problems, the worst GA solution (albeit a few isolated cases) has an objective function value away
from the true optimal solution. This is because reasonable values (but ®xed for all problems) of GA parameter values are used in this study. With a parametric study of important GA parameters (for example,
population size (here, 10 times the number of variables is used), gc for SBX operator (here, 1 is used), gm for
mutation operator (here, a linear variation from 1 to 100 is used)), the overall performance of GAs and the
worst GA solution can both be improved.
It is clear that in most cases the proposed constraint handling strategy has performed with more ef®-
ciency (in terms of getting closer to the best-known solution) and with more robustness (in terms of more
number of successful GA runs ®nding solutions close to the best-known solution) than previous methods.
5. Conclusions
The major diculty in handling constraints using penalty function methods in GAs and in classical
optimization methods has been to set appropriate values for penalty parameters. This often requires users
to experiment with dierent values of penalty parameters. In this paper, we have developed a constraint
handling method for GAs which does not require any penalty parameter. The need of a penalty parameter
arises in order to maintain the objective function value and the constraint violation values of the same
order. In the proposed method, solutions are never compared in terms of both objective function value and
constraint violation information. Thus, penalty parameters are not needed in the proposed approach.
Infeasible solutions are penalized in a way so as to provide a search direction towards the feasible region
and when adequate feasible solutions are found a niching scheme is used to maintain diversity. This aids
GA's crossover operator to ®nd better and better solutions with generation. All these have been possible
mainly because of the population approach of GAs and ability to have pair-wise comparison of solutions
using the tournament selection operator. It is important to note that the proposed constraint handling
approach is not suitable for classical point-by-point search methods. Thus, GAs or other evolutionary
computations methods have a niche over classical methods to handle constraints with the proposed
approach.
On a number of test problems including an engineering design problem, GAs with the proposed constraint handling method have repeatedly found solutions closer to the true optimal solutions than earlier
GAs. On one test problem, a solution better than that reported as the optimal solution earlier is also found.
It has also been observed that since all problems used in this study are de®ned in the real space and the
feasible regions are usually of arbitrary shape (convex or concave), the use of real-coded GAs with a
controlled search operator are more suited than binary GAs in ®nding feasible children solutions from
feasible parent solutions. In this respect, the use of real-coded GAs with SBX and a parameter-based
mutation operator have been found to be useful. It would be worthwhile to investigate how the proposed
constraint handling method would perform with binary GAs to problems having discrete variables.
Table 11
Summary of results of this study (a `±' indicates that information is not available)
Prob No.
True Optimum
Best-known GA
Results of this study
Best
Median
Worst
Best
Median
Worst
2#
ÿ1.905()
ÿ1.905()
ÿ1.915
ÿ1.915
ÿ1.915
ÿ15.000
ÿ15.000
ÿ15.000
ÿ15.000
ÿ15.000
ÿ15.000
ÿ13.000
4#
7049.331
7485.667
8271.292
8752.412
7060.221
7220.026
10230.834
5#
680.630
680.642
680.718
680.955
680.634
680.642
680.651
6#
ÿ30665.5
ÿ30005.7
ÿ30665.537
ÿ30665.535
ÿ29846.654
0.054
0.054
0.060
0.557
0.054
0.241
0.508
8#
24.306
24.690
29.258
36.060
24.372
24.409
25.075
Weld#
2.381
2.430
2.381
2.383
2.384
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

All problem-independent GA parameters are used in this study. In all test problems, reasonable values
for these GA parameters are used. It would be worthwhile to do a parametric study of important GA
parameters to improve the performance of GAs even further.
The results on the limited test problems studied here are interesting and show promise for a reliable and
ecient constrained optimization task through GAs.
Acknowledgements
The author greatly appreciates the programming help provided by couple of his students: Samir Agrawal
and Priya Rawat. Comments made by Zbigniew Michalewicz on an earlier version of the paper are highly
appreciated. Some portions of this study have been performed during the author's visit to the University of
Dortmund, Germany, for which the author acknowledges the support from Alexander von Humboldt
Foundation.
Appendix A. Simulated binary crossover and parameter-based mutation
The development of simulated binary crossover operator (SBX) and parameter-based mutation operator for handling ¯oating point numbers were performed in earlier studies [14,15]. Here, we simply present
the procedures for calculating children solutions from parent solutions under crossover and mutation
operators.
A.1. Simulated binary crossover (SBX) operator
The procedure of computing children solutions y1 and y2 from two parent solutions x1 and x2 are as
follows:
1. Create a random number u between 0 and 1.
2. Find a parameter b using a polynomial probability distribution, developed in [14] from a schema processing point of view, as follows:
b 
2u1=gc1
if u 6 0:5;
21ÿu
1=gc1
otherwise;
A:1
where gc is the distribution index for SBX and can take any nonnegative value. A small value of gc allows
solutions far away from parents to be created as children solutions and a large value restricts only nearparent solutions to be created as children solutions.
3. The children solutions are then calculated as follows:
y1  0:5
x1
ÿ
h
 x2
ÿ b x2

ÿ x1
i
y2  0:5
x1
ÿ
h
 x2
 b x2

ÿ x1
i
The above procedure is used for variables where no lower and upper bounds are speci®ed. Thus, the
children solutions can lie anywhere in the real space [ÿ1, 1] with varying probability. For calculating the
children solutions where lower and upper bounds (xl and xu) of a variable are speci®ed, Eq. (A.1) needs to
be changed as follows:
b 
au1=gc1
if u 6 1
a ;
2ÿau
ÿ
1=gc1
otherwise;
A:2
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

where a  2 ÿ bÿgc1 and b is calculated as follows:
b  1 
y2 ÿ y1 min
x1
ÿ
ÿ xl
xu
ÿ
ÿ x2
It is assumed here that x1 < x2. A simple modi®cation to the above equation can be made for
x1 > x2. The above procedure allows a zero probability of creating any children solution outside the
prescribed range [xl, xu]. It is intuitive that Eq. (A.2) reduces to Eq. (A.1) for xl  ÿ1 and xu  1.
For handling multiple variables, each variable is chosen with a probability 0.5 in this study and the
above SBX operator is applied variable-by-variable. This way about half of the variables get crossed over
under the SBX oparator. SBX operator can also be applied once on a line joining the two parents. In all
simulation results here, we have used gc  1.
A.2. Parameter-based mutation operator
A polynomial probability distribution is used to create a solution y in the vicinity of a parent solution x
[15]. The following procedure is used for variables where lower and upper boundaries are not speci®ed:
1. Create a random number u between 0 and 1.
2. Calculate the parameter d as follows:
d 
2u1=gm1 ÿ 1
if u 6 0:5;
1 ÿ 21 ÿ u1=gm1
otherwise;
A:3
where gm is the distribution index for mutation and takes any nonnegative value.
3. Calculate the mutated child as follows:
y  x  dDmax;
where Dmax is the maximum perturbance allowed in the parent solution.
For variables where lower and upper boundaries (xl and xu) are speci®ed, above equation may be
changed as follows:
d 
2u  1 ÿ 2u1 ÿ dgm1
h
i1=gm1
ÿ 1
if u 6 0:5;
1 ÿ 21 ÿ u  2u ÿ 0:51 ÿ dgm1
h
i1=gm1
otherwise;
A:4
where d  minx ÿ xl; xu ÿ x=xu ÿ xl. This ensures that no solution would be created outside the
range [xl, xu]. In this case, we set Dmax  xu ÿ xl. Eq. (A.4) reduces to Eq. (A.3) for xl  ÿ1 and xu  1.
Using above equations, we can calculate the expected normalized perturbance (y ÿ x=xu ÿ xl) of the
mutated solutions in both positive and negative sides separately. We observe that this value is O1=gm.
Thus, in order to get a mutation eect of 1% perturbance in solutions, we should set gm  100. In all our
simulations wherever mutation is used, we set gm  100  t and the probability of mutation is changed as
follows:
pm  1
n 
t
tmax
ÿ 1
n
where t and tmax are current generation number and the maximum number of generations allowed, respectively. Thus, in the initial generation, we mutate on an average one variable (pm  1=n) with an expected 1% perturbance and as generations proceed, we mutate more variables with lesser expected
perturbance. This setting of the mutation operator is arbitrarily chosen and has found to have worked well
in all problems tried in this paper. No effort is spent in tuning these parameters for obtaining better results.
Appendix B. Terms and parameters used in test function 2
The following terms are required to compute the objective function and constraints for the test problem
2 [23,24]:
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338

y1~x  x1  x2  41:6;
c1~x  0:024x4 ÿ 4:62;
y2~x  12:5=c1~x  12:0;
c2~x  0:0003535x1x1  0:5311x1  0:08705y2~xx1;
c3~x  0:052x1  78:0  0:002377y2~xx1;
y3~x  c2~x=c3~x;
y4~x  19:0y3~x;
c4~x  0:04782x1 ÿ y3~x  0:1956x1 ÿ y3~x2=x2  0:6376y4~x  1:594y3~x;
c5~x  100:0x2;
c6~x  x1 ÿ y3~x ÿ y4~x;
c7~x  0:95 ÿ c4~x=c5~x;
y5~x  c6~xc7~x;
y6~x  x1 ÿ y5~x ÿ y4~x ÿ y3~x;
c8~x  0:995y4~x  y5~x;
y7~x  c8~x=y1~x;
y8~x  c8~x=3798:0;
c9~x  y7~x ÿ 0:0663y7~x=y8~x ÿ 0:3153;
y9~x  96:82=c9~x  0:321y1~x;
y10~x  1:29y5~x  1:258y4~x  2:29y3~x  1:71y6~x;
y11~x  1:71x1 ÿ 0:452y4~x  0:58y3~x;
c10~x  12:3=752:3;
c11~x  1:75y2~x0:995x1;
c12~x  0:995y10~x  1998:0;
y12~x  c10~xx1  c11~x=c12~x;
y13~x  c12~x ÿ 1:75y2~x;
y14~x  3623:0  64:4x2  58:4x3  146312:0=y9~x  x5;
c13~x  0:995y10~x  60:8x2  48:0x4 ÿ 0:1121y14~x ÿ 5095:0;
y15~x  y13~x=c13~x;
y16~x  148000:0 ÿ 331000:0y15~x  40y13~x ÿ 61:0y15~xy13~x;
c14~x  2324:0y10~x ÿ 28740000:0y2~x;
y17~x  14130000:0 ÿ 1328:0y10~x ÿ 531:0y11~x  c14~x=c12~x;
c15~x  y13~x=y15~x ÿ y13~x=0:52;
c16~x  1:104 ÿ 0:72y15~x;
c17~x  y9~x  x5:
The values of ai and bi for i  1; . . . ; 18 are as follows:
ai  f0; 0; 17:505; 11:275; 214:228; 7:458; 0:961; 1:612; 0:146; 107:99; 922:693; 926:832;
18:766; 1072:163; 8961:448; 0:063; 71084:33; 2802713:0g;
bi  f0; 0; 1053:6667; 35:03; 665:585; 584:463; 265:916; 7:046; 0:222; 273:366; 1286:105;
1444:046; 537:141; 3247:039; 26844:086; 0:386; 140000:0; 12146108:0g:
K. Deb / Comput. Methods Appl. Mech. Engrg. 186 (2000) 311±338
