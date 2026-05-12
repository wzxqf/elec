# Liu和Wu - 2007 - Portfolio optimization in electricity markets

## Metadata

- source_pdf: 参考文献/Liu和Wu - 2007 - Portfolio optimization in electricity markets.pdf
- extraction_method: pymupdf
- extraction_status: success
- title: 
- doi: 

## Abstract

In a competitive electricity market, Generation companies (Gencos) face price risk and delivery risk that affect their proﬁtability. Risk management
is an important and essential part in the Genco’s decision making. In this paper, risk management through diversiﬁcation is considered. The problem
of energy allocation between spot markets and bilateral contracts is formulated as a general portfolio optimization problem with a risk-free asset
and n risky assets. Historical data of the PJM electricity market are used to demonstrate the approach.
© 2006 Elsevier B.V. All rights reserved.

## Body

1. Introduction
Deregulation in the electricity industry has introduced competitive markets. Generation companies (Gencos) no longer
enjoy guaranteed rate of return as in the old regulated environment. The price of electricity Gencos receive in a competitive
market depends on many factors: bidding prices of all market
participants, load demand, unit outages, etc. It is uncertain and
volatile. There is usually more than one market for a Genco
to enter. Gencos are faced with the prospect of making more
proﬁt or the risk of losing money. The scheduling decisions of
Gencos are important in determining their proﬁtability. Recognizing market risk and management of such risks are essential
for Gencos in a competitive market.
Risk refers to the possibility of suffering harm or loss; danger or hazard. Risks result from uncertainty. However, there is a
difference between risk and uncertainty: risk is something that
usually can be controlled whereas uncertainty is beyond anybody’s control. In the electricity market, the proﬁts of Gencos are
inﬂuenced by many uncertain factors: unit outage, other genco’s
bidding strategy, congestion in transmission, demand change,
etc. These uncertainties bring about risks in electricity pricing
and delivery. Risks of spot price volatility in electricity markets
are especially signiﬁcant. Operating data have shown that daily
∗Corresponding author. Tel.: +86 851 4732915; fax: +86 851 4730394.
E-mail address: minliu@graduate.hku.hk (M. Liu).
spot price volatility in electricity is much higher than that of any
other commodity. The main reason for this may be attributed to
the particular characteristic of non-storability of electricity.
Risk management is the process of achieving a desired
return/proﬁt, taking into considerations of risks, through a particular strategy. In the ﬁnancial ﬁeld, there are two means to
control risk. One is through risk ﬁnancing by using hedging to
offset losses that can occur and the other is through risk reduction
using diversiﬁcation to reduce exposure to risks. Instruments for
risk management include forward contracts, futures contracts,
options, etc. Forward contracts are agreements to buy/sell an
agreed amount of the commodity at a speciﬁed price at a designated time. Futures contracts are standardized forward contracts
that are traded on exchange and no physical delivery is necessary. Options are contracts that provide the holder the right but
not the obligation to buy/sell the commodity at a designated time
at a speciﬁed price. Hedging is to use these ﬁnancial instruments
with the payoff patterns to offset the market risks. Diversiﬁcation
is to engage in a wide variety of markets so that the exposure to
the risk of any particular market is limited. Applying this concept
to energy trading in an electricity market, diversiﬁcation means
to trade energy through different physical trading approaches.1
In the energy market, both physical trading approach (e.g., spot
1 Physical trading approach refers to the trading approach in which actual
physical energy are traded while ﬁnancial trading approach only involves ﬁnancial settlement, no actual physical energy are traded through ﬁnancial trading.
0378-7796/$ – see front matter © 2006 Elsevier B.V. All rights reserved.
doi:10.1016/j.epsr.2006.08.025

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
market, contract market) and ﬁnancial trading approach (e.g.,
futures contracts, options, swaps, etc.) are available. A combination of these trading approaches is deﬁned as a portfolio and
the corresponding risk-control methodology is called portfolio
optimization.
A commonly adopted measure for risk assessment, i.e.,
assessing risk exposure of ﬁnancial portfolios, is the Value at
Risk (VaR), which is the monetary value that the portfolio will
lose less than that amount over a speciﬁed period of time with a
speciﬁed probability.
Various aspects of risk management have been applied to
the electricity market. Different forward contracts that can provide hedging to the risk of spot prices for market participants
are proposed [1–4]. The usefulness of the application of futures
contracts in an electricity market is demonstrated in [5–8]. Valuation of different contracts is considered in [9–11]. Monte Carlo
simulation and decision analysis have been applied to ﬁnd the
optimal contract combination [12–15]. Various issues related to
the combined spot/bilateral-contract dispatch are investigated in
[16–18]. VaR has been applied to risk assessment in electricity
markets [19–22]. Concepts from ﬁnancial option theory have
been utilized in the valuation of generation assets [23–25].
We are addressing the problem of trading scheduling for a
Genco, i.e., to optimally use both physical trading approaches
and ﬁnancial trading approaches to maximize its proﬁt potential,
taking into consideration the associated risk factors. It involves
the optimal allocation of the Genco’s output energy among multiple markets (e.g., spot market, contract market, futures market,
etc.) with the objective of maximizing its beneﬁt and minimizing the corresponding risk. We apply the approaches of portfolio optimization in Modern Portfolio Theory (MPT) [26] to
the problem. The method explicitly considers decision-makers’
risk aversion and the statistical correlation among alternative
outcomes. Although MPT is widely known in the ﬁnancial literature, its application in electricity markets might be of interest.
The reason is that electricity contracts have different risk characteristics under different electricity markets with different pricing
systems, which is due to the congestion in transmission. It is
further explained in Section 2 through the introduction of trading environment in electricity markets. Only price risk, delivery
risk and physical trading approaches are considered in this paper.
Price risk due to spot market ﬂuctuations and delivery risk due to
transmission congestion are related to power system operation.
In terms of applications, an electricity spot market that adopts
uniform marginal pricing scheme displays only price risk and
a spot market that adopts locational marginal pricing or zonal
pricing displays not only price risk but also delivery risk. In the
following, Section 2 introduces the background of the electricity
market with different pricing system which includes the trading
environment and associated risks. Section 3 describes the basic
theory and methodology to portfolio optimization which can be
applied to electricity markets with different pricing system. Section 4 develops an approach to energy allocation among physical
trading approaches, i.e., spot market and contract market. Example demonstrates the proposed energy allocation method using
historical data of the PJM market. Finally, Section 5 concludes
the paper.
2. Electricity markets
Most electricity markets provide two types of markets in
which energy is traded: the spot market and the (physical) forward market [27]. In the contract market, Gencos trade energy
by way of signing contacts, which are referred to as physical
forward contracts, with their counterparters (e.g., energy consumers). Speciﬁc details such as trading quantity (MW), trading
duration (h), trading price ($/MWh) and delivery point are bilaterally negotiated between Gencos and consumers or their agents.
Bilateral contracts are signed before the actual trading period.
In other words, trading quantity and price are set in advance.
Physical forwards can be traded on an exchange or in a bilateral
manner through over the counter (OTC)2 transactions.
As for the spot market, in this paper, we adopt FERC deﬁnitioninitsstandardmarketdesign[28]thatalltheenergytradedin
the real-time and day-ahead market as spot energy. The common
ground among these markets is that they all involve a centralized auction mechanism, by ISO, RTO or any such organization,
to determine which generation units should be deployed and
how much energy each selected unit should produce to meet the
demand. From a Genco’s point of view, selling energy in the
spot market means to submit a bid (price and quantity) to the
exchange (Power Pool/ISO) and get either of the two alternative results: (1) the exchange accepts the Genco’s bid and pays
the Genco the market clearing price (MCP) for its actual energy
output; or (2) the exchange rejects the Genco’s bid, i.e., the
Genco sells nothing in the spot market. The MCP depends on
everybody’s bids, as well as the load demand, and is therefore
uncertain. Three types of pricing systems have been adopted in
the spot market: uniform marginal pricing, zonal pricing and
nodal pricing (or locational marginal pricing (LMP)).
Uniform marginal pricing was adopted in the earlier England and Wales market and is followed by many markets around
the World. In such a market, only one energy price is used for
ex post settlement for each trading interval. Uniform marginal
pricing makes it easier to achieve market liquidity. A Genco, on
the other hand, can make certain of its revenue by signing bilateral contracts with its customers at ﬁxed energy prices. Hence,
bilateral contracts can be considered as risk-free transactions if
the Genco’s production cost is assumed deterministic.3 Trading
schedule in an electricity market with uniform pricing is to optimally allocate energy between risky spot market and risk-free
contract market.
In a zonal pricing system, an interconnected power system
is partitioned into pre-deﬁned geographical areas, called zones,
based on the knowledge that limitations in transmission exist
between these zones.4 When there is no congestion, one uni-
2 OTC is a kind of derivatives market in which non-standard products (e.g.,
contracts) are traded. Trades on the OTC market are negotiated directly through
dealers.
3 Only fuel-ﬁred plants are considered and fuel prices are assumed deterministic in order to focus on the risk of electricity prices in this paper.
4 Transmission system has operating constraints that limit the maximum
amount of power that are allowed to ﬂow through transmission lines. The limit
is set either by conductor thermal loading limit or by system stability consid-

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
form MCP is used throughout the system. When congestion
occurs, the power system is split into two zones with congested
lines in between and each zone has its own MCP, called zonal
prices. Transmission congestion limits the amount of generation
on the cheaper MCP to supply the demand that pays more on the
more expensive MCP side. The product of the price difference
between the zones involved and the trading amount is called the
congestion charge, which has to be added to the trading cost for
the bilateral trading. Congestion and the resulting zonal prices
are uncertain and unpredictable, which makes inter-zonal bilateral contracts risky. Only intra-zonal contracts in such a market
are risk-free. Zonal pricing was used in pre-2002 California market and is still used in Nordic and some other markets.
Locational marginal (or nodal) price, as the name implies, is
the marginal price at each location or node of a power system.
LMPs are obtained when nodal power balance equations (power
ﬂow equations), as well as transmission line loading limits,
are explicitly incorporated in the welfare maximization problem (optimal power ﬂow or OPF). When there is no congestion,
there is one price in the interconnected system if transmission
losses are ignored. When any one of the transmission line is
congested, the marginal energy prices will vary from locations.
LMP is recommended by FERC in its standard market design
[28] and adopted by many RTOs and ISOs [29–32]. Same as
in zonal pricing, the difference between locational prices represents congestion charges to the bilateral trading. Since demand,
generation pattern and transmission loading vary over time, network congestion, locational prices, and congestion charges also
vary over time and are uncertain. Except contract with customers
at the same location, all contracts are risky. Note that from theoretical point of view, methodology for energy trading scheduling
under LMP and zonal pricing is the same. Our presentation will
be stated in terms of LMP, knowing that it is equally applicable
to zonal pricing. Besides, trading schedule in a uniform pricing
market is just a special example in the LMP market. A market
with LMP is therefore supposed general in this paper.
3. Portfolio optimization
3.1. Modern Portfolio Theory
The Modern Portfolio Theory [26] is principles underlying analysis and evaluation of rational portfolio choices based
on risk-return trade-offs and efﬁcient diversiﬁcation. In other
words, MPT is an approach to measuring the risk of an asset,
quantifying trade-off between risk and expected return, and
ﬁnally forming an optimal portfolio of assets. Markowitz is
the father of MPT. His original article [33] and book [34] on
the subject clearly depicted, for the ﬁrst time, Modern Portfolio Theory. The book was ﬁlled with insights and suggestions
that anticipated many of the subsequent developments in the
erations. When transmission constraints limit power generation from otherwise
economic considerations, we say the transmission system is congested. Two
types of locational pricing systems have been adopted in electricity markets for
congestion management: zonal pricing and nodal pricing (or locational marginal
pricing (LMP)).
ﬁeld. The important message of the theory was that asset could
not be selected only on characteristics that were unique to the
security. Rather, an investor had to consider how each security
co-moved with all other securities. Furthermore, taking these
co-movements into account resulted in an ability to construct a
portfolio that had the same expected return and less risk than
a portfolio constructed by ignoring the interactions between
securities. Portfolio theory is a well-developed paradigm. There
are excellent textbooks such as [26] and [35] on this subject.
There are also good reviews in more advanced texts such as
[36]. Finally, there are good review articles such as [37].
In the following, we formulate the general portfolio optimization problem as a quadratic programming problem. We also give
a brief tutorial treatment of the standard approach in ﬁnancial
theory.
3.2. Portfolios of assets
A portfolio of assets is a combination of all potential assets.
Let indexes 1–n denote risky assets, while index n + 1 denotes
the risk-free asset. All risk-free assets can be accumulated into
one risk-free asset. Given each asset’s rate of return (return for
short), ri (i = 1–n + 1), the portfolio’s return (denoted by rC)
is the weighted average of the component asset return with
the investment proportions as weights (denoted by wi), i.e.,
rC = n+1
i=1 wiri. The portfolio consists of risky assets and is also
risky. In other words, its return (rC) is uncertain. Let us assume
that we do have some knowledge of the situation in terms of
the probability distribution of the outcomes. The mean of the
probability distribution of the return, or the expected return, is
an indication of the expected proﬁtability. The variance of the
distribution indicates how wide spread is the possible outcomes
around the mean. The larger is the variance, the more uncertain
is the outcome. Therefore, the variance or the standard deviation of the distribution can be used as an indication of the risk
involved. This is the basis of the mean-variance criterion (MVC)
used by Markowitz and Tobin to develop the modern theory of
investment choice under uncertainty.
The expected return (E(rC)) and its variance (σ2(rC)) can be
expressed as follows:
E(rC) =
n+1
i=1
wiE(ri)
(1)
σ2(rC) =
n+1
i=1
n+1
j=1
wiwjσij =
n+1
i=1
w2
i σ2
i +
j̸=i
wiwjσij
(2)
where n+1
i=1wi = 1, wi ≥0, σ2
i is the variance of the return on
the ith asset; σij is the covariance between the returns on the ith
asset and the jth asset. The covariance σij measures how many
the returns on two assets move in tandem.
3.3. Portfolio selection
According to the MVC, a decision that results in higher
expected return and lower risk would be preferred. In other

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
words, a decision-maker’s objective is to maximize the expected
return and minimize the variance of the return. Combining the
two, we can deﬁne the objective function or the utility function
of a decision-maker in terms of the expected return E(r) and
variance of returns σ2(r) as follows [26]:
U = E(r) −1
2Aσ2(r)
(3)
where U is the utility function and A the weighting factor that
reﬂects the decision maker’s preference or aversion of risk. Positive A indicates a person is risk aversion, negative A indicates
risk loving, and A = 0 indicates the person is risk neutral. The
larger is the value of A, the more risk aversion is the person. The
determination of the exact value of the weighting factor A is typically the most difﬁcult part of any theory trying to combine two
objectives together. A broad range of studies, taking into account
the full range of available assets, places the degree of risk aversion for the representative investor in the range of 2.0–4.0 [38].
In the ﬁnancial text book [26], A = 3 is used as average risk aversion, and consequently, A > 3 for more risk aversion and A < 3
for less risk aversion.
Using the above utility function, the optimal portfolio is
obtained by maximizing the utility function with respect to the
weights of assets:
Max
wi U = E(rC) −1
2Aσ2(rC)
s.t.
n+1
i=1
wi = 1
wi ≥0
(4)
where E(rC) and σ2(rC) are given by (1) and (2), respectively.
The solution ¯wi(i = 1–n + 1) gives optimal allocation of assets
into a risk-free asset ¯wn+1 and n risky assets ¯wi(i = 1–n). The
optimal portfolio selection problem (4) is a quadratic programming problem. It, of course, can be solved directly using a
standard quadratic programming solution algorithm [39] or a
software package such as one available in Matlab. However,
more insights can be gained and a graphical interpretation can
be obtained if we solve it in two steps, following human intuition in solving this problem. Most individuals would divide the
problem into two sub problems. They would ﬁrst decide on what
risky portfolio to invest, i.e., the allocation of risky investment
into an optimal portfolio to diversify its risk. They would then
consider the problem of how much money to take out from the
bank to invest in risky investments, assuming money in the bank
is considered risk-free.
In the Appendix A, we show mathematically that the solution
of the portfolio optimization problem (4) is equivalent to the
solutions of the following two optimization problems: one with
n risky assets (5) and the other with one risk-free asset and one
risky asset (6). The two-step approach is the standard approach
followed in Portfolio Theory and is usually intuitively presented.
We have not found a rigorous mathematical justiﬁcation of such
intuitive arguments as we proved in the Appendix A.
Step 1: Optimal risky portfolio
Max
wi s = E(rP) −rB
σ(rP)
s.t.
n
i=1
wi = 1
wi ≥0
(5)
where
E(rP) =
n
i=1
wiE(ri)
σ(rP) =
n
i=1
w2
i σ2
i +
i̸=j
wiwjσij
1/2
The solution method to this optimization problem (5) can be
found in [35]. Let the solution to the optimal risky portfolio
problem be denoted by w∗
i (i = 1–n).
Step 2: Optimal allocation between a risk-free and a risky
investment
Max
y
U(y) = E(rC) −1
2Aσ2(rC)
(6)
where y is the investment proportion allocated to the risky asset,
and
E(rC) = (1 −y)rB + yE(r∗
P)
σ2(rC) = y2σ2(r∗
p)
Setting the derivative of this expression to zero and solving for
y yields the optimal position for risk-averse investors in the
risky asset, y*, as follows:
y∗= E(r∗
P) −rB
Aσ2(r∗
P)
(7)
where
E(r∗
P) =
n
i=1
w∗
i E(ri)
σ2(r∗
P) =
n
i=1
n
j=1
w∗
i w∗
jσij.
Combining the solutions to the sub problems (5) and (6),
we obtain the solution to the optimal portfolio selection (4) as
follows:
Risk-free asset, ¯wn+1 = 1 −y∗;
Risky assets, ¯wi = y∗w∗
i
(i = 1–n).
3.4. Graphical interpretation
The two-step solution has an intuitive graphical interpretation. Let us consider the optimal risky portfolio problem (5). The
feasible set on the mean-standard deviation (E(r) – σ) diagram

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
is the points corresponding to all combinations of wi(i = 1–n)
that satisfy:
E(rP) =
n
i=1
wiE(ri)
σ(rP) =
n
i=1
w2
i σ2
i +
i̸=j
wiwjσij
1/2
n
i=1
wi = 1
wi ≥0
where E(ri)(i = 1–n) and σij(i, j = 1–n) are given.
It can be shown that the feasible set is connected and convex
to the left [36], as showed in Fig. 1. Given an expected return
value, a lowest portfolio variance can be attained through optimally selecting the weights of risky-assets. In the mean-standard
deviation graph, given an expected return value, we can always
ﬁnd a portfolio with the given expected return and the lowest
variance. All the portfolio points with such an attribute form a
frontier. This frontier is called the minimum-variance frontier
(showed in Fig. 1). The bottom part of the minimum-variance
frontier is inefﬁcient since for any portfolio in the lower portion of the minimum-variance frontier, there is a portfolio with
the same standard deviation and a greater expected return positioned directly above it. Hence all the portfolios, which lie on the
minimum-variance frontier from the minimum-variance portfolio and upward, provide the best risk-return combinations and
thus are candidates for the optimal portfolio. These portfolios
form the so called efﬁcient frontier (see Fig. 1).
Assume that there is a risk-free asset denoted by B, portfolio
selection can be achieved with two steps. The ﬁrst step is to ﬁnd
the optimal risky portfolio. Any point in the efﬁcient frontier
represents a risky portfolio. A straight line connecting risk-free
asset B and any risky portfolio in the efﬁcient frontier is called
Capital Allocation Line (CAL) (see Fig. 2). The slop of the
CAL is the reward-to-risk ratio, a higher reward-to-risk ratio is
preferred by the investor. Graphically, the optimal risky portfolio
P, is the point where CAL is tangent to the efﬁcient frontier since
this CAL has the highest reward-to-risk ratio, i.e., the steepest
Fig. 1. Efﬁcient frontier.
Fig. 2. Portfolio selection with a risk-free asset.
slope. This is precisely the Optimal Risky Portfolio Problem
(5).
The second step of the portfolio selection is to allocate budget
between the risk-free asset B and the optimal risky portfolio P.
Graphically, the optimal combination is the point where CAL
touches the highest value of U, i.e., it is tangent to the constant
U curve, or the indifference curve, as showed in Fig. 2.
The presence of a risk-free asset makes it possible to solve
the overall optimization problems (4) in two-steps (5) and (6).
A remarkable implication of the two-step solution is that there
is a separation principle. Note that the optimal risky portfolio
selection problem (5) is independent of the decision-maker’s
risk preference A. In other words, the optimal risky portfolio on
the efﬁcient frontier can be determined once there is a reference
point of a risk-free asset and this optimal portfolio is the same
for all decision makers regardless of their risk preferences. Risk
preference only comes in at the second step to decide on the
proportion of the risk-free asset and this optimal risky portfolio.
In the absence of a risk-free asset, the determination of which
point on the efﬁcient frontier is optimal has to rely on a utility
functionsuchasin(4)thatincorporatesthedecision-maker’srisk
preference. The optimization problem (4) can indeed be applied
to solve the optimal portfolio selection without a risk-free asset.
4. Energy allocation between spot market and contract
market
We now apply the portfolio optimization method presented
in Section 3 to the energy allocation between spot market and
contract market when the market adopts a LMP pricing scheme.
Assume that a Genco trades energy through both spot market
and bilateral contract market. For bilateral contracts, if there is
congestion in the transmission system, the Genco will have to
pay all or part of the congestion charges which depends on the
speciﬁc market rules or on the negotiation between the Genco
and its consumers. The congestion charge between any two locations is the product of the spot price difference between these two
locations and the transmitted energy (quantity in MWh). Locational spot prices are uncertain and ﬂuctuate. In such a case, only
local contracts signed with local customers are risk-free trades,
non-local contracts signed with non-local customers are risky
trades due to the uncertainty in congestion charges. Therefore,

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
there are three types of trading approaches for a Genco: riskfree (local) contracts, risky (non-local) contracts and risky spot
market.
Assume that there are n areas or pricing nodes in an electricity market. A Genco is located in Area 1; other areas are
labeled from Area 2 to Area n. To simplify, suppose that the
Genco could sign one bilateral contract with each area’s customersatﬁxedenergyprice.Hence,theGencohasn + 1potential
transactions during the planning period,5 i.e., one risk-free localbilateral contract, n −1 risky non-local bilateral contracts and
one risky transaction traded in the spot market. The question
is how to allocate energy among these potential transactions in
order to maximize proﬁts with relatively low risk. Energy allocation without and with local bilateral contract will be considered
respectively in the following.
4.1. Energy allocation without local bilateral contract
Without local bilateral contract, i.e., the risk-free option, a
trading portfolio consists of only risky options and is therefore called a risky portfolio. The optimal risky portfolio can
be achieved by solving optimization problem (4).
4.2. Energy allocation with local bilateral contract
If local bilateral contract (i.e., risk-free transaction) is available, the optimal trading portfolio can be obtained by solving
optimization problem (4) directly or by solving problem (5) and
(6) step-by-step.
Let us now specify the return characteristics of all trades
(i.e., expected returns (E(ri)), variances (σ2
i ) and covariances
(σij)). Assume that the Genco has a quadratic cost-curve,
c(p,t,λF) = (a + bp + cp2)tλF, where p is the output power (MW);
t the trading time of each trading interval (h); λF the fuel price
($/MBtu) which is assumed given during the contract period; a,
b, c are fuel consumption coefﬁcients. There are M trading intervals in the trading period (i.e., contract period). The following
notation will be used:
Cov: covariance,
E: expectation,
i: the index of the trading area or pricing node,
k: the index of the trading interval,
rB: return on the local contract,
ri: return on the ith trade, i = 2–N denotes non-local bilateral
contract signed with the ith Area’s customers; i = 1 denotes the
transaction traded in the spot market. Expectation and variance
of ri are denoted by E(ri) and σ2
i , respectively.
λB
i,k: the kth trading interval’s contract price signed with customers of Area i.
λF
k : the kth trading interval’s fuel price,
5 The planning period could be one day, one week, one month, one year or
several years, etc.
λS
i,k: the kth trading interval’s spot price of Area i, the corresponding expectation is denoted by E(λS
i,k).
4.2.1. Return characteristics of local contract
According to the deﬁnition of return, return = (revenue
−cost)/cost, if all the energy are traded through the local
contract,6 the corresponding return during the contract period
is:
rB =
M
k=1pktλB
1,k −M
k=1(a + bpk + cp2
k)tλF
k
M
k=1(a + bpk + cp2
k)tλF
k
= K
M
k=1
pktλB
1,k −1
(8)
where K = 1/M
k=1(a + bpk + cp2
k)tλF
k . λB
1,k and λF
k
are
assumed certain when the Genco makes trading decisions, then
E(rB) = rB, σ2(rB) = 0, i.e., local bilateral contract is a risk-free
transaction.
Similarly, return characteristics of non-local contracts and
spot markets can be expressed as follows.
4.2.2. Return characteristics of non-local contracts
The congestion charge is accounted when calculating the
actual revenue on the non-local bilateral transaction. Generally,
congestion charges should be paid by the associated bilateral
transaction. But who (Gencos or energy purchasers) should
pay how many percentage of the involved congestion charges
depends on the speciﬁc market rules. That is, from a Genco’s
point of view, its congestion charge is between zero and the complete congestion charge of the associated bilateral transaction. In
this paper, a factor β (0 ≤β ≤1), which is decided by a speciﬁc
electricity market, is used to denote the payment proportion of
the Genco. Then we have:
E(ri) = K
M
k=1
pkt
λB
i,k −β(E(λS
i,k) −E(λS
1,k))

−1
(i = 2–N)
(9)
σ2
i = K2
M
k=1
(pktβ)2 
σ2(λS
1,k) + σ2(λS
i,k) −2Cov(λS
1,k, λS
i,k)

(i = 2–N)
(10)
4.2.3. Return characteristics of spot markets
E(r1) = K
M
k=1
pkt · E(λS
1,k) −1
(11)
σ2
1 = K2
M
k=1
(pkt)2σ2(λS
1,k)
(12)
6 Scheduled trading quantity of the energy of each transaction is the product
of the corresponding allocation ration wi and total energy (pt).

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
4.2.4. Correlations between risky trades
σij = Cov(ri, rj)
= K2
M
k=1
(pktβ)2

σ2(λS
1,k) −Cov(λS
1,k, λS
i,k)
−Cov(λS
1,k, λS
j,k) + Cov(λS
i,k, λS
j,k)

(i, j = 2–N)
(13)
σ1i = Cov(r1, ri) = K2
M
k=1
(pkt)2β
σ2(λS
1,k) −Cov(λS
1,k, λS
i,k)

(i = 2–N)
(14)
E(λS
i,k), σ2(λS
i,k) and Cov(λS
i,k, λS
j,k) should be estimated
before applying above equations. The estimation of these data
is an applied area of research and is not the focus of this paper.
In this paper, they are simply estimated based on historical data
according to the statistical method.
Strictly speaking, when congestion occurs, the spot prices at
various locations at that particular moment are related (through
physical laws, i.e., Kirchoff laws and Ohms law, in the OPF).
Over a period of time, however, such dependency might be captured statistically by the correlation coefﬁcients or covariances
of spot prices.
Example.
We use PJM market data to illustrate the energy
allocation methodology introduced in the last Section. In this
example, three areas (pricing nodes) in an electricity market are
considered. A Genco is located in Area 1. The Genco has four
trading choices, i.e., one trading in the spot market, one bilateral
contract signed with local customers (denoted by Contract 1),
one bilateral contract signed with customers of Area 2 (denoted
by Contract 2) and one bilateral contract signed with customers
of Area 3 (denoted by Contract 3).
The trading period is one month and the trading time of
each trading interval is one day or 24 h. The Genco’s unit characteristics are: a = 313.9102 MBtu/h, b = 7.6126 MBtu/MWh,
c = 0.00199 MBtu/MW2h, p = 400 MW. The historical data of
daily spot price of Area 1, Area 2 and Area 3 are calculated
based on the hourly price of month August from 1998 to 2005
in PENELEC, PEPCO and PECO, respectively (these data are
available on the website of PJM [29]). Based on historical data,
E(λS
i,k), σ2(λS
i,k) and Cov(λS
i,k, λS
j,k) (i = 1–3, k = 1–31) can be
calculated with statistical method. Daily spot prices statistics of
these three areas are thus obtained:
E(λ1) = 40.54 $/MWh,
σ(λ1) = 26.96 $/MWh (66.51%),
E(λ2) = 46.15 $/MWh,
σ(λ2) = 33.42 $/MWh (72.40%),
E(λ3) = 46.44 $/MWh,
σ(λ3) = 39.69 $/MWh (85.46%).
Let pi = 400 MW, λF
i = 3.0 $/MBtu, β = 1. Two cases are
considered to demonstrate the impact of the risk aversion of the
Genco and contract prices on the energy allocation or trading
schedule.
Table 1
Optimal energy allocation ratios for Case 1
Risk
aversion (A)
Energy allocation ratios
Utility
value
Contract 1
Contract 2
Contract 3
Spot market
0.1460
0.2148
0.1245
0.5147
0.4336
0.4306
0.1432
0.0830
0.3432
0.4271
0.5730
0.1074
0.0622
0.2574
0.4239
Table 2
Optimal energy allocation ratios for Case 2
Risk
aversion (A)
Energy allocation ratios
Utility
value
Contract 1
Contract 2
Contract 3
Spot market
0.0271
0.3598
0.1864
0.4267
0.4446
0.3513
0.2399
0.1243
0.2845
0.4405
0.5136
0.1799
0.0932
0.2133
0.4384
Case 1.
Suppose that λB
1 = 39, λB
2 = 44.5, λB
3 = 44.5. With
the proposed energy allocation approach, energy allocation
ratios for different risk-aversion level are calculated and showed
in Table 1. Simulation results indicate that energy allocation
depends on the Genco’s risk-aversion. More energy is allocated
to more risky trade if the Genco is less risk-averse and vice versa,
which is consistent with institution.
Case 2. Suppose that λB
1 = 39.5, λB
2 = 45.2, λB
3 = 45.3. Optimal energy allocation ratios for different risk-aversion level are
showed in Table 2. In this case, risk-aversion level has the same
impact on energy allocation as in Case 1. But for the same riskaversion level, more energy is allocated to the bilateral contracts
compared to Case 1. The reason is that contract prices are higher
in Case 2 than in Case 1. That is, for the same market situation,
same decision-maker would trade more energy through bilateral
contract market if contract prices are higher. It is also consistent
with intuition.
5. Conclusion
We have formulated the general portfolio optimization problem as a quadratic programming (QP) problem. The problem
can be solved numerically by a standard QP algorithm. Nevertheless, we link this problem to the standard approach in the
ﬁnancial literature. We show that the solution to the overall portfolio optimization problem with one risk-free and n risky assets
can be obtained in two steps: ﬁrst by optimal selection of n risky
portfolio and then by optimal allocation between the risk-free
asset and the risky portfolio obtained in the ﬁrst step. We also
include a brief tutorial treatment of the standard approach in
ﬁnancial theory.
The general portfolio optimization methodology with n risky
assets can be applied to energy allocation between spot market and bilateral contracts in a market where locational pricing,

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
either zonal or nodal, is adopted to mitigate transmission congestion. A Genco, while trading with a non-local customer, may
pay congestion charge, which is a function of the locational
price difference. Therefore, all non-local contracts are risky as a
result of congestion charge. An example using PJM market data
is used to illustrate the methodology. The results are consistent
with intuition. The method, indeed, can be used to quantify the
intuitive approach of allocating energy between markets. More
accurate estimation on the statistics of spot prices (i.e., E(λS
i,k),
σ2(λS
i,k) and Cov(λS
i,k, λS
j,k)) makes the trading schedule more
applicable.
Themethodologypresentedhereisgeneralandcanbeapplied
in a more sophisticated manner to more detailed practical problems than the example in the paper illustrates.
Acknowledgements
This work has been supported by the Research Grant Council,
Hong Kong, SAR, China, through Grant HKU7174/04E, and
Guizhou University, Guizhou, China, under Grant GUT2004-
014. The authors thank Ms. Xiaojiao Tong for the assistance in
the proof of Theorem A.1.
Appendix A. Theorem of portfolio optimization and its
proof
The utility function deﬁned by U(·) = E(r) −(1/2)A·σ2(r), is
convex. The optimal portfolio can be achieved by directly maximizing the utility function with respect to the weights y and
wi(i = 1–n), i.e.,
Max
y,wi U(·) = E(rC) −1
2Aσ2(rC)
s.t.
n
i=1
wi = 1
wi ≥0
(i = 1, · · ·, n)
(A.1)
where
E(rC) = (1 −y)rB + yE(rP)
σ2(rC) = y2σ2(rP)
E(rP) =
n
i=1
wiE(ri)
σ2(rP) =
n
i=1
n
j=1
wiwjσij
We call problem (A.1) the Overall Portfolio Optimization
Problem. We are going to show that the Overall Portfolio Optimization problem can be solved in two steps.
Step 1 (Optimal risky portfolio)
Find the optimal risky portfolio wi(i = 1–n) by maximizing
the slope of the CAL with respect to variables wi(i = 1–n),
i.e.,
Max
wi s = E(rP) −rB
σ(rP)
s.t.
n
i=1
wi = 1
wi ≥0
(i = 1, · · ·, n)
(A.2)
where
E(rP) =
n
i=1
wiE(ri)
σ(rP) =

σ2(rP) =

n
i=1
n
j=1
wiwjσij
Denote the optimal solution of (A.2) by w∗
i , and the corresponding values of rP, E(rP) and σ2(rP) are rP = n
i=1w∗
i ri,
E(r∗
P) = n
i=1w∗
i E(ri) and σ2(r∗
p) = n
i=1
n
j=1w∗
i w∗
jσij,
respectively.
Step-2 (Optimal allocation between risk-free and risky investments)
Find the optimal allocation y between a risk-free investment
and a risky investment by maximizing the utility function with
respect to the variable y, i.e.,
Max
y
U(·) = E(rC) −1
2Aσ2(rC)
(A.3)
where
E(rC) = (1 −y)rB + yE(r∗
p)
σ2(rC) = y2σ2(r∗
p)
Denote the optimal solution to (A.3) by y*.
The following theorem shows that the solutions to problems
(A.2) and (A.3), taking together, form the solution to problem
(A.1).
Theorem A.1.
Suppose that w∗
i (i = 1, · · ·, n) and y* are solutions to problem (A.2) and problem (A.3), respectively, then
(w∗
i , y∗) is a solution to problem (A.1).
Proof.
The three optimization problems are all convex. We are
going to show that the Kuhn-Tucker optimality conditions for
problems (A.2) and (A.3) together are equivalent to the optimality conditions to problem (A.1). Thus, the optimal solutions of
(A.2) and (A.3) are the optimal solution to (A.1).
Denote the Lagrangian function of (A.2) by
L1(wi, λ, μ) = E(rp) −rB
σ(rp)
+ λ
n
i=1
wi −μTw,
where λ ∈R, μ ∈Rn.
Since w* is a solution to (A.2), according to the optimal conditions, there exists multipliers λ*, μ* such that (w∗, λ∗, μ∗)

M. Liu, F.F. Wu / Electric Power Systems Research 77 (2007) 1000–1009
satisﬁes
∂wL1(w∗, λ∗, μ∗)
ξ1
σ(r∗
P) −E(r∗
P) −rB
σ(r∗
P)
ξ∗
2σ2(r∗
P) + λ∗e −μ∗= 0,
n
i=1
w∗
i = 1,
w∗
i ≥0,
μ∗
i ≥0,
w∗
i μ∗
i = 0
(A.4)
where e = (1, 1, · · ·, 1)T, ξ1 =(E(r1), E(r2), · · ·, E(rn))T and
ξ2 =
2σ11w1 + (σ12 + σ21) w2 + · · · + (σ1n + σn1) wn
(σ21 + σ12) w1 + 2σ22w2 + · · · + (σ2n + σn2) wn
. . .
(σn1 + σ1n) w1 + (σn2 + σ2n) w2 + · · · + 2σnnwn
The ξ∗
2 means the value of ξ2 at point w∗.
Thesolutionto(A.3)isobtaineddirectlybysolving∂yU(·) = 0
as follows:
y∗= E(r∗
P) −rB
Aσ2(r∗
P)
(A.5)
(A.4) and (A.5) can be written equivalently
E(r∗
p) −rB
Aσ(r∗p)
ξ1
σ(r∗
P) −
(E(r∗
p) −rB)2
Aσ2(r∗
P)
ξ∗
2σ2(r∗
P) + E(r∗
P) −rB
Aσ(r∗
P)
λ∗e
−E(r∗
P) −rB
Aσ(r∗
P)
μ∗= 0,
n
i=1
w∗
i = 1, w∗
i ≥0,
E(r∗
P) −rB
Aσ(r∗
P)
μ∗
i ≥0,
w∗
i
E(r∗
P) −rB
Aσ(r∗
P)
μ∗
i = 0,
y∗= E(r∗
P) −rB
Aσ2(r∗
P)
(A.6)
On the other hand, denote the Lagrangian function of problem
(A.1) by
L(w, y, λ, μ) = E(rC) −1
2Aσ2(rC) + λ
n
i=1
wi −μTw
Then the ﬁrst-order necessary optimal condition can be written
as a point (w, y, λ, μ) which satisﬁes
∇wL(w, y, λ, μ) = yξ1 −1
2Ay2ξ2 + λe −μ = 0,
∂yL(w, y, λ, μ) = −rB + E(rP) −Ayσ2(rP) = 0,
n
i=1
wi = 1,
wi ≥0,
μi ≥0,
wiμi = 0, (i = 1, · · ·, n)
(A.7)
where e, ξ1, ξ2 are deﬁned as before. From the second expression
of (A.7), y is solved directly and is then substituted to the ﬁrst
expression of (A.7). Therefore (A.7) follows
∇wL(w, y, λ, μ)
= E(rP) −rB
Aσ(rP)
ξ1
σ(rP) −(E(rP) −rB)2
Aσ2(rP)
ξ∗
2σ2(rP)
+λe −μ = 0,
n
i=1
wi = 1,
wi ≥0,
μi ≥0,
wiμi = 0,
(i = 1, · · ·, n)
y = E(rP) −rB
Aσ2(rP)
(A.8)
Comparing with the expression of (A.6) and (A.8), it shows that
z∗≡
w∗, y∗, E(r∗
P) −rB
Aσ(r∗
P)
λ∗, E(r∗
P) −rB
Aσ(r∗
P)
μ∗
is a solution to (A.8).
From the convex property of problem (A.1), the corresponding variables (w∗, y∗) of z* is a solution to (A.1).
