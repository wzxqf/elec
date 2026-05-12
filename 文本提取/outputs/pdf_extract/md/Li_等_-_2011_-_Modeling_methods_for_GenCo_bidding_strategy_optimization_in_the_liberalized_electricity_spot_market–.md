# Li 等 - 2011 - Modeling methods for GenCo bidding strategy optimization in the liberalized electricity spot market–

## Metadata

- source_pdf: 参考文献/Li 等 - 2011 - Modeling methods for GenCo bidding strategy optimization in the liberalized electricity spot market–.pdf
- extraction_method: pymupdf
- extraction_status: partial
- title: 
- doi: 

## Abstract



## Body

1. Introduction
In a traditional monopolistic or vertically integrated electricity
market, power providers mainly aim to minimize the expected
costs while maintaining an adequate security of supply [1]. Since
1980s, however, the electricity markets have been gradually
evolving toward liberalized or deregulated structures, which are
characterized by open competitive energy markets, unbundling
electricity services, open access to the network, etc. To establish
a competitive electricity market and improve its efﬁciency, the
restructured market allows for gaming on the market power and
tends to stimulate the emergence of new technologies [2,3]. The
electricity spot market can be regarded as a market where the
electricity can be sold or purchased at varying price and delivered
either immediately or at a given moment. Participants have to
make decisions independently under complicated situations with
insufﬁcient information about their rivals and various uncertainties
in the market such as load variations, competitor’s behavior, and
power system contingencies.
The deregulated electricity market behaves more like an
imperfect competition or oligopoly market due to the special
characteristics of the actual electricity market, such as a limited
number of suppliers, long construction periods of power plants,
large capital investment sizes, transmission constraints, and
transmission losses [4]. Typically, only a few dominating generation
companies (GenCos) serve a given geographic region. In such an
oligopolistic market, an individual GenCo has its market power,
that is, it can affect and manipulate market price via its strategic
bidding behavior [5]. This indicates an opportunity for the GenCos
to increase their proﬁts through strategic bidding. Therefore, it is
possible for the GenCos to maximize their proﬁts by optimizing the
bidding strategy in the deregulated electricity spot market while
minimizing the associated risks. Also, the electricity market is
a complex dynamic system with complicated interactions among
physical structures, market rules and all participants. As such, each
participating agent faces risks and uncertainties while pursuing
proﬁt maximization [6,7].
Meanwhile, renewable energies such as wind, solar, and
biomass are regarded as a key factor in tackling global climate
change and energy shortage crisis [8]. For example, wind energy
has globally experienced fast growth during the past decade [9,10].
Along with the ongoing worldwide utilization of wind power and
its increasing penetration into electricity markets, more wind
* Corresponding author. Tel.: þ1 701 231 7119; fax: þ1 701 231 7195.
E-mail address: jing.shi@ndsu.edu (J. Shi).
Contents lists available at ScienceDirect
Energy
journal homepage: www.elsevier.com/locate/energy
0360-5442/$ e see front matter  2011 Elsevier Ltd. All rights reserved.
doi:10.1016/j.energy.2011.06.015
Energy 36 (2011) 4686e4700

GenCos (WGenCos) will participate in the electricity market by
presenting offers and committing the delivery of the agreed
amount of wind energy at a given moment [11]. It is well-known
that regulation power must be reserved to deal with the possible
fast load variations and unforeseen problems with production
capacity [12]. With a high penetration of such intermittent generations, however, the regulation power needed will further increase
in that such generations are characterized by large variations in
addition to the aforementioned load swings and outages of
production capacity [13]. When bidding in the electricity markets,
wind GenCos must pay for energy production deviations resulted
from the prediction error, which can be as much as 10% of the total
generator energy incomes [14]. The optimal bidding strategy,
especially in the deregulated electricity markets with high penetration of intermittent renewable energy, must consider the cost
paid for the energy imbalance in relation with the abovementioned uncertainties and constraints.
Therefore, one critical issue faced by all GenCos participating in
the electricity spot market is how to optimize their bidding strategies according to the limited information available to maximize
their proﬁts. This problem has attracted much attention from both
the academic circle and the industry. In particular, numerous
publications on modeling and analyzing GenCo bidding strategy in
the competitive electricity market have emerged since 2000, and
great progress has been witnessed. However, to the best of our
knowledge, the most recent literature survey on GenCos bidding
strategy was performed 10 years ago [5]. This paper thus aims to
provide a comprehensive literature review on the state-of-the-art
research on this topic. It should also be noted that several review
papers related to electricity markets can be found in literature, but
they focus on different perspectives and are brieﬂy summarized in
the following. Foley et al. review the electricity systems modeling
techniques, discuss various key proprietary models for electricity
systems available in the U.S. and Europe, and provide an information resource on the choice of model in investigating different
aspects of the electricity system [15]. Anderson reviews the
restructuring efforts in the U.S., Australia, the EU and several EU
nations, as well as the responses of consumers in those markets
[16]. Aggarwal et al. review the main methodologies used in forecasting the electricity price in the deregulated markets [17]. Mideksa and Kallbekken review the research on how climate change
would impact the electricity markets [18]. Besides, the review of
the optimal power generation dispatch problem in both regulated
and deregulated electricity markets can be found in [19] and [20].
The remainder of the paper is organized as follows. Section 2
gives a general introduction to the structure and operations of
the liberalized electricity market. Thereafter, the related literature
is categorized according to their modeling algorithms in Section 3
and further summarized in Sections 4e7, respectively. Conclusive
remarks as well as some possible directions for future research are
ﬁnally presented in Section 8.
2. Liberalized market structures and management rules
Participating in electricity market implies presenting bids and
committing the delivery of the agreed amount of energy at a given
moment. Reliable delivery of electrical energy to load centers
entails a continuous process of scheduling and adjusting electricity
generation in response to constantly changing demand [7]. If the
actual energy delivered by one generator is greater or smaller than
what is committed, the GenCo will pay for additional cost of
maintaining the balance between the generation and the load. In
deregulated electricity markets, a GenCo is usually an entity owning generating facilities and participating in the market with the
sole objective of maximizing its beneﬁt [21]. Most evidently,
individual bidding strategies are of essence to the interactions
where the participants’ actions affect others’ possible outcomes.
Bidding strategies should be developed based on the market
structure, auction rules, and bidding protocols. A variety of electricity markets restructuring models have been proposed, which
generally can be categorized into three types: (a) market pools
(PoolCo), (b) bilateral contract (BC) markets, and (c) hybrid markets
(HM) [22]. PoolCos are popular among the approaches to organizing electricity trading. Essentially, a PoolCo is a more centralized
marketplace where an Independent System Operator (ISO) clears
the market according to the bids from both the sellers and the
buyers, operates and manages the entire system, and maintains its
reliability. In a PoolCo, different GenCos compete not for speciﬁc
customers but for the right to supply energy to the grid. A bilateral
contract model deﬁnes a ﬂexible market where the participants can
specify and negotiate the terms and conditions of trading agreements independent from the ISO. The ISO mainly ensures enough
transmission capacity and security. Generally, the market structure
of a fully deregulated hybrid electricity market can be illustrated by
Fig. 1. It may contain not only power pools and bilateral contracts
but also ancillary services (AS) such as frequency and voltage
controls, load following, energy imbalance, spinning reserve, and
supplementary reserve and standby reserve [15]. Competing
generators offer their electricity output to retailers in a wholesale
electricity market; the retailers then re-price the electricity and
take it to the retail market. Although the wholesale pricing used to
be the exclusive domain of large retailing suppliers, the markets are
increasingly opening up to large end-buyers as well. The participants can directly sign bilateral contracts with others and/or bid in
the PoolCos. Also, a fully deregulated hybrid market should allow
the end buyers to choose among different competing suppliers in
the electricity retail market. It should be mentioned that this paper
mainly focuses on the literature pertaining to the electricity
wholesale market in view that the GenCos usually do not interact
with the end consumers directly. Besides, Haas and Auer emphasize
six prerequisites for effective competition in reformed wholesale
electricity markets: (1) separation of the grid from generation and
supply; (2) wholesale price deregulation; (3) sufﬁcient transmission capacity for a competitive market and non-discriminating
grid access; (4) excess generation capacity from many competing
generators; (5) an equilibrium relationship between short-term
spot markets and long-term ﬁnancial instruments for marketers
to manage spot-market price volatility; and (6) an essentially
hands-off government policy that encompasses reduced oversight
and privatization [23]. Moreover, some representative electricity
wholesale markets formed in different geographical locations are
provided in Table 1.
Generally, a bid may include several energy price segments
together with the corresponding quantity of electricity. As the most
common structure, the pool-based market is an auction center in
which all competitive participants are required to submit quantity -
price pairwise bids they commit to receive from or pay to the pool.
As illustrated in Fig. 2, once the bidding period ends, an ISO ranks
and then matches the selling offers with buying bids so that the
buying bids of the highest price are matched with the selling offers
of the lowest price [24]. When all the demands are met, the price of
either the last accepted offer or the ﬁrst rejected offer could be set
as the market clearing price (MCP). The pricing mechanism for all
the rest dispatched GenCos can be either uniform pricing (UP) or
pay-as-bid (PAB). The UP auction indicates that all the winning
suppliers are paid at the same MCP. The PAB auction means that
each winning supplier bidder is paid at its bidding price of the
committed amount of electricity.
Each different market structure has its own auction rules and
bidding protocols. Various auction rules can be categorized into
G. Li et al. / Energy 36 (2011) 4686e4700

two types: static or dynamic. In static auctions, the bidders submit
sealed bids, whereas dynamic auctions allow the bidders to
observe others’ bids so that they can revise their bids sequentially
[5]. The static auction may work based on either the UP or PAB rule.
Bidding strategy in PAB-based market is more complex and
potentially more important than that in UP-based market. In this
case, the GenCos should estimate the uncertain MCP and bid
slightly less than it. The PAB rule represents the future trend in the
deregulated electricity markets and is expected to lower the
market prices and reduce the price volatility [25]. Although the
majority of operating electricity markets currently still employ the
sealed bid auction with the UP auction rule, extensive research has
been carried out on the applications of PAB rule as well [26].
According to different market designs, the bidding protocols can be
divided into two types: single-part bids or multipart bids [5].
Under the single-part bidding protocol, as adopted in the California
type power exchange (PX) market, the GenCos bid independent
prices for each hour; the winning bid and the schedules for each
hour are determined via a simple market clearing process
according to the intersection of supply and demand bid curves.
This decentralized approach does not require the ISO to make unit
commitment decisions. Instead, the GenCos have to consider all
involved costs and constraints in preparing their bids. Therefore,
whenever a signiﬁcant physical or technical constraint occurs in
a generation unit, a modiﬁcation mechanism should be applied to
the schedule, e.g., via short-term balancing market. In contrast,
a multipart bid, as addressed in the EnglandeWales or British type
electricity market, may include separate prices for ramps, start-up
costs, shut-down costs, no-load operations, and energy. Although
this type of bidding protocol can reﬂect the cost structure and the
technical constraints of generation units, the non-convex Unit
Commitment (UC) problem might not converge to a global optimal
solution for large scale systems, possibly resulting in inequitable
dispatches for different GenCos.
Table 1
Typical electricity wholesale markets in the world.
Markets
Introduction
ISO New England
ISO New England, established in 1997, operates the region’s day-ahead and real-time energy spot market with LMP, capacity market,
forward reserves market, regulation market, and ﬁnancial transmission rights market.
PJM
Pennsylvania, New Jersey, and Maryland (PJM) market. It is a regional transmission organization that coordinates the wholesale electricity
in 13 states and the District of Columbia; day-ahead and real-time spot market with LMP, regional and locational capacity market,
ﬁnancial transmission rights market, and bilateral contract market.
MISO
Midwest ISO, established in 2002, administers the day-ahead and real-time energy market known as the Day-2 market with
hourly LMP updating, and a monthly ﬁnancial transmission rights (FTR) allocation and auction.
California ISO
California ISO (CAISO), established in 1996, operates the region’s wholesale electricity market including day-ahead, hour-ahead and
real-time energy spot market with LMP, ancillary services, and ﬁnancial transmission rights market.
NYISO
New York ISO, established in 1999, operates the day-ahead and real-time spot market with LMP, regional and locational capacity market,
and ﬁnancial transmission rights market.
ERCOT
Electric Reliability Council of Texas (ERCOT), formed in 1970, operates both the transmission and the competitive energy
market within the Texas interconnection as a single system. It operates the retail market including customer switching and
the wholesale markets for balancing and ancillary service markets.
Dutch type APX
Dutch Amsterdam Power Exchange (APX), established in 1999, allows the participants to electronically send before 10:30 their
offers/bids for each hour between 00:00 and 24:00 of the next day; requires them to be responsible for the balance between
the committed amount and the actual production.
Nordic Elspot
The spot market, Elspot at the Nordic power exchange (PX) Nord Pool, takes the form of a pool-based market where
participants exchange power contracts for physical delivery in the following operation day.
APX-ENDEX
APX-ENDEX (European Energy Derivatives Exchange), the successor of Dutch APX, is an Anglo-Dutch PX operating
thespotandderivativesmarkets forelectricityandnatural gasin the Netherlands, the United Kingdom, and Belgium. APX-ENDEX
provides exchange trading, central clearing and settlement, and data distribution services as well as benchmark data and industry indices.
Spanish PoolCo
Spanish day-ahead spot market, introduced in 1998, works as follows. Before 11:00 a.m., qualiﬁed buyers and sellers present
their offers for the 24 hourly periods of the following day. Offers can consist of up to 25 different price/quantity pairs for each
period and for each generating unit they own (simple offer), or include indivisibility conditions, a minimum revenue condition,
production capacity and scheduled stop conditions (complex offer).
NEM Pool
The National Electricity Market, under reconstruction since 1998, is an electricity wholesale market for Queensland,
New South Wales, Victoria, South Australia, the Australian Capital Territory and Tasmania. Generators in the NEM bid their
output in a wholesale pool where retailers and other participants buy the electricity at the wholesale spot price determined
by NEMMCO based on the lowest cost generator bids required to meet the electricity demand.
JEPX
Japan Electric Power Exchange (JEPX), established in 2003, operates the wholesale electric power exchange and the
related businesses.
Ontario PoolCo
Ontario electricity market, opened in 2002, consists of physical markets for energy and operating reserves as well as a
ﬁnancial market for transmission rights, administered by the Independent Electricity Market Operator (IEMO). The information
available in advance to the participants is the forecasted demand and the predicted hourly Ontario energy price, while information
related to historical bidding or hourly dispatch of competitors is not accessible.
GenCo
GenCo
GenCo
Supplier
Supplier
Large
consumer
End buyer
Electricity
Retail
Market
End buyer
End buyer
Electricity Wholesale Market
Ancillary
services
Power
pools
Bilateral
contracts
TransCo
ISO
Fig. 1. The general market structure of deregulated electricity markets.
G. Li et al. / Energy 36 (2011) 4686e4700

3. Modeling methods for bidding electricity in spot markets
In 1988, Schweppe noticed that some electricity utilities were
moving away from rigid ﬁxed prices toward a spot price marketplace, and he discussed the management of four aspects of such
new price
structures:
the
electricity
marketplace
itself,
the
customer, the utility, and the regulatory commission [27]. Later,
David formally addressed the strategic bidding issue for competitive power suppliers and developed a conceptual optimal bidding
model and a dynamic programming method for EnglandeWales
type electricity markets, in which each GenCo bids a ﬁxed price for
each generation block [28]. Since then, the strategic bidding
problem for competitive electricity supplier has attracted more
attention, and various modeling approaches have been proposed to
generate strategic bidding strategies [29].
We classify the various modeling literature for bidding strategy
analysis in the electricity spot market into four groups: (1) single
GenCo optimization models, (2) game theory based models, (3)
agent-based models, and (4) hybrid or other models, as illustrated
in Fig. 3. Each group of models may be further divided into small
subgroups according to the model formulation and solution algorithms. For example, the single GenCo optimization models include
many mathematical programming methods such as Mixed Integer
Programming (MIP), Nonlinear Programming (NLP), and Dynamic
Programming (DP); the game theory models might adopt different
competition rules: Bertrand competition, Cournot competition,
Supply Function Equilibrium (SFE), and some other newly proposed
competition rules; the agent-based models can be categorized in
terms of different learning algorithms such as model-based adaptation algorithms (MA), genetic algorithms (GA), Q-Learning (QL),
computational learning (CL), Ant Colony Optimization (ACO), etc.
Generally, the single GenCo optimization models focus on only one
speciﬁc player while simplifying the rest players and the inﬂuencing factors as a set of deterministic or stochastic independent
variables, while game theory and agent-based approaches deal
with the situation of more than one player in the market. Game
theory equilibrium models investigate the bidding strategies from
the perspective of players’ mutual interactions. Agent-based
models tend to mimic human behaviors and simulate optimal
bidding strategies [30]. The main characteristics of the three
modeling approaches are provided in Table 2. Recently, some
hybrid and non-conventional methods have also been proposed.
The representative publications in the four groups are reviewed
and summarized in the following sections, respectively.
4. Single GenCo optimization models
In earlier publications, the issue of optimal bidding strategy
selection is often addressed as a cost-minimization problem and
solved via traditional cost-based UC algorithms [31]. More recently,
under the assumption that the MCP could be regarded as an
exogenous
variable
[32],
many
mathematical
programming
approaches have been applied to address the problem of optimal
bidding strategy selection. The majority of model formulations
incorporate stochastic probabilistic elements, either in the problem
data (e.g., the objective function and the constraints), or in the
algorithm (through random parameter values, random choices,
etc.), or in both [33]. An insightful discussion on the application of
stochastic programming methods to the energy market can be
found in [34]. The typical optimization methods adopted in the
literature include Integer Linear Programming (ILP), Mixed Integer
Programming (MIP), Multi-Objective Linear Programming (MOLP),
Nonlinear Programming (NLP), Dynamic Programming (DP) [15],
newsvendor, and Markov decision process (MDP) models. As
mentioned before, the literature adopting these models typically
optimizes the bidding strategy for a single-market participant
while ignoring or simplifying the behavior aspects of other players.
The representative references are summarized in Table 3, and more
discussion is provided in the following.
As reviewed in [35], many mathematical programming problems in a competitive electric energy framework can be modeled as
mixed-integer linear programming (MILP) models. In [36], an MILP
model is formulated for a price-maker GenCo to solve the selfscheduling problem and maximize the proﬁt in a pool-based
electricity market. Conejo et al. propose several mathematical
programming models for a price-taking thermal GenCo to derive
the optimal bidding strategy in a pool-based market with highly
uncertain MCPs. The problem is ﬁrst modeled as a stochastic
mixed-integer linear programming (SILP) model and then transformed to two MILP models, one of which could be easily solved
using a commercial ILP solver. Then a simple but informative bidding rule is derived from the solution [37]. Similarly, from the
perspective
of
a
price-taking
hydropower
GenCo
(HGenCo)
Quantity
[MW]
Price
[$/MWh]
Price cap
Aggregated
supply
Aggregated
demand
Scheduled
MCP
Fig. 2. Market clearing mechanism.
Game theory
based
Single GenCo
optimization
Hybrid
Agent-based
Modeling Methods
Cournot
Bertrand
SFE
QL
MA
CL
GA
MIP
NLP
DP
MDP
ACO
Fig. 3. Modeling methods for bidding electricity in the spot market.
G. Li et al. / Energy 36 (2011) 4686e4700

participating in the PX of Nord Pool, a day-ahead power market,
Fleten and Kristoffersen transform a two-stage SILP model to an
MILP model for determining optimal bidding strategies by taking
into account the discrete market price scenarios. The effect of
market price uncertainty on bidding optimization is explicitly
explored by comparing the stochastic approach to the deterministic
counterpart [38]. Angarita et al. present the application of
stochastic optimization technique in maximizing the joint proﬁt of
hydro and wind generators in a pool-based electricity market [39].
To handle the uncertainty of wind prediction, the hourly wind
power output is considered as a discrete random variable in the
optimization problem. Compared to other possible bid strategies
that make use of the expected wind power value, the combined
bidding strategy obtains signiﬁcant improvements. Also, it is
a useful tool for GenCos to avoid penalty costs or income reduction.
Sen et al. propose a multi-stage SILP model for scheduling and
hedging in wholesale electricity markets [40]. This SILP model
captures stochastic electricity demand, electricity forward price,
gas forward price, and spot price of electricity. Based on the
structure of the SILP model, a nested column generation decomposition strategy is proposed to decompose the model into three
interrelated sub-problems. The experimental results demonstrate
that the proposed approach could provide robust decisions for
scheduling and hedging problems. Besides, Ni et al. develop
a stochastic mixed-integer program (SMIP) to systematically
handle the MCP uncertainties, bidding risk management, and selfscheduling requirements for a hydrothermal GenCo to maximize
proﬁts under a deregulated market. The model is solved by the
proposed
algorithm
combining
Lagrangian
relaxation
and
stochastic dynamic programming (SDP) method [41].
De Ladurantaye et al. introduce an SILP model to maximize the
proﬁts for a hydropower GenCo of multiple power plants along
a river in a deregulated market. The proposed model aims to
support the price-taking GenCo in its day-ahead bidding decisions
[42]. Morales et al. address a multi-stage SILP problem which
determines the best bidding strategy for a WGenCo in an electricity
market including various trading ﬂoors (e.g., day-ahead market and
a sequence of adjustment markets). In the SILP problem, four
uncertain sources are considered: wind power generation, dayahead market price, adjustment market price and imbalance
energy price. The multi-stage SILP problem is formulated as a linear
programming (LP) model, and a case study for a WGenCo in Kansas
is conducted for the illustration of solving the LP model [43].
Gross and Finlay develop a nonlinear programming (NLP) model
to optimize strategic bids of a GenCo in a multi-period auction
market under the assumption of perfect competition, and propose
a Lagrangian relaxation (LR) method to solve the NLP model [44].
Similarly, to deal with the GenCo’s bidding optimization and selfscheduling problem, Zhang et al. develop an NLP model by
considering uncertain bidding information of other participants,
and solve this model by an LR method [45]. Yucekaya et al. present
two particle swarm optimization (PSO) algorithms to determine bid
prices and quantities under the rules of the Pennsylvania, New
Jersey,
and
Maryland
(PJM)
market.
The
ﬁrst one
employs
a conventional PSO technique whereas the second uses a decomposition technique in conjunction with the PSO approach. It is
found that the latter algorithm can dramatically outperform the
former. Also, it is shown that for nonlinear cost functions, PSO
solutions provide higher expected proﬁts than marginal cost-based
bidding [22]. Similarly, while considering the non-convex operating
cost functions of thermal generating units and minimum up/down
time constraints, Boonchuay et al. propose an optimal risky bidding
strategy for a GenCo by self-organizing hierarchical particle swarm
optimization with time-varying acceleration coefﬁcients (SPSOTVAC) [46]. With rivals’ behavior in competitive environment being
simulated via the Monte Carlo method, the signiﬁcant risk index
based on meanstandard deviation ratio (MSR) is maximized to
generate the optimal bid. The proposed SPSOeTVAC approach is
concluded to be capable of providing a higher MSR than other PSO
methods.
Wen and David propose a stochastic nonlinear programming
(SNLP)
model
for
deciding
optimal
bidding
strategies
for
Table 2
Characteristics of three types of models.
Models
Characteristics
Single GenCo optimization
Developing optimization models to describe the entities in the electricity market with the objective of ﬁnding an optimal solution:
 Well-established and solid mathematical foundation
 Generally focusing on one speciﬁc player in the system by simplifying the rest of the system as a set of exogenous variables
 Usually modeling no aspects of players’ intelligent behaviors
 Difﬁcult to model the complex, uncertain and dynamic systems or analytically derive the optimal bidding strategy for the
GenCos in the deregulated electricity markets
Game theory
Modeling the electricity market as a game and mathematically capturing the players’ behavior in the game where one player’s success in
making choices depends on the others’ choices
 Usually mathematically well-deﬁned, involving a set of game players, a set of bidding strategies, and a speciﬁcation of payoffs for
each possible combination of bidding strategies
 Analyzing the economic equilibria of the electricity market by focusing on the players’ interactions
 Capable of providing analytical rationale and explanation on how strategic bidding behaviors affect the GenCos’ market
power and proﬁts
 All players are assumed to be rational, which does not generally hold in the reality
 The frustrating issue of multiple equilibria often occurs in solving the model
Agent-based
Modeling the complex electricity market as collections of rule-based agents interacting with one another dynamically and intelligently,
simulating human beings’ behavior to make optimal bidding strategies
 Only a few simple rules are speciﬁed for and followed by various agents that situated in the network and behave intelligently
in the system
 Agents usually have and only require imperfect, local information and visibility
 No centralized control or planning is required although random elements often exist either among variable agents or in the system
 Agents can interact with each other directly or through the environment, resulting in complex emergent global behavior
of dynamic-equilibrium and adaptation
 More ﬂexible, robust, and easily implemented compared with analytical approaches
 Capable of capturing the details about agents behaviors, which is helpful in ﬁguring out the relationships between individual
decisions and system behavior
 Capable of modeling the dynamics of systems that are not in equilibrium as well
 The underlying mathematical foundation is still not well developed
 Requiring computation-intensive procedures
G. Li et al. / Energy 36 (2011) 4686e4700

competitive power suppliers in a sealed bid auction-based electricity market. The model assumes that the power supplier bids
a linear supply function and is paid at the MCP with the system
dispatch levels being stipulated by a market operator to minimize
customers’ payments. It is shown that the MCP can be signiﬁcantly
higher than the competitive levels if the suppliers bid strategically,
and that the market power of the suppliers will be reduced if the
load is elastic to the price of electricity [4]. Similarly, they build
more SNLP models and propose a genetic algorithm based method
to build bidding strategies for power suppliers in the Californiatype day-ahead energy market in which power suppliers are
required to simultaneously bid 24 linear energy supply functions,
one for each hour, and the system dispatch levels are stipulated
separately for each hour by utilizing the UP pricing rule. The
method is believed to be especially suitable for those suppliers with
marginal or near-marginal generating units [47]. Ma et al. develop
an SNLP model for optimizing bidding strategies by considering the
risks for the GenCos participating in a pool-based single-buyer
electricity market. Each GenCo is assumed to bid a linear supply
function and the system is dispatched to minimize the total
purchasing cost of the single-buyer. Each GenCo chooses the coef-
ﬁcients in the linear supply function for making tradeoff between
two conﬂicting objectives: proﬁt maximization and risk minimization [48]. Guan et al. develop a bidding strategy based on the
theory of ordinal optimization. The basic idea is using an approximate model to describe the inﬂuence of bidding strategies on the
MCP. A nominal bid curve is obtained by solving optimal power
generation for a given set of MCPs via Lagrangian relaxation. The
best bid is then selected by solving full hydrothermal scheduling or
unit commitment problems [49].
Usaola et al. formulate a stochastic linear programming model
by including the probability density of wind forecasting and
analyze the optimal bidding strategy for a WGenCo in a spot
market. It is found that the most accurate prediction is achieved
when bids are updated in intraday markets by using more recent
predictions. However, the most accurate prediction cannot ensure
the highest revenues due to the different prices of spilled and
bought energy and the bias of the prediction programs. In order to
generate maximum revenue, the uncertainty of the power prediction must be considered [50]. Zhang uses a dynamic random effect
ordered probit model to analyze the GenCo’s bidding behavior in
the NYISO day-ahead wholesale market. The results show that
generators in higher-priced groups tend to withhold their capacity
strategically to push up market prices. It is also veriﬁed that
Table 3
Representative optimization based modeling methods for bidding strategy analysis.
Ref.
Models/Methods Auction
rules
Markets
Assumptions
Applications
de la Torre et al. [36]
MILP
UP
PoolCo
Price-maker GenCo
Self-scheduling in a PoolCo; market
power analysis
Conejo et al. [37]
SILP/MILP
UP
PoolCo
Price-taking GenCo; uncertain MCP
A probabilistic framework for treating
uncertain MCP
Fleten et al. [38]
SILP/MILP
UP
Nord Pool
Price-taking GenCo; uncertain MCP
Exploring the effects of the uncertain
market price in day-ahead markets (DAMs)
Angarita et al. [39]
SILP/MILP
UP
Spanish type PX
Price-taking WGenCo and HGenCo
Combined strategy for bidding and
operating in DAMs.
Sen et al. [40]
Multi-stage SILP UP
Forwards and Spot Monthly forward contracts; forward
prices reﬂect the expected spot prices
Power portfolio optimization in the
wholesale market
Ni et al. [41]
SMIP
UP
ISO New England
Power blocks can be different for
different hours; Markovian MCPs
Maximizing the proﬁt and managing
bidding risks in DAMs
De Ladurantaye et al. [42] SILP
UP
General DAM
Price-taking GenCo; known MCPs
Hydropower GenCos’ proﬁt maximization in DAMs
Morales et al. [43]
SILP/MILP
UP
Hybrid
Rational bidders; risk-neutral case; no
penalty for balancing market bidder
Decreasing WGenCos’ risk of proﬁt variability
Gross et al. [44]
NLP
UP
British type PX
Perfect competition for GenCos;
individual bids do not affect MCP
Multi-period auction rule
Zhang et al. [45]
NLP
UP
ISO New England
Quadratic bid function with rivals’
being available as distributions
Cost minimization of bidding in DAMs
Yucekaya et al. [22]
NLP/PSO
UP
PJM
The uncertain MCP is exogenous to
the bidding decisions
GenCo bidding for maximum proﬁts
Boonchuay et al. [46]
NLP/PSO
UP
PoolCo
Step-wise bidding protocol
Risk management of proﬁt variation
of GenCo in the pool
Wen et al. [4,47]
SNLP/GA
UP
California ISO
Sealed bids on 24 linear supply functions
w/o price caps
Market power analysis for DAMs
Ma et al. [48]
SNLP
UP
PoolCo
Single buyer; sealed bid on linear supply
function with power caps
Minimizing GenCos’ risks while
maximizing their proﬁts.
Guan et al. [49]
NLP
UP
California ISO
Bid can be submitted for individual unit
or as an aggregated bid
Providing “good enough” bids in DAMs
with less computation
Usaola et al. [50]
SLP
UP
Spanish PoolCo
Zero bid price of WGenCo; perfect
price forecast
WGenCo bidding for maximum proﬁts
Zhang [51]
Statistical model UP
NYISO
Perfectly inelastic and exogenously
given load proﬁle
Demand side management
Rahimiyan et al. [52]
News-vendor
PAB
General DAM
Normally distributed MCPs
Risk analysis and proﬁt maximization
in a PAB PoolCo for DAMs
Song et al. [53]
MDP
UP
General DAM
Operational constraints & demand-side bid
are ignored; risk-neutral GenCo
Optimizing bidding decisions over a
planning horizon
Gajjar et al. [54]
MDP
UP
PoolCo
Known forecast daily load with load
variation being ignored
Long-term proﬁt maximization with
stochastic risks
Bathurst et al. [55]
MDP
UP
Hybrid
Different imbalance price assumptions
WGenCo imbalance cost minimization
in the hybrid
Pinson et al. [56,57]
News-vendor
UP
Dutch type APX
Price-taking WGenCo; no internal balancing
control; selling all for MCP
Revenue maximization of WGenCo in DAMs
G. Li et al. / Energy 36 (2011) 4686e4700

demand conditions might affect market prices signiﬁcantly [51].
Rahimiyan et al. formulate the bidding decision-making problem
from a supplier’s viewpoint in a PAB auction spot market by
assuming a normally distributed MCP, and analyze the effect of
some risk factors on the supplier-expected beneﬁt and selling
amount [52]. Song et al. propose a bidding decision-making
strategy in which the impacts of production limit and market
share on the optimal bidding strategies are considered. It is
concluded that the Markov decision process (MDP) model is able to
optimize the decision over a planning horizon. However, the model
makes a few strong assumptions such as ignoring the power system
operational constraints and giving no provision for incorporating
risk attitude in an ordinary MDP [53]. Gajjar et al. formulate the
GenCo optimal bidding in a deregulated power market in the
framework of MDP [54]. An optimal strategy is devised to maximize
the proﬁt by employing the temporal difference technique and
actor-critic learning algorithm. The method is concluded to be
especially
useful
for
long-term
proﬁt
maximization
under
stochastic risks. Bathurst et al. present a strategy for bidding a few
hours before the operation time for the wind producers under the
New Electricity Trading Arrangements (NETA), changed to British
Electricity Trading Transmission Arrangements since 2005, by
using Markov processes for simulating a wind farm [55]. The
method demonstrates substantial reductions in the imbalance
costs as well as the effect of market closure delays and window
lengths of wind forecasting. Pinson et al. study daily bidding by
using the rules of the Dutch APX electricity market, where bids are
presented only once per day and not updated in intraday markets
[56]. It is found that as a result of reduced regulating market costs
from better hourly predictions to the market, wind power producer
could obtain up to 8% more proﬁt if the time between market bids
and delivery is shortened from the day-ahead Elspot market. They
also formulate a general methodology for deriving optimal bidding
strategies based on probabilistic forecasts of wind generation in the
form of predictive distribution. This ﬂexible methodology can be
tailored based on the needs of a speciﬁc market participant [57].
Besides, bi-level optimization is often applied either to represent the strategic interaction among suppliers or in hybrid markets
where electrical energy and spinning reserve are simultaneously
traded [58,59] or in the presence of future contracts [60] and
bilateral contracts [61]. Also, the inﬂuence of extra objectives such
as the minimization of supplier emission of pollutants [62], or the
inﬂuence of unit reliability [63] has been analyzed by using optimization models. The competition process can be represented as
a dynamic feedback system as well [2]. Attaviriyanupap et al.
propose an algorithm for determining the optimal bidding strategy
for a GenCo in the deregulated day-ahead power and reserve
markets [64]. The optimal bidding parameters for both markets are
determined by solving an optimization problem, which considers
unit commitment constraints such as generating limits and unit
minimum up/down time constraints. In the study, evolutionary
programming (EM) technique is used to solve the problem. In
addition, a few other single GenCo optimization studies on bidding
strategy in the electricity spot market can be also found in literature
[65e72].
5. Game theory models
Game theory models, also called equilibrium models, optimize
the bidding strategies by investigating players’ interactions and
analyzing economic equilibria of the system. Typically, in a game,
each player chooses the strategy from its own strategy set; then
a payoff will be assigned to each player by the payoff function; as
a result the optimal solution can be reached via Nash equilibrium.
Nash equilibrium is a strategy combination of all players in which
no player can increase its payoff by changing its own strategy alone
so that every player will ﬁnally choose its strategy exactly as the
equilibrium strategy combination. Game theory models provide
analytical rationale and explanation on how market power can be
exercised via strategic bidding behavior, but the assumption that all
players are rational usually does not hold in practice [30]. Also, it is
limited by the requirement of common knowledge on all GenCos’
actual production costs [29].
One major criterion for classifying game theory based methods
is the level of competition: cooperative and non-cooperative [73].
According to the competition level in the liberalized electricity
market, three general types of game models in imperfect competition, namely, Bertrand, Cournot, and SFE, have been proposed in
the recent literature. As the most competitive model, a Bertrand
competition model is an oligopolistic framework where the GenCos
compete with one another by using prices as the strategic variables
and ignore their capacity constraints. In classic Cournot models,
however, the GenCos compete by using quantities as strategy
choices, under the assumptions of homogenous products, priceresponsive demand, and an MCP is determined by the intersection of aggregated supply and market demand curves. In the SFE
models, the GenCos compete through the simultaneous choice of
supply functions [63]. The competition level as well as the derived
price equilibria generally lies between the Bertrand and Cournot
model [74]. Recently, some other methods are also used to model
and analyze the strategic behavior in deregulated electricity
markets.
5.1. Bertrand competition based models
In a standard Bertrand competition model, identical sellers are
assumed to have constant unit costs and no capacity constraints in
competing on the price offers to consumers, and this will inexorably cause the identical sellers to price at marginal cost [75].
However, the electricity spot market deviates from the standard
Bertrand price game. It is a capacity-constrained oligopoly and the
marginal cost pricing is unlikely to be an optimal bidding strategy
[76]. Therefore, only a few relevant references can be found in
recent publications. The representative Bertrand competition based
models are summarized in Table 4.
Federico and Rahman analyze the effects of changing auction
rule from UP in the wholesale market to PAB under two polar
market structures (i.e., a perfect competition or Bertrand structure
and a perfect collusion or monopoly bidding) with demand
uncertainty [77]. It is found that under Bertrand structure there is
a tradeoff between efﬁciency and consumer surplus while changing
to the PAB rule. Also, a move from UP to PAB under monopoly
conditions has a negative impact on proﬁts and output (weakly),
a positive impact on consumer surplus, and ambiguous implications for welfare and average prices [77]. Based on generalized
Bertrand game, Ernst et al. propose to optimize the proﬁt and
obtain a strategic bid by assuming that each GenCo has a constant
marginal cost over its domain of generation and its rivals do not
change their bids from the last round of market to the next one [78].
They further adopt quadratic cost functions for GenCos and include
a supply function in the bids instead of playing a Bertrand game
[79]. Hu et al. deﬁne Bertrand-Edgeworth (B-E) auctions as
a modiﬁed version of Bertrand-Edgeworth games where the
demand is inelastic and a price cap is set exogenously [80]. B-E
auctions are motivated by the discriminatory procurement auctions
used in some wholesale electricity markets. They characterize the
equilibrium structure for B-E auctions with multiple asymmetric
bidding suppliers. Based on a proposed numerical algorithm, it is
numerically illustrated that a weak (low-capacity) bidder does not
necessarily price more aggressively in an oligopoly market [80].
G. Li et al. / Energy 36 (2011) 4686e4700

Bunn et al. [81] develop a stylized Bertrand game with constraints
to explore the main strategic decisions faced by the GenCos in the
England and Wales (E&W) market based on the proposed NETA.
5.2. Cournot competition based models
Compared with Bertrand price-setting strategies, the quantitysetting equilibrium is more realistic for the electricity market. The
Bertrand equilibrium assumes that by providing a lower price than
others, a ﬁrm can capture the entire market demand and then meet
it by expanding its output. However, such assumption is not tenable
in view of the increasing marginal cost of electricity generation at
a time point and the generation capacity constraints. Because the
GenCos provide a homogeneous product, the Cournot assumption
that ﬁrms make strategic decisions by quantity-setting behavior is
considered to be a better approximation to reality than the pricesetting assumption [82,83]. The representative Cournot competition based models are summarized in Table 5.
Borenstein and Bushnell use demand and plant-level cost data
to simulate the competition in a restructured California electricity
market [84]. This approach recognizes that ﬁrms might have an
incentive to restrict output in order to raise price, and it enables the
explicit analysis on each ﬁrm’s ability to do so. It is found that, while
the results make the deregulation of generation less attractive than
where there is no market power, they do not suggest that deregulation is a mistake. It is argued that policies promoting the
responsiveness of both consumers and producers to price ﬂuctuations can signiﬁcantly affect the reduction of market power. Willems studies a Cournot competition based game with two GenCos
who share one transmission line with a limited capacity to supply
price-taking consumers [85]. In the model, three different rules for
the network operator to allocate transmission capacity are investigated,
namely,
all-or-nothing,
proportional,
and
efﬁcient
rationing. Their effects on the GenCos competition and revenue are
analyzed. Kian et al. use feedback Nash-Cournot strategies for the
market participants to bid in dynamic electricity double-sided
auctions [86]. The simulation results show that compared with
single-sided auctions, double-sided auctions are more efﬁcient and
lead to more stable and competitive MCPs. Tamaschke et al. focus
on measuring the extent to which market power has been exercised
in a deregulated electricity generation sector, and emphasize the
need to consider the concept of market power in a long-term
dynamic context [87]. A market power index is constructed by
considering the differences between the actual market returns and
long-term competitive returns, and it is estimated by using
a mathematical optimization model. The results suggest that
generators have exercised signiﬁcant market power from deregulation. To investigate whether individual participants can increase
proﬁts by withholding generation from the market, Ahn and Niemeyer develop a Cournot-based model of Korean power system for
a set of loads representing the load duration curve for Korea’s
system loads in 2002 [88]. The results indicate a strong possibility
for the exercise of market power to increase market price in Korean
market.
Krause et al. perform a Nash equilibrium analysis by deﬁning
a pool market as a repeatedly played matrix game and compare it
with an agent-based model. It is concluded that GenCos may act
strategically by bidding above their marginal production costs [3].
Kang et al. also propose a bidding model by using a two-player
static game theory and analyze both demand-ﬁxed and demandunder-uncertainty scenarios. The objective is to exploit a methodology to build the optimal bidding strategies for competitive power
suppliers in day-ahead auction-based electricity markets with only
the information on their possible future proﬁts estimated from
forecast system demand. It is indicated that a precise forecast of the
demand can help the player to gain the advantage in the game [21].
Under the Cournot assumption, Park et al. analyze the power
transactions in a deregulated energy marketplace such as PoolCo by
Table 5
Representative Cournot competition based models for bidding strategy analysis.
Ref.
Model bases
Auction
rules
Markets
Assumptions
Applications
Borenstein et al. [84] Static Cournot UP
Early California
Various assumptions for bidders in multiple scenarios
Comprehensive market analysis and
new market design
Willems [85]
Cournot
UP
Simpliﬁed market 2 GenCos; 1 TransCo; 3 allocating rules
Market design
Kian et al. [86]
Cournot
UP
Simpliﬁed market Double-side auctions; no limit to price area
Proﬁt maximization in the dynamic
oligopolistic market
Tamaschke et al. [87] Cournot
UP
NEM Pool
Present value of the generation capital is assumed to be
repaid over the useful plant life; several ancillary
costs are included.
Market power analysis in both short-run
and long-run contexts
Ahn et al. [88]
Cournot
UP
KEPCO
Varying electricity inputs; with/without withholding
Market power analysis
Krause et al. [3]
Cournot
UP
PoolCo
Constant loads; linear GenCo marginal costs
Assessment of the market dynamics of
network-constrained PoolCo
Kang et al. [21]
Cournot
UP
PoolCo
Only 2 GenCos, knowing each other’s payoffs and strategies Proﬁt maximization
Park et al. [89]
Cournot
UP
PoolCo
2-player game with complete information
Proﬁt maximization and market analysis
Table 4
Representative Bertrand competition based models for bidding strategy analysis.
Ref.
Model bases
Auction
rules
Markets
Assumptions
Applications
Federico et al. [77]
Bertrand
PAB
British type PoolCo
Polar structuree perfect competition or collusion
Market design under the PAB
auction rule
Ernst et al. [78]
Bertrand
UP
Simpliﬁed 2-node PoolCo
The same bid posted as the previous MCP by
other agents; known transmission capacity and loads
GenCo’s proﬁt maximization
and power market design
Minoia et al. [79]
Bertrand
UP
Simpliﬁed 2-node PoolCo
Each session of the market is regarded as a stage
GenCo’s proﬁt maximization and
market design
Hu et al. [80]
Bertrand-Edgeworth
PAB
Simpliﬁed
Inelastic demand and exogenous price cap
Proﬁt maximization in the
dynamic oligopolistic market
Bunn et al. [81]
Bertrand
PAB
E&W PoolCo
NETA-based; transmission limits ignored;
GenCos independent of suppliers
Analysis of market power and
market design policy issues
G. Li et al. / Energy 36 (2011) 4686e4700

modeling it as a non-cooperative game with complete information
and determining the solution in a continuous strategy domain. A
new hybrid solution approach employing a two-dimensional
graphical approach as well as an analytical method is proposed to
provide more apprehensible analysis [89].
5.3. SFE based modeling methods
The Supply Function Equilibrium (SFE), originally introduced by
Klemperer and Meyer [90], is a way of describing how competitors
could maximize proﬁts in the competitive market of a single
product with uncertain demands. In such a market, the participants
prefer to set supply functions rather than compete in prices (Bertrand competition) or quantities (Cournot competition) [90]. Green
and Newbery further advance the SFE theory by considering
capacity constraints in analyzing the competition in the British
electricity spot market, and they develop a model for the market of
privatization [91]. It is veriﬁed that supply curve bidding (SCB) can
better beneﬁt the GenCos compared to ﬁxed quantity-price bids.
The advantages of SFE models compared with other models are also
discussed [92]. The SFE model, which constitutes a good compromise between the Cournot and Bertrand models, is believed to most
accurately reﬂect the actual behavior of players in the real power
markets. Also, it is more appropriate for the centralized markets
where each GenCo bids in terms of a supply curve [6]. Therefore,
the worldwide ongoing deregulation of the electricity market has
been stimulating the SFE based modeling analysis of strategic
bidding behavior. The representative SFE based models are
summarized in Table 6, followed by further discussion.
Li and Shahidehpour propose an SFE based modeling method for
analyzing the competition and bidding strategy among GenCos
with incomplete information while taking into account transmission constraints [93]. The competition is modeled as a bi-level
problem in which the upper subproblem maximizes individual
GenCos’ payoffs and the lower subproblem clears the market.
Sensitivity functions are developed for each GenCo’s payoff with
respect to its bidding strategies in order to solve the bi-level
problem. An eight-bus system is employed to illustrate the
proposed method, and the numerical results show the impact of
transfer capability on GenCos’ bidding strategies. In the Standard
Market Design (SMD) setup for electricity markets, Al-Agtash
presents an SCB approach that iteratively alters the SFE model
solutions and selects the best bid based on both the marketclearing LMP and network conditions [94]. This enables the GenCos to derive their best offering strategy both in the DAM and the
long-term contractual markets. However, the results could vary
signiﬁcantly from one system to another according to the system
characteristics such as network topology and the associated
generation capacities of suppliers [94]. Other general SFE based
modeling studies for strategic bidding in the electricity market can
be found in literature [95e98].
It is observed that the SFE models are difﬁcult in embracing
congestion conditions, capacity constraints, and large systems with
signiﬁcant number of generators, unless strong restrictions are
placed [99]. The supply curve bids obtained from the SFE models
may not lead to a maximum proﬁt, especially when the network is
highly constrained. In view of this, Alagtash and Yamin reformulate
the supply function equilibrium model for a GenCo owning
a number of generators and present a new approach for optimal
supply curve bidding (OSCB) using Benders decomposition [99]. In
their model, the offers of individual generators are simultaneously
optimized to maximize the GenCo’s total proﬁt by taking into
account physical constraints as well as transmission security
constraints.
Many existing applications found in literature are limited to
small systems due to the difﬁculty of analytically calculating the
SFE for large systems. In view of this, Bompard et al. present an
analytical approach to represent the strategic bidding behavior of
the GenCos using an SFE model for large systems, in which the
decision variables of the GenCos are the slopes of the supply
function [100]. The proposed approach proves to be rather precise
in determining the SFE and it can consider the GenCos’ capacity
limits. It is also shown that generation capacity constraints may
contribute to GenCos’ market power. One GenCo may be pivotal if
Table 6
Representative SFE based models for bidding strategy analysis.
Ref.
Model bases Auction rules Markets
Assumptions
Applications
Klemperer et al. [90]
SFE
UP
Simpliﬁed market
GenCo’s cost curves are independent of
SF choices; Various other assumptions
Market power analysis; market design
Green et al. [91]
SFE
UP
Early British PoolCo Smooth supply schedule with
GenCo capacity limits
Market power analysis with capacity
constraints
Hobbs et al. [95]
SFE
UP
Simpliﬁed PoolCo
Slope-ﬁxed linear SF curve; unchanged rival bids Bidding optimization with
transmission constraints
Niu [96]
SFE
UP
ERCOT
Homogeneous and divisible generations;
no transaction cost.
Market efﬁciency analysis and market
power analysis
Sioshansi et al. [97]
SFE
UP
ERCOT BES
Dispatched Generations; constant marginal cost
Bidding under feasibility constraints
in the balancing electricity service
(BES) market; BES market design
Haghighat et al. [98]
SFE
PAB/UP
PoolCo
Slope-ﬁxed linear afﬁne SF curve;
unchanged rival bids
Bidding in the PAB/UP market;
pricing rule analysis
Alagtash et al. [99]
SFE
UP
General PX (DAM)
Constrained transmission; single-side bidding
Solving OSCB problem for GenCos
owning many generators
Bompard et al. [100]
SFE
UP
General DAM
Bid on the slope of the linear supply function
Bidding with generation capacity and
network constraints for DAMs
Genc et al. [101]
SFE
UP
General DAM
GenCos of symmetric SFE and enough capacity
Market power analysis with capacity
constraints for DAMs
Holmberg [102]
SFE
UP
General DAM
Asymmetric GenCos with capacity constraints
Market power analysis with capacity
constraints for DAMs
Vahidinasab et al. [6]
SFE
UP
PoolCo
Single-side auction; quadratic bidding curve
Impacts of GenCo’s pollutant emission
on bidding strategies
Gao et al. [30]
SFE
UP
General DAM & Fuel Single-sided auction; linear demand function
Multi-period and multi-market bidding
optimization; transmission planning
Sahraei-Ardakani et al. [103] SFE
UP
PoolCo
Rational GenCos; slope-ﬁxed linear afﬁne
SF curve; unchanged rival bids; ﬁxed demands
Bidding under many realistic
constraints; market power analysis
G. Li et al. / Energy 36 (2011) 4686e4700

its rivals are capacity constrained. It could substantially raise the
market price by unilaterally withholding the output.
To fully consider the impact of capacity constraints and pivotal
ﬁrms on equilibrium predictions, Genc and Reynolds characterize
the set of symmetric SFEs for capacity-constrained GenCos in the
UP-auction wholesale market. It is shown that the rise of GenCos’
capacities could lead to the increase of this set of equilibria [101].
Holmberg also numerically studies the asymmetric supply function
equilibrium with capacity constraints and shows that in this case,
the valid SFE can be calculated by means of an algorithm that
combines numerical integration with an optimization procedure
that searches for an end condition [102].
To develop optimal bidding strategies for the GenCos in
oligopolistic energy markets, Vahidinasab and Jadid study the
impacts of GenCos’ pollutant emission on their bidding strategies
[6]. By neglecting demand-side bidding and bilateral contracts, the
GenCos are assumed to submit quadratic bidding curves to the
market and bid the quadratic coefﬁcient of bidding curves under
the locational marginal pricing (LMP) mechanism. The model
employs SFE to represent each supplier’s strategic behavior. Normal
boundary intersection approach is used for generating Pareto
optimal set and fuzzy decision making is employed to select the
best compromise solution. The MCP is obtained based on the multiobjective optimal power ﬂow method. The optimal bidding strategies are mathematically developed by using a bi-level optimization problem solution.
In view that most previous research mainly focuses on a singleperiod, single-market model, Gao and Sheble develop an SFE model
for GenCos’ bidding optimization in a multiple-period and multimarket scenario. For the proposed SFE model, they obtain the
equilibrium condition by using discrete time optimal control which
considers fuel resource constraints. The multiple-period optimal
bid strategy is analytically derived by manipulating the intercept
parameters of the SFE model. Both electricity and fuel markets are
simultaneously considered, and the SFEs with resource constraint
and transmission constraint are investigated, respectively [30].
Sahraei-Ardakani et al. propose an n-player dynamic game model
to analyze the bidding strategy in an oligopolistic electricity market
with either ﬁxed or variable demands [103]. In this analytical shortterm model, all players ﬁrst choose their own strategy, and then,
deﬁne an improvement matrix to improve the bidding strategy
after seeing one another’s strategy and the resulting payoff. A statespace approach is used to obtain the supply function equilibrium
strategy for the model under many realistic constraints such as
production
bounds,
ramping
limits,
up/down
times,
system
contingencies, and transmission congestions.
Generally, in a centralized market, where the power pool
collects suppliers’ bids and loads and determines the dispatching
schedule, the competition level as well as the model type is
dependent upon the bidding procedure and the pricing rule. The
GenCos may bid on prices without worrying about quantities
(Bertrand competition), but they may also bid on their production
quantity as a function of the prices to be received from the equilibrium prices (SFE). In a decentralized market, the transactions are
performed in bilateral or multilateral markets and the type of
short-term competition is endogenous. The GenCos may compete
by choosing the quantity they are willing to put on the market and
the MCP is to be determined by an ISO (Cournot competition). The
supply function model is a good compromise between Cournot and
Bertrand competitions in a highly decentralized market [74].
5.4. Other game theory based modeling methods
Besides the above three types of game based methods, some
other
game
models
are
also
proposed
in
recent
literature
[104e107,109,110]. They are listed in Table 7 and brieﬂy discussed in
the following.
Vytelingum et al. [104] ﬁrst develop an adaptive-aggressiveness
(AA) strategy which balances proﬁt and transaction probability for
participants in the Continuous Double Auction (CDA) power
exchange market. Based on this, they further construct a twopopulation game model to analyze the participants’ strategic bidding actions in the CDA market. Borghetti et al. [105] analyze the
inﬂuence of GenCos’ technical constraints on their bidding strategy
selections using an optimization approach integrating the normalform game theory (NGT) representation with a cost-based UC
algorithm. The integration with the UC program enables the selection of the most convenient bidding strategy and embraces both an
accurate estimation of production costs and various technical
constraints. De la Torre et al. develop a Nash equilibrium (NE) based
model for pool-based multi-period PoolCo markets under the
assumptions of price-sensitive demands and proﬁt-maximizing
GenCos [106]. A model considering multi-period bidding, price
elasticity, and network modeling of the market is built. Participants’
bidding strategies can then be possibly detected by deriving Nash
equilibria from the output data of an iterative simulation process.
The strategies are iteratively evaluated to remove those dominated
by others that generate higher proﬁts [106]. Rodriguez and Anders
present a methodology for a GenCo to design an hourly bidding
strategy according to its degree of risk aversion (DRA) (either risk
seeker or risk averse); the bidding curve is composed of optimal
offer blocks determined from forecasted MCP and demand [107].
Bid Function Equilibria (BFE) is a model extended from multi-unit
auction model [108] of wholesale electricity markets by assuming
heterogeneous costs for different generation units of a GenCo.
Crawford et al. investigate the application of BFE in the British
electricity spot market and reveal that BFE is very effective for
electricity market design and operation [109]. However, being
different from SFE and other auction theories, BFE assumes the
availability of complete information regarding each GenCo’s
marginal cost and the market demand, which is usually impractical.
Liu et al. propose an incentive electricity bidding mechanism based
Table 7
Other types of game theory based models for bidding strategy analysis.
Ref.
Model bases
Auction
rules
Markets
Assumptions
Applications
Vytelingum et al. [104]
AA
UP
General DAM
Various market settings; CDA mechanism
Bidding in the CDA DAMs; market efﬁciency analysis
Borghetti et al. [105]
NGT
UP
General DAM
Complete (but imperfect) knowledge of
other participants’ cost functions & strategies
Bidding under feasibility constraints in DAMs
De la Torre et al. [106]
NE
UP
PoolCo
Price-sensitive demand
Bidding in the multi-period PoolCo market.
Rodriguez et al. [107]
DRA
UP
Ontario PoolCo
Inﬁnite fuel supply of ﬁxed price
Thermal GenCo bidding in the pure PoolCo
Crawford et al. [108]
BFE
UP
E&W DAM
Symmetric demand; no capacity
limits; complete information
Market design and bidding strategy
analysis for DAMs
Liu et al. [110]
GSM
UP
PoolCo
Single price-quantity offer and bid; congestion,
system ramp, and other constraints ignored
Market design; market power analysis
G. Li et al. / Energy 36 (2011) 4686e4700

on the signaling game theory, deﬁned as the Generator Semirandomized Matching (GSM) mechanism [110]. This new bidding
mechanism is veriﬁed via a multi-agent simulation model. It is
illustrated that, compared to the High-Low Matching (HLM) bidding
mechanism, the GSM model can decrease the clearing price and
GenCos’ proﬁts and increase the total transaction volume and the
buyers’ overall beneﬁts.
6. Agent-based models
The restructured electricity markets typically involve pricequantity pairwise bids for the sale of large amounts of electricity
by a small number of GenCos, resulting in extremely complex
market processes in which traditional analytical and statistical tools
are difﬁcult to be applied [111]. In an agent-based model, market
participants are modeled as adaptive agents with different bidding
preferences and strategies, and the suppliers are enabled to utilize
their past experiences to improve their behaviors in the market.
Each agent may develop the optimal bidding strategy by learning
from its past experiences obtained from the direct interaction with
environment. This brings a new type of numerical analysis theory to
deal with complex trading issues in the restructured electricity
market [112]. Generally, the agent-based modeling procedure can
be described as follows: (1) deﬁne the research questions to be
resolved; (2) construct a model comprising an initial population of
agents; (3) specify the initial model state by deﬁning the agents’
attributes and the structural and institutional framework of the
electricity market within which the agents operate; (4) have the
model
evolve
over
time
without
further
intervention;
(5)
analyze simulation results and evaluate the regularities observed in
the data [113].
Agent-Based Computational Economics (ACE) is a fairly young
research paradigm that offers methods for realistic electricity
market modeling to overcome some shortcomings of the other
methods discussed above [113]. A growing number of researchers
have developed many agent-based models for simulating electricity markets [114,115]. However, compared with single GenCo
optimization models and game theory based models, agent-based
modeling studies for bidding strategy analysis are much less in
literature. The representative publications are listed in Table 8.
Rahimiyan et al. compare the Q-learning (QL) approach [116]
and the model-based (MB) approach in optimizing supplier’s bidding strategy under electricity PAB auction rule [26]. The suppliers’
behaviors are modeled in a multi-agent system, and the simulation
results show that the Q-learning algorithm can enable the suppliers
to ﬁnd the optimal bidding strategy in the PoolCo market. For
different PDFs, the QL algorithm can always converge to the optimal
solution obtained using the model-based approach. The results also
show that the suppliers could adopt the bidding strategy according
to their rivals’ behavior and other effective factors in power system
operation using the QL algorithm. Sheble proposes a genetic algorithm (GA) and an ACE framework for optimizing sellers’ bidding
strategies in a double-sided auction market where some players
attempt to beneﬁt from applying different strategies to cause
economic instabilities and intentionally drive market prices [117].
Naghibi-Sistani et al. propose a Q-learning algorithm for the
participants to ﬁnd the optimal bidding strategy in the PoolCo
electricity market. In this method, each bidder, independent of the
others, learns about its state and the spot price. The results show
that with the temperature variation reinforcement learning, the
suppliers can learn the optimal policy found by game theory and
can be adaptive to the parameter variations in the market [118].
Wen and David adopt Monte Carlo simulation and a reﬁned
genetic algorithm to build optimally coordinated bidding strategies
for competitive suppliers in energy and spinning reserve markets.
Under the assumptions that each supplier bids a linear supply
function into the energy and spinning reserve markets, respectively, and that the two markets are dispatched separately, each
supplier chooses the coefﬁcients in the two linear supply functions
to maximize the total beneﬁts, considering the rivals’ possible
bidding policy [119]. They also apply a similar GA-based method to
develop an overall bidding strategy for the suppliers in the dayahead market [47]. Gountis and Bakirtzis propose a GA approach
to optimize the proﬁts of individual GenCos with multiple generating units. The model uses Monte Carlo simulation to calculate the
expected proﬁt and GA to ﬁnd the optimal strategy. It is assumed
that each supplier bids a linear supply function and considers the
other bidders’ bidding behaviors in the forms of PDF. However, such
an assumption is usually not realistic since the rational generator
can be expected in a competitive- hit81 bidding environment.
Therefore, the proﬁts estimated by the proposed algorithm are not
realizable [120]. Earlier agent-based simulations employing GA
algorithm for bidding strategy analysis can be found in [121,122].
Besides, some other learning algorithms have also been investigated. For example, Fujii et al. apply a multi-agent model, which
learns a bidding strategy autonomously through trial-and-error
search action, to numerically analyze the price formation process
of an open electricity market [123]. The model is believed to be
helpful in analyzing more general electricity markets which have
several different types of power plants with unit commitment
costs, seasonal and hourly demand ﬂuctuation, real-time regulation
market and operating reserve market. Bunn et al. develop an agentbased simulation model by using computational learning (CL)
algorithm to investigate the impact of vertical integration between
Table 8
Representative agent-based modeling methods for bidding strategy analysis.
Ref.
Learning rules
Auction rules
Markets
Assumptions
Applications
Rahimiyan et al. [26]
MB
PAB
PoolCo
Stochastic MCP of known PDF
Adaptive bidding in the deregulated PoolCo
Rahimiyan et al. [26]
QL
PAB
PoolCo
Past experiences learned to ﬁnd
optimal bid price
Adaptive bidding in the deregulated PoolCo
Sheble [117]
GA
UP
PoolCo
Double-sided auction
Market power analysis in double-sided auction
Naghibi-Sistani et al. [118]
QL
UP
PoolCo
Unknown & unnecessary rivals’ information
Adaptive bidding in the deregulated PoolCo
Wen et al. [119]
GA
UP
Hybrid
Linear SF of energy & spinning reserve
markets being separately dispatched
Adaptive bidding in the deregulated PoolCo
including spinning reserve markets
Gountis et al. [120]
GA
UP
PoolCo
Linear supply function; known rivals’ bids
Adaptive bidding in the deregulated PoolCo
Fujii et al. [123]
TAE
UP
Hybrid
Different types of GenCos; seasonal & hourly
demand ﬂuctuation; real-time regulation
and operating reserve markets
Adaptive bidding in the deregulated PoolCo &
reserve markets
Bunn et al. [124]
CL/MA
PAB
PoolCo
Price-taking demand, GenCos offering above
marginal cost; capacity payments ignored
Adaptive bidding in the deregulated PoolCo
including Korean market; Market design
Azadeh et al. [125]
ACO
UP/PAB/Vickrey
PoolCo
Fixed demand; no transmission constraint;
a step-wise discrete bid function.
Adaptive bidding in the deregulated PoolCo;
design of market clearing rule
G. Li et al. / Energy 36 (2011) 4686e4700

electricity generators and retailers on market power in a competitive wholesale market setting [124]. It is found that in various cases,
whilst vertical integration generally reduces spot prices, it can
increase or decrease the market power of other generators,
depending upon the market share and the technology segment of
the market.
Azadeh et al. propose an agent-based simulation model based on
Ant Colony Optimization (ACO) algorithm to compare three
wholesale electricity market clearing strategies: UP, PAB, and
generalized Vickrey rules. The proposed model is suitable for highdimension bidding functions and enables modelers to avoid “curse
of dimensionality”. They investigate step-wise discrete bid functions
and their impacts on the efﬁciency of the market under different
available market settlement rules. The assumptions in the proposed
algorithm include: inelastic and ﬁxed demand; call market with
different price settlement rules (UP, PAB, and generalized Vickrey
rules); no transmission constraint; and a step-wise discrete bid
function. The method can solve dynamic and static combinatorial
optimization problems of market strategy optimization [125].
Besides, there are a few other studies on agent-based modeling
applications in literature [25,126e132]. Xiong et al. compare the UP
and PAB auction rules by using a multi-agent approach, where each
adaptive GenCo develops the bidding prices based on Q-learning
algorithm [25]. The experimental results show that the PAB auction
can result in lower expected market prices and price volatility. It is
also shown that the demand-side response has less effect in
reducing market prices under the PAB auction rule since in this
auction the bidders bid as close to the market prices as possible,
which makes the aggregate supply curve more ﬂattened than that
under the UP rule. Xiong et al. propose an evolutionary daily bidding
strategy for the supplier in a perfectly competitive day-ahead electricity auction market under the PAB auction rule [126]. The feasibility of the proposed bidding strategy is veriﬁed by agent-based
simulations. Walter and Gomide present a GA-based approach to
obtain evolutionary GenCo bidding strategies in the power market.
Simulation results illustrate that the strategies derived by this
approach is superior in enhancing the GenCo’s proﬁtability over the
commonly adopted marginal cost-based approaches [127]. They
also introduce a co-evolutionary algorithm to obtain a proﬁtable
bidding strategy for the participant by using information commonly
available in a much dynamic environment. The results demonstrate
that this approach can further improve the GenCo’s proﬁts [128].
Gao et al. build an adaptive multi-agent model to analyze and
compare
the
application
performances
of
genetic
algorithm,
evolutionary programming, and particle swarm in simulating
participants’ bidding behaviors [129]. Sueyoshi and Tadiparthi
develop an agent-based decision support system (DSS) for analyzing
the dynamic price change in the competitive electricity wholesale
environment. The proposed DSS is effective in assessing new trading
strategies in the electricity PoolCo market [130].
Agent-based models can mimic human behaviors and simulate
optimal bidding procedures. In such models, market participants
are handled as adaptive agents with different bidding preferences
and strategies, and their bidding decisions are inﬂuenced by many
uncontrollable factors. This group of models is more ﬂexible,
robust, and easily implemented compared with the previous two
groups of approaches, and thus it opens up a new type of modeling
analysis to deal with business complexity of electricity trading.
However, the underlying mathematical foundation for agent-based
modeling has not yet been clearly veriﬁed to date [30].
7. Hybrid and other modeling methods
Besides the above three major groups of modeling approaches,
there are a few other innovative methods developed recently for
strategic bidding analysis. In particular, the hybrid approach that
combines multiple modeling methods stimulates enormous interests among the researchers [133e136].
Yamin and Shahidehpour develop a hybrid model combining LR
algorithm and GA for the GenCos to generate a proper unit
commitment scheduling and derive the optimal supply curves. It is
demonstrated that the proposed hybrid model is better than the LR
approach and the traditional unit commitment approach in terms
of helping the GenCos to increase proﬁts [133]. They also adopt the
augmented
Lagrangian
relaxation
algorithm
to
solve
selfscheduling and energy bidding problems in competitive electricity markets constrained by transmission congestions, fuel, and
emission. The supply curve is derived as a function of generation
schedule to achieve the maximum proﬁt. The slope of the supply
curve is dependent upon the forecast price and the power output
obtained in the self-scheduling result [134].
Sueyoshi proposes an agent-based approach equipped with
game theory for analyzing the strategic collaboration among
learning agents during a dynamic market change in the 2000e2001
California electricity crisis [135]. The concept of partial reinforcement learning is incorporated into trading agents who can learn
from both the dynamic market change and the collaboration with
other traders. It is found that the learning speed of traders becomes
slow when a large ﬂuctuation occurs in the power exchange
market. Azevedo and Correia propose a model by combining Bayes’
rule and the game theory for the participants in the ﬁrst stage of an
electric energy bilateral contract auction in the Brazilian market.
The model can help one agent to attribute bids to the other agents
and observe the consequences [136].
In addition, Song et al. present the concept of conjectural variation
(CV) and its applications to the strategic bidding of GenCo in the
oligopolistic electricity spot market. The conjecture of a ﬁrm is
deﬁned as its belief or expectation of how its rivals will react to the
change of its output. It is veriﬁed that, in a real spot market containing
multipleplayers,CVbasedbiddingstrategy(CVBS)enablesaGenCoto
integrate its rivals’ responses into one pseudo-competitor’s response
and to make optimal decision accordingly based on available imperfect information announced in the market. It is also demonstrated
that classical game theoretical bidding strategies (GTBS), such as
Bertrand,Cournot, Stackelberg and monopoly (collusion), are actually
the special cases in the CVBS solution family [29].
8. Conclusive remarks
In the near future, GenCos are encouraged to directly participate
in the liberalized electricity wholesale market by submitting their
strategic supply offers. This makes it possible for the GenCos to
maximize their proﬁts by optimizing their bidding strategies. In the
literature, many models have been proposed to analyze the bidding
behaviors in the oligopolistic electricity wholesale market and their
performances depend on different market structures, auction
mechanisms, and other factors. This paper provides a systematical
review on the recent publications analyzing the bidding strategy in
the liberalized electricity wholesale market. Generally, various
models can be classiﬁed into three main groups: single GenCo optimization models, game theory models, and agent-based models.
Besides, some hybrid models have also been developed recently.
In addition to the dynamics and uncertainty of the load, the
price, and the renewable generation, the GeoCos bidding strategy is
directly affected by and dependent upon the deregulated electricity
market itself. However, since the electricity wholesale market is
still under reconstruction, many interesting and meaningful issues
related to bidding optimization under various possible market
designs should be investigated and tackled, which include but are
not limited to the following topics:
G. Li et al. / Energy 36 (2011) 4686e4700

 Bidding under different market prototypes: Various electricity
market structures are being designed and tested to ensure free
access, fair competition, high efﬁciency, and systems security
and reliability. The analysis on GenCos’ strategic bidding or
market power is important for the market design efforts.
 Bidding under PAB auction rule: Bidding strategy under the
discriminatory pricing rule has its advantages and disadvantages in deregulating the market. However, the relevant
research efforts are still far from enough.
 Bidding under double-sided auction mechanism: Allowing the
buyers as well as the sellers to submit their competitive bids,
double-sided auctions are commonly regarded as a better
setting for deregulating the electricity markets. However, bidding under such settings may also face with increased uncertainty and risks, and this calls for quantitative modeling
analysis.
 Bidding in hybrid or combined markets: The hybrid market
often exists in which electrical energy is traded together with
spinning reserves simultaneously or in the presence of future
contracts and bilateral contracts. Also, with the combination
of day-ahead, hour-ahead, and real-time markets, many
reconstructed electricity markets allow the participants to
update day-ahead supply offers and purchasing bids before
the actual delivery. Besides, the future electricity markets, by
combining the wholesale and retailing markets, should allow
the end users to bid in the markets directly. Bidding optimization in the combined markets will be a signiﬁcant topic for
research.
 Advance in modeling methods and algorithms: To better
represent the transaction behaviors in complex electricity
markets, the shortcomings of each method should be overcome
by continuously pushing for new theoretical developments.
 Bidding
cooperatively
or
with
internal
control
strategy:
Without affecting the oligopolistic nature of the deregulated
electricity market, locational GenCos, especially for those
owning different generation resources, could bid in the market
cooperatively. For example, a WGenCo can cooperate with
neighboring WGenCos or other type of GenCos, or the WGenCo
can adopt some internal balancing strategies, e.g., integrating
the wind generation with storage technology. The effects of
such cooperative bidding strategies or internal controls should
be further investigated.
 Risk management under uncertainty: To deal with the risks of
bidding under the uncertainty in demand, price and production, different risk control measures and operational scenarios
need to be investigated. This is especially important for those
renewable GenCos with distributed and intermittent productions. For instance, besides the uncertain load and electricity
price, the wind generators should also consider the uncertain
wind generations before submitting their supply offers to the
electricity markets.
