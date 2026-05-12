# Boroumand 等 - 2015 - Hedging strategies in energy markets The case of electricity retailers

## Metadata

- source_pdf: 参考文献/Boroumand 等 - 2015 - Hedging strategies in energy markets The case of electricity retailers.pdf
- extraction_method: pymupdf
- extraction_status: success
- title: 
- doi: 

## Abstract

As market intermediaries, electricity retailers buy electricity from the wholesale market
or self generate for re(sale) on the retail market. Electricity retailers are uncertain about
how much electricity their residential customers will use at any time of the day until they
actually turn switches on. While demand uncertainty is a common feature of all commodity
markets, retailers generally rely on storage to manage demand uncertainty. On electricity
markets, retailers are exposed to joint quantity and price risk on an hourly basis given the
physical singularity of electricity as a commodity. In the literature on electricity markets,
few articles deals on intra-day hedging portfolios to manage joint price and quantity risk
whereas electricity markets are hourly markets. The contributions of the article are twofold.
First, we deﬁne through a VaR and CVaR model optimal portfolios for speciﬁc hours (3am,
6am, .
,12pm) based on electricity market data from 2001 to 2011 for the French
market. We prove that the optimal hedging strategy diﬀers depending on the cluster hour.
Secondly, we demonstrate the signiﬁcantly superior eﬃciency of intra-day hedging portfolios
over daily (therefore weekly and yearly) portfolios. Over a decade (2001-2011), our results
∗Associate Professor, Department of Applied Economics, PSB Paris School of Business, 59 rue Nationale 75013 Paris France.
†Universit´e Paris 8 (LED), 2 rue de la Libert´e, 93526 Saint-Denis Cedex, France. Researcher of the
Chaire European Electricity markets (CEEM) of Paris Dauphine University.
‡London School of Economics and Political Science, Houghton Street, WC2A 2AE, London, England.
This paper has beneﬁted from the support of the Chaire European Electricity Markets of the Paris
Dauphine Foundation, supported by RTE, EDF and EPEX Spot. The views and opinions expressed in
this Working Paper are those of the authors and do not necessarily reﬂect those of the partners of the
CEEM.

clearly show that the losses of an optimal daily portfolio are at least nine times higher than
the losses of optimal intra-day portfolios.

## Body

Raphaël Homayoun Boroumand, Stéphane Goutte,
Simon Porcher and Thomas Porcher
Hedging strategies in energy markets: the
case of electricity retailers

Article (Accepted version)
(Refereed)
Original citation:
Boroumand, Raphaël Homayoun and Goutte, Stéphane and Porcher, Simon and Porcher,
Thomas (2015) Hedging strategies in energy markets: the case of electricity retailers. Energy
Economics, 51. pp. 503-509. ISSN 0140-9883

DOI: 10.1016/j.eneco.2015.06.021

Reuse of this item is permitted through licensing under the Creative Commons:

© 2015 Elsevier B.V.
CC BY-NC-ND 4.0

This version available at: http://eprints.lse.ac.uk/82976/

Available in LSE Research Online: June 2017

LSE has developed LSE Research Online so that users may access research output of the
School. Copyright © and Moral Rights for the papers on this site are retained by the individual
authors
and/or
other
copyright
owners.
You
may
freely
distribute
the
URL
(http://eprints.lse.ac.uk) of the LSE Research Online website.

Hedging strategies in energy markets: the case of
electricity retailers
Rapha¨el Homayoun BOROUMAND ∗, St´ephane GOUTTE †,
Simon PORCHER ‡ and
Thomas PORCHER ∗.
August 1, 2015
Abstract
As market intermediaries, electricity retailers buy electricity from the wholesale market
or self generate for re(sale) on the retail market. Electricity retailers are uncertain about
how much electricity their residential customers will use at any time of the day until they
actually turn switches on. While demand uncertainty is a common feature of all commodity
markets, retailers generally rely on storage to manage demand uncertainty. On electricity
markets, retailers are exposed to joint quantity and price risk on an hourly basis given the
physical singularity of electricity as a commodity. In the literature on electricity markets,
few articles deals on intra-day hedging portfolios to manage joint price and quantity risk
whereas electricity markets are hourly markets. The contributions of the article are twofold.
First, we deﬁne through a VaR and CVaR model optimal portfolios for speciﬁc hours (3am,
6am, .
,12pm) based on electricity market data from 2001 to 2011 for the French
market. We prove that the optimal hedging strategy diﬀers depending on the cluster hour.
Secondly, we demonstrate the signiﬁcantly superior eﬃciency of intra-day hedging portfolios
over daily (therefore weekly and yearly) portfolios. Over a decade (2001-2011), our results
∗Associate Professor, Department of Applied Economics, PSB Paris School of Business, 59 rue Nationale 75013 Paris France.
†Universit´e Paris 8 (LED), 2 rue de la Libert´e, 93526 Saint-Denis Cedex, France. Researcher of the
Chaire European Electricity markets (CEEM) of Paris Dauphine University.
‡London School of Economics and Political Science, Houghton Street, WC2A 2AE, London, England.
This paper has beneﬁted from the support of the Chaire European Electricity Markets of the Paris
Dauphine Foundation, supported by RTE, EDF and EPEX Spot. The views and opinions expressed in
this Working Paper are those of the authors and do not necessarily reﬂect those of the partners of the
CEEM.

clearly show that the losses of an optimal daily portfolio are at least nine times higher than
the losses of optimal intra-day portfolios.
Keywords: Electricity; Risk; Retailer; Hedging; Portfolio; Intra-day; VaR;
CVaR.
JEL classiﬁcation: C02, L94, G11, G32.
Introduction and literature review
In competitive wholesale and retail electricity markets, electricity retailers buy electricity from producers through long term contracts, on the day-ahead/spot market, or selfgenerate, for (re)sale on the retail market. On the residential segment, retailers have
to serve ﬂuctuating load at usually ﬁxed predetermined prices (Boroumand and Zachmann, 2012; Bushnell, 2008). As market intermediaries, retailers have the contractual
obligation to harmonize their upstream (sourcing) and downstream (sales) portfolios of
electricity. Demand uncertainty is a common feature of all commodity markets and is traditionally managed through inventories. For all commodity retailers, inventories enable
intertemporal arbitrages and facilitate matching between sourcing and selling portfolios
in accordance with supply/demand variability. However, in electricity markets, retailers
are uncertain about how much electricity their customers will consume at any hour of
the day until they turn actually switches on. In standard electricity retail contracts, retailers operate under an obligation to serve and cannot curtail delivery (except in the
case of the so-called interruptible contracts). On the supply side, the economic non storability of (large) electricity volumes contributes to make electricity markets very speciﬁc.
Consequently, electricity needs to be generated and consumed simultaneously. This nonstorability contributes to the exceptionally high volatility of electricity wholesale prices
in most spot markets around the world (Geman, 2008). The crucial dimension of price
formation in electricity markets is the instantaneous nature of the product (Bunn, 2004)
leading to structural price jumps (Goutte and al. 2013 and 2014). Regardless of how
retailers hedge their expected load, they will inevitably be short or long given demand
stochasticity. Any corresponding adjustment on the spot market will be made at volatile
hourly prices whereas retail prices are generally ﬁxed for a signiﬁcantly longer period
given consumers risk aversion (generally one year minimum with tacit conduction). This
asymmetry of price patterns combined to demand variability can generate very high losses
for retailers which are not eﬃciently hedged. Indeed, retailers cannot pass through in-

creases of wholesale prices to their customers either because of potential losses of market
shares on a longer run or because electricity prices are frozen (like in most US states).
Given the strong positive correlation and multiplicative interaction between load level
and spot price (Stoft, 2002), any under or over- contracted position will be settled at
the most unfavorable times. Most likely, when retailers are short (consumption exceeds
demand forecasts), spot prices are high and above retail prices. Reversely, when retailers are long, spot prices will most likely be lower than their average sourcing cost. To
sum up, the hourly variability of demand, its inelasticity, and the rigidity of supply (non
storability and plant outages) expose retailers net proﬁts to hourly volumetric and price
risks, both correlated with weather conditions (Stoft, 2002). Price and quantity risks can
be very severe given that supply and demand conditions usually shift adversely (Stoft
2002). Suppliers proﬁts depend on electricity demand, spot price, and retail price. Since
retail prices are usually ﬁxed for residential customers (Henney, 2006), proﬁt is strongly
impacted by hourly spot price variations. Consequently, retailers are unable to hedge
their electricity sales by only trading in forward and spot markets on a monthly, weekly,
or daily basis. They need to engage in risk management strategies on an hourly basis
to mitigate the exposure of their proﬁts or their opportunity cost (if they self-generate)
exposed to joint price and volumetric risk. As a consequence of electricity liberalization,
a wide variety of hedging instruments have emerged to enable economic agents to manage
their risks (Hull, 2012; Geman, 2008; Hunt, 2002; Hunt and Shuttleworth, 1997). Since
quantity risk is non tradable (i.e. cannot be transferred by a retailer to another economic
agent), hedging consists in price-based ﬁnancial instruments (Brown and Toft, 2002). In
electricity markets, eﬃcient hedging should be against variations in total costs (quantity
times price), which is complex with hourly demand variability. A retailer proﬁt facing a
multiplicative risk of price and quantity is nonlinear in price. Therefore, hedging with
linear payoﬀinstruments (forward and futures contracts) is not eﬃcient (Boroumand and
Zachmann, 2012). Conventional hedging strategies deal with one source of uncertainty.
Methodologies to hedge price risk have been studied by the literature. However, hedging
joint price and quantity risk for electricity retailers remains an outstanding issue. The
literature on risk management within electricity markets adopts usually the perspective
of electricity producers (Pineda and Conejo, 2012; Conejo et al 2008, Roques et al 2006,
Paravan and al, 2004). Chao et al. (2008) deals with the vertical allocation of risk bearing
within the electricity value chain. On retailers perspective, Boroumand and Zachmann
(2012) compare the risk proﬁles of diﬀerent ﬁnancial and physical hedging portfolios according to the Value at Risk (95%). By deﬁning optimal annual hedging portfolios, they

show the risk management beneﬁts of relying on ﬁnancial options and physical assets
with diﬀerent marginal costs (base, semi base, and peak plants). Chemla et al (2011)
show the superior eﬃciency of vertical integration over forward hedging when retailers
are highly risk averse. Xu and al (2006) present a midterm power portfolio optimization
and the corresponding methodology to manage risks. Oum et al 2006 and Oum and Oren
2010 obtain the optimal hedging strategy with electricity derivatives by maximizing the
expected utility of the hedged proﬁt (Oum et al, 2006) and the expected proﬁt subject to
a VaR constraint (Oum and Oren 2010). The authors explore optimal procurement time
of the hedging portfolio. Vehvilinen and Keppo (2003) study the optimal hedging of price
risk using a mix of electricity derivatives. Carrion et al (2007) develop a risk-constrained
stochastic programming framework to decide which forward contracts the retailer should
sign and at which price it must sell electricity in order to maximize its expected proﬁt for
a given risk exposure. Carrion et al (2009) propose a bilevel programming approach to
solve the medium-term decision-making problem of an electricity retailer.
However, to our knowledge, few articles propose portfolio optimization based on intraday hedging for electricity intermediaries, despite the well-known structural electricity
price spikes subsequent notably to the non storability of electricity. The frequency of spot
hourly price spikes reinforces the necessity of intra day hedging strategies.
Our results clearly demonstrate that the optimal hedging portfolio varies in relation
with the hours of the day. The contribution of the article is twofold. First, our model
demonstrates that the average of the cumulated hourly losses [as measured by the average
VaR and CVaR] of the eight homogeneous group of hours is lower than the VaR (95%) and
the corresponding CVaR of a single daily optimal portfolio. Therefore, we propose several
optimal hedging portfolios per day. Secondly, for any group of hours, we demonstrate that
the optimal portfolio is speciﬁc.
The article is structured as follows: Section 1 presents the statistical features of the
simulated data. Section 2 presents our methodology. In section 3, we present the results
of our simulations. The last section concludes and provides policy recommendations.

Data
The methodology is an extension of Boroumand and Zachmann (2012) with two key differences. First, we realize simulations on electricity price and volume data over a ten
year period (2001- 2011). The extensive data simulation contributes to the high robustness of our results. Secondly, we test intra-day portfolios rather than annual portfolios.
Therefore, we calculate intra-days VaR for each hourly cluster. We take the French spot
electricity price from 27 Nov 2001 to 8 March 2011.
Our model relies on data from the French spot electricity market from 27 Nov 2001
to 8 March 2011. This market is relevant for several reasons. First, the spot price is the
reference price of the French wholesale market. Indeed, many retailers index their price
on the referential spot price. Overall, the EPEX spot auction represents 70% of all day
ahead transactions. Admittedly, the size of the market in 2001 was smaller but it has
never been an extension of the incumbent, which is an actor among others. Indeed, EDF
uses mainly its production for its own portfolio of clients. The French spot market is the
3rd biggest market in Europe in terms of volume (687 TWh in 2011), the HHI index is
low (691 for the last semester of 2011), and the liquidity is high with 57858 transactions
for the ﬁrst semester of 2011 (CRE1, 2011).
We deﬁne eight diﬀerent hourly prices, namely our cluster hours, which are: 3am,
6am, 9am, 12am, 3pm (15), 6pm (18), 9pm (21), 12pm (24).
Figure 1 clearly exhibits spot price spikes.
Figure 2 shows the diﬀerent levels of
consumption volume and variability for each cluster hour.
Hedging strategies
We demonstrate that a retailer cannot reproduce the risk- reducing beneﬁts of physical
hedging by pure contractual portfolios. For this purpose, we compare the risk proﬁles of
diﬀerent portfolios of hedging with the traditional Value at Risk (VaR) indicator. The
Value at Risk (VaR) is an aggregated measure of the total risk of a portfolio of contracts
and assets. The VaR summarizes the expected maximum loss (worst loss) of a portfolio
over a target horizon (10 years in this article) within a given conﬁdence interval (generally
1Observatoire des march´es de l’´electricit´e et du gaz.

Figure 1: Spot electricity price for each cluster hour from 27 Nov 2001 to 8 March 2011.
Price of the spot at3 hour
Price of the spot at6 hour
Price of the spot at9 hour
Price of the spot at12 hour
Price of the spot at15 hour
Price of the spot at18 hour
Price of the spot at21 hour
Price of the spot at24 hour

Figure 2: Electricity load for each cluster hour from 27 Nov 2001 to 8 March 2011.
x 10
Load at3 hour
x 10
Load at6 hour
x 10
Load at9 hour
x 10
Load at12 hour
x 10
Load at15 hour
x 10
Load at18 hour
x 10
Load at21 hour
x 10
Load at24 hour

95%). Thus, VaR is measured in monetary units, Euros in our article. As the maximum
loss of a portfolio, the VaR(95%) is a negative number. Therefore, maximizing the VaR
is equivalent to minimizing the portfolios loss. We rely on the Value-at-Risk because it is
a good measure of the downside risk of a portfolio and is for example used as preferred
criteria for market risk in the Basel II agreement. We strengthen the robustness of our
results with the CVaR.
The Conditional Value-at-Risk, CVaR, is strongly linked to the previous risk measure
(i.e. VaR) which is, as mentioned above, the most widely used risk measure in the practice
of risk management. By deﬁnition, the VaR at level α ∈(0, 1), V aR(α) of a given portfolio
loss distribution is the lowest amount not exceeded by the loss with probability α (usually
α ∈[0.95, 1)). The Conditional Value at Risk at level α CV aR(α) is the conditional
expectation of the portfolio losses beyond the V aR(α) level. Compared to VaR, the CVaR
is known to have better mathematical properties. It takes into account the possible heavy
tails of portfolio loss distribution. Risk measures of this type were introduced by Artzner
et al. (1999) and have been shown to share basic coherence properties (which is not the
case of V aR(α).
2.1
Payoﬀof the assets and contracts within a hedging portfolio
A retailer is assumed to have concluded a retail contract (the retail contract is given ex
ante and is therefore not a portfolios parameter of choice) with its customers that imply
stochastic demand Vt for t = 1 : T. The demand distribution is known to the retailer
and the uncertainty about the actual demand Vt is completely resolved in time t. To
fulﬁll its retail commitments the retailer can buy electricity on the spot market at the ex
ante uncertain spot market price Pt. The spot market price distribution is known by the
retailer. To reduce its risk from buying an uncertain amount of electricity at an uncertain
price, the retailer can conclude ﬁnancial contracts and/or acquire physical generation
assets.
All contracts (including the retail contract and the physical assets generation
volumes) are settled on the spot market that is assumed to be perfectly liquid. Thus, the
payoﬀstreams depend on a given number of hourly spot market realizations.

2.1.1
Portfolios’ structures
Let denote by πi,t, the price at time t = 1 : T of a particular contract with name i. We
consider ﬁve diﬀerent contracts/assets
namely a retail contract, a forward contract, a
power plant, a call option on the spot price and a put option on the spot price given the
spot price. In Table (1), we recall the payoﬀof these ﬁve contracts.
Table 1: Payoﬀs of diﬀerent contracts/assets given the spot price Pt.
Contract
Payoﬀ
Retail contract
πretail,t = −Pt.Vt + E[Pt.Vt]
Forward
πforward,t = Vforward.Pt −E[Vforward.Pt]
Power plant
πplant,t = Vplant × max (Pt −mc, 0) −E [Vplant × max (Pt −mc, 0)]
Call option
πcall,t = Vcall × max (Pt −K, 0) −E [Vcall × max (Pt −K, 0)]
Put option
πput,t = Vput × max (K −Pt, 0) −E [Vput × max (K −Pt, 0)]
If for example, the electricity spot price (Pt) is above the strike price of the options
(K) there is a positive payoﬀof the call option, while the payoﬀof the put option is zero.
The payoﬀof the power plant, depends on the installed capacity of the plant (Vplant) and
its marginal cost (mc) and only the payoﬀof the retail contract depends on the stochastic
demand Vt. We subtract the expected value E(.) from the gross payoﬀall contracts/assets
to obtain a zero expected value. That is, we assume to be in a perfect and complete market
(no market power, no transaction costs, full transparency, etc.). Consequently, arbitrage
would not allow for the existence of systematic proﬁts.
Without this assumption, the method for the evaluation of contracts and assets would
drive our results. Indeed, the net loss calculated for each portfolio would be strongly
determined by the valuation method of the assets or contracts within each portfolio
2.2
Methodology of numerical simulations
The marginal generation cost of the power plant is set to the median of the simulated
spot prices mc Euro/MWh (second line of Table (2)), thus representing a peak load
power plant. The strike price of the options is set to the expectation value of the spot
price K = E[Pt] Euro/MWh (ﬁrst line of Table (2)).
We clearly see in Table 2, that all statistical indicators on a 10 year basis vary considerably depending on the cluster. For instance, the variance price for cluster 3am is

158.03 whereas it is 2790.30 for cluster 9am. In the same vein, the Mean price of cluster
3am is 24.11 whereas it is 57.99 for cluster 12am. This is related to the fact that electricity markets are hourly markets. Price and demand variability are on an hourly basis.
This hourly feature and the presence of price spikes justify an intra-day hedging approach
rather than a daily approach.
Table 2: Descriptive Statistics of the simulated data for each cluster hour
Clusters Hours
3am
6am
9am
12am
Mean price (E[Pt])
24,11
23,97
46,66
57,99
Median price (mc)
21,77
21,94
42,01
49,87
Mean load
46978,33
46970,76
57137,90
59106,19
Median load
45428,00
45383,00
55431,00
57793,00
Variance price
158,03
153,92
2790,30
4473,27
Variance load
36966692,94
37830907,83
41246907,38
28520369,27
Clusters Hours
3pm
6pm
9pm
12pm
Mean price (E[Pt])
48,50
44,08
45,17
35,76
Median price (mc)
42,52
39,33
40,52
32,99
Mean load
56482,52
54875,10
55260,57
53092,89
Median load
55659,00
52932,00
54308,00
51468,00
Variance price
1047,84
619,90
1268,30
252,82
Variance load
24607724,92
40756544,24
39911753,29
29013300,90
2.3
The risk minimization
We can calculate the cumulated annual payoﬀs of the N=3347 hourly price/volume combinations for all 2000 simulations given the portfolio (Vforward, Vplant, Vcall, Vput):
πi
N
X
t=1
πretail,t(P i
t , V i
t )
Vforward × πforward,t(P i
t )
Vplant × πplant,t(P i
t , mc)
Vcall × πcall,t(P i
t , K)
(2.1)
Vput × πput,t(P i
t , K)
Thus πi is the global payoﬀof the ith hourly price and volume simulation of a day

given the portfolio deﬁned by (Vforward, Vplant, Vcall, Vput). Using an optimization routine2,
the portfolio that produce the lowest VaR(95%) can be identiﬁed. As the routine does
not necessarily converges for this non-linear problem (especially for the three and four
assets case), we rerun the optimization for each case with 100 diﬀerent randomly drawn
starting values. The result of the best run can be considered suﬃciently close to the global
optimum, as all results tend to be within a fairly narrow range.
The objective is to ﬁnd the portfolio consisting of one 1 MWh baseload retail contract
and a linear combination of ﬁnancial contracts as well as physical assets that reduces
the retailers risk. Thus, the factors for the other contracts/assets are also measured in
MWh. The next Tables give the results given by two types of portfolios that maximize
the VaR(95%)
– portfolios containing one retail contract.
– portfolios containing one retail contract and diﬀerent power plants .
2.4
Optimization results
All hourly optimization results are given in Appendix (Tables 6 to 13). To present more
complete results, we give the corresponding Daily optimization results in Table 14.
As shown by Table 3, the simulations show that the optimal hedging varies considerably for each cluster.
A critical result of this Table is that this variation of optimal hedging strategy is not
only in terms of VaR or CVaR values (i.e. we obtain results in the range of −1615.38
to −676, 94 for the VaR and −2692, 99 to −954, 53 for the CVaR) but also in terms of
hedging portfolio: 5 (resp. 4) out of 8 optimal portfolios for the VaR (resp. CVaR) criteria
are composed by a combination of a forward contract and 3 powerplants.
Remark 2.1. The complementarity and the non-correlation between the payoﬀand the
risk level of a forward and 3 diﬀerent powerplants (baseload, semi-peak and peak) portfolio
enable more ﬂexibility given the hourly variability of electricity demand.
2We proceed under constrained nonlinear optimization or nonlinear programming using the function
fmincon in Matlab.

Table 3: Optimal hedging portfolio for each cluster hour, and for a day. The values of
the corresponding VaR and CVaR are also given.
VaR
CVaR
Hour
Optimal Hedging Portfolio
Value
Optimal Hedging Portfolio
Value
3am
Forward and 3 plants
-676,94
Forward and 3 plants
-954,53
6am
All possible contracts
-782,23
Only forward
-1073,72
9am
Forward and Vplant,75
-1615,48
Without options
-2692,99
12am
Forward and 3 plants
-1449,12
Vplant,25 and Vplant,75
-2499,38
3pm
Forward and 3 plants
-1353,29
Forward and 3 plants
-2295,76
6pm
Vplant,25 and Vplant,75
-1496,32
Vplant,25 and Vplant,75
-1872,97
9pm
Forward and 3 plants
-1210,55
Forward and 3 plants
-1979,57
12pm
Forward and 3 plants
-943,84
Forward and 3 plants
-1687,96
Daily
Only Options
-16095,31
Forward and Vplant,75
-21917,63
Therefore, if a retailer is hedged on a daily basis given its liquidity or cost constraints, it
should at least choose this portfolio (i.e. forward contract and 3 powerplants) to minimize
its losses.

Figure 3: VaR values obtained by the optimal hedging portfolio for each cluster hour on
a ten years basis (in blue). Corresponding mean in red.
3am
6am
9am
12am
3pm
6pm
9pm
12pm
−1800
−1600
−1400
−1200
−1000
−800
−600
Cluster Hours
VaR 95%
Figure 4:
CVaR values obtained by the optimal hedging portfolio for each cluster hour
on a ten years basis (in blue). Corresponding mean in red.
3am
6am
9am
12am
3pm
6pm
9pm
12pm
−2800
−2600
−2400
−2200
−2000
−1800
−1600
−1400
−1200
−1000
−800
Cluster Hours
CVaR 95%

Moreover, we show that a daily hedging optimization is worst than any hourly hedging
optimization (we obtain a VaR of −14102, 12 and a CVaR of −21917, 63). This implies
that intra-day hedging portfolios are much more appropriate than single daily portfolios
to manage joint volumetric and price risks on electricity markets.
Conﬁrming on a 10 years period and on an hourly basis, one of the results in Boroumand
and Zachmann (2012), a single forward hedging is not only never optimal but also ineﬃ-
cient given that electricity demand is not constant. Table 4 gives the increasing loss using
a single forward hedging instead of the optimal hedging portfolio given in Table 3.
Table 4: Increasing diﬀerential loss between the single forward hedging portfolio and
optimal hedging one given in Table 3.
Hour
Increasing loss in percentage
VaR
CVaR
3am
105,64%
6,37%
6am
102,22 %
0,00%
9am
61,72%
5,71%
12am
27,81%
22,45%
3pm
21,97%
18,19%
6pm
106,35 %
59,12%
9pm
116,92%
11,75%
12pm
35,80%
10,56%
Daily
46.87%
24.48%
Indeed, forward hedging is not relevant within markets where demand is stochastic
and correlated to the spot price.
Over a decade (2001-2011), our results show that the losses of an optimal daily portfolio
are ten times higher for the VaR criteria (resp.
nine times higher for the the CVaR
criteria) than the losses of any optimal intra-day portfolio. We obtain for the optimal
daily hedging portfolio a VaR value of −16095, 31 (resp. a CVaR value of −21917, 63)
against, −1615, 48, for the worst one in cluster hour optimization (9am). (resp. −2692, 99
for the worst one again in cluster hour optimization (9am).
2.4.1
In and out of the money case
An interesting extension of our hedging portfolio optimization is to test the case of in and
out of the money option. We run our optimization process for the cluster hour 6pm (peak

demand) with diﬀerent strike values for the call option. As mentioned in Section 2.2, the
strike price of the options is set to the expectation value of the spot price K = E[Pt]
Euro/MWh. Thus, regarding the ﬁrst line of Table (2) for the cluster hour 6pm, we have
a value of at the money strike equals to K = 44, 08 euros. We take a range of strike price
values of −10 to +10 of K with step of 5.
Table 5: Optimal VaR obtained with respect to the strike K + α of the call option
Values of α
Portfolio
All possible contracts
-1842,64
-1842,64
-1757,36
-1633,56
-1467,77
Only options
-1928,39
-1848,05
-1760,97
-1633,56
-1467,77
The more a call option is in the money the higher is its intrinsic value. Thus, the spread
between all possible contracts and only options portfolio increases. To the contrary, this
spread vanishes in the out of money case.
Conclusion an Policy recommendations
Our article contributes to the literature on electricity retailers risk hedging. We simulate
optimal intra-day portfolios given that electricity markets are hourly markets. First, we
demonstrate that the optimal hedging strategy diﬀers depending on the cluster hour with
respect to VaR and CVaR risk indicators. Secondly, we prove the signiﬁcantly superior ef-
ﬁciency of intra-day hedging portfolios over daily (therefore weekly and yearly) portfolios.
Over a decade (2001-2011), our results clearly show that the losses of an optimal daily
portfolio are at least nine times higher than the losses of optimal intra-day portfolios.
A clear understanding of risk management strategies within electricity markets is crucial
for market players, energy regulators, and ﬁnancial investors. Without appropriate risk
management instruments, the contribution of electricity retail markets to the global performance of the electricity industry will remain uncertain. We believe that this article
contributes to a better understanding of risk management issues in electricity markets.
The challenge for energy regulators is to enhance the liquidity of risk management instruments such as intra-day options. A relevant research extension is to propose a dynamic
framework for hedging strategies with distinct and/or additional ﬁnancial derivatives.

Acknowledgments
The authors would like to thank the participants of the ICCM January 2014 workshop,
specially Professor Helyette Geman, Professor Derek Bunn, and Professor Ehud I. Ronn.
The authors are also grateful to the members of the Department of Economics, City
University London for their suggestions.
