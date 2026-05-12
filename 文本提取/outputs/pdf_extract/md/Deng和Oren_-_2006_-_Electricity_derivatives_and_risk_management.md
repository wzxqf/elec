# Deng和Oren - 2006 - Electricity derivatives and risk management

## Metadata

- source_pdf: 参考文献/Deng和Oren - 2006 - Electricity derivatives and risk management.pdf
- extraction_method: pymupdf
- extraction_status: success
- title: 
- doi: 

## Abstract

Electricity spot prices in the emerging power markets are volatile, a consequence of the unique physical
attributes of electricity production and distribution. Uncontrolled exposure to market price risks can lead to
devastating consequences for market participants in the restructured electricity industry. Lessons learned from the
ﬁnancial markets suggest that ﬁnancial derivatives, when well understood and properly utilized, are beneﬁcial to
the sharing and controlling of undesired risks through properly structured hedging strategies. We review different
types of electricity ﬁnancial instruments and the general methodology for utilizing and pricing such instruments. In
particular, we highlight the roles of these electricity derivatives in mitigating market risks and structuring hedging
strategies for generators, load serving entities, and power marketers in various risk management applications.
Finally, we conclude by pointing out the existing challenges in current electricity markets for increasing the
breadth, liquidity and use of electricity derivatives for achieving economic efﬁciency.
q 2005 Elsevier Ltd. All rights reserved.

## Body

1. Introduction
Electricity spot prices are volatile due to the unique physical attributes of electricity such as nonstorability, uncertain and inelastic demand and a steep supply function. Uncontrolled exposure to market
price risks could lead to devastating consequences. During the summer of 1998, wholesale power prices
in the Midwest of US surged to a stunning $7000 per MWh from the ormal price range of $30–$60 per
MWh, causing the defaults of two power marketers in the east coast. In February 2004, persistent high
prices in Texas during a 3-day ice storm led to the bankruptcy of a retail energy provider that was
exposed to spot market prices. And of course, the California electricity crisis of 2000/2001 and its
devastating economic consequences are largely attributed to the fact that the major utilities were not
properly hedged through long-term supply contracts. Such expensive lessons have raised the awareness
Energy 31 (2006) 940–953
www.elsevier.com/locate/energy
0360-5442/$ - see front matter q 2005 Elsevier Ltd. All rights reserved.
doi:10.1016/j.energy.2005.02.015
* Corresponding author. Tel.: C1 404 894 6519; fax: C1 404 894 2301.
E-mail address: deng@isye.gatech.edu (S.J. Deng).

of market participants to the importance and necessity of risk management practices in competitive
electricity market.
Hedging of risk by a corporation should in principle be motivated by the goal of maximizing ﬁrm’s
value. Hedging achieves value enhancement by reducing the likelihood of ﬁnancial distress and its
ensuing costs, or by reducing the variance of taxable incomes and its associated present value of future
tax liabilities. Regulatory rules also play an important role in hedging practices. In California, for
instance, the regulators granted the incumbent investor-owned utilities (IOUs) a ﬁxed time frame to
recover their stranded generation costs through the Competition Transition Charge. Fearing adverse
market conditions causing insufﬁcient recovery of the stranded costs, one major utility company hired
investment bankers to structure and implement an extensive hedging strategy for its stranded-cost
recovery. On the other hand, the reluctance of the regulators in California to immunize the IOUs against
ex-post prudence review of long-term supply contracts discouraged the adoption of such contracts,
resulting in over-reliance of the IOUs on the spot market for electricity procurement. This excessive
exposure led to the near collapse of the California utility industry in 2001, with devastating economic
losses due to prolonged outages and substantial rate increases.
As the competitive but volatile electricity markets mature, generation companies, power marketers
and load serving entities (LSEs) seek certainty in their costs and revenues through hedging practices and
contracting and active trading. Such activities involve quantifying, monitoring and controlling trading
risks in the wholesale and retail power markets, which in turn require appropriate risk management tools
and methodology.
On the supply side, managing risk associated with long-term investment in generation and
transmission requires methods and tools for planning under uncertainty and for asset valuation. Much of
the demands for generation asset valuation methods were spurred by the mandatory divestiture of
generation assets already owned by major utility companies in various jurisdictions. For example, in
California, most of the fossil-fuel plants held by the three IOUs, which account for about 60% of the total
installed capacity in California by 2000, have been or will be divested to other parties. The need for asset
valuation also rises from analysis of investment in new generation capacity and from efforts by
regulators in the US and abroad to develop incentives for investment in generation capacity to meet
supply adequacy and system reliability objectives.
A fundamental vision underlying the worldwide movement toward a competitive electricity industry
has been that most of the efﬁciency gains from restructuring come from long-run investments in
generating capacity. Under the state-ownership or required rate-of-return regulatory regime, utility
companies were allowed to earn a regulated rate of return above their cost of capital. Once regulators
approved the construction costs of a power generating plant, the costs would be passed onto consumers
through regulated electricity prices over the life of the investment, independent of the ﬂuctuation in
market value of the investment over time due to changing energy prices, improving technology, and
evolving supply and demand conditions. Most of the investment risks in generating capacity were
allocated to consumers rather than producers. Firms, therefore, had little incentives to avoid excessive
cost of investment and they focused on improving and maintaining quality of service rather than on
developing and adopting new generation technology.
Electricity market reforms around the world have shifted much of the investment risk from consumers
to producers. Under the ideal theoretical paradigm, shareholders bear all the investment risk and
consumers bear the price risk, with competitive entry pushing generation capacity toward desired longterm equilibrium. In such an ideal market environment, suppliers and consumers are free to choose their
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

desired level of risk exposure, achieved through voluntary risk management practices. Unfortunately,
this idealized vision of a competitive electricity market is not working as expected, primarily due to such
market imperfections as lack of demand response, abuse of locational market power, and political
resistance to high prices reﬂecting scarcity rents and shortages.
With few exceptions such as Australia (where electricity spot prices are allowed to rise to $10,000 per
MWh), most restructured electricity markets in the US and around the world have backed away from the
idealized economic market models and instituted price caps and various capacity payment mechanisms.
Such regulatory interventions allocate risks between consumers and producers by limiting price
volatility for consumers and assuring investment cost recovery for generators. From a risk management
perspective, these intervention schemes are mandatory backstop hedging that limits the exposures of
consumers and producers. The proper design of such schemes requires the same pricing and asset
valuation tools as voluntary risk management practices in a competitive market. For instance, a price cap
of $1000/MWh can be viewed as a mandatory call option imposed on all produced electricity with a
strike price of $1000/MWh, with the option premium being the proper capacity payment for generators
abiding by the cap.
The organization of the rest of the paper is as follows. Section 2 describes the institutional features of
several types of commonly traded electricity instruments. Section 3 highlights the essential elements in
electricity derivative pricing and introduces the pricing methodologies. Section 4 illustrates the roles of
these electricity instruments in risk management applications. Section 5 concludes.
2. Different types of electricity ﬁnancial and physical instruments
This section reviews various electricity ﬁnancial/physical instruments traded on the exchanges and
over the counters. Most of the electricity futures and options on futures are traded on the New York
Mercantile Exchange (NYMEX) [1]. However, the trading volume of electricity futures is less than
electricity forwards traded in the over-the-counter (OTC) markets. A large variety of electricity
derivatives are traded among market participants in the OTC markets, including forward contracts,
swaps, plain vanilla options, and exotic (i.e. non-standard) options like spark spread options, swing
options and swaptions [2–6]. Other important trading vehicles for hedging the price risk of long-term
revenue streams and service obligations are termed as structured transactions, including tolling
agreements [7,8] and load-serving full requirement contracts. The institutional details of these
instruments are given below.
2.1. Electricity forwards, futures and swaps
The plainest forms of electricity derivatives are forwards, futures and swaps. Being traded either on
the exchanges or over the counters, these power contracts play the primary roles in offering future price
discovery and price certainty to generators and LSEs.
2.1.1. Electricity forwards
Electricity forward contracts represent the obligation to buy or sell a ﬁxed amount of electricity at a
pre-speciﬁed contract price, known as the forward price, at certain time in the future (called maturity or
expiration time). In other words, electricity forwards are custom-tailored supply contracts between
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

a buyer and a seller, where the buyer is obligated to take power and the seller is obligated to supply. The
payoff of a forward contract promising to deliver one unit of electricity at price F at a future time T is
Payoff of a Forward Contract Z ðST KFÞ
(1)
where ST is the electricity spot price at time T. Although the payoff function (1) appears to be the same as
for any ﬁnancial forwards, electricity forwards differ from other ﬁnancial and commodity forward
contracts in that the underlying electricity is a different commodity at different times. The settlement
price ST is usually calculated based on the average price of electricity over the delivery period at the
maturity time T.
Consider a forward contract for the on-peak electricity on day T. ‘On-peak electricity’ refers to the
electricity delivered over the daily peak-period, traditionally deﬁned by the industry as 06:00–22:00.
The daily ‘off-peak’ period is the remaining hours of the day. In this case, ST is obtained by averaging the
16 hourly prices from 06:00 to 22:00 on day T.
Based on the delivery period during a day, electricity forwards can be categorized as forwards on
on-peak electricity, off-peak electricity, or ‘around-the-clock’ (24 h per day) electricity. As almost all
electricity derivatives have such categorization based on the delivery time of a day, we will not repeat
this point.
Generators such as independent power producers (IPPs) are the natural sellers (or, short-side) of
electricity forwards while LSEs such as utility companies often appear as the buyers (or, long-side). The
maturity of an electricity forward contract ranges from hours to years although contracts with maturity
beyond two years are not liquidly traded. Some electricity forwards are purely ﬁnancial contracts, which
are settled through ﬁnancial payments based on certain market price index at maturity, while the rest are
physical contracts as they are settled through physical delivery of underlying electricity. Examples of
ﬁnancially settled electricity forwards include the Contract for Differences in the United Kingdom and
Australian power markets.
Electricity forwards with short maturity like 1 h or 1 day are often physical contracts, traded in the
physical electricity markets such as the Pennsylvania–New Jersey–Maryland (PJM) power pool market
and the energy balancing market operated by the California Independent System Operator (CAISO) in
US. Those with maturity of weeks or months can be either physical contracts or ﬁnancial contracts and
they are mostly traded through brokers or directly among market participants (namely, traded in the OTC
markets).
Electricity forward contracts are the primary instruments used in electricity price risk management.
LSEs (e.g. local distribution companies) typically combine several months of forward/futures contracts
to form a close match to the long-term load shape of their customers. Other power marketers usually use
forwards to hedge their positions in electricity options and other complex electricity derivatives.
2.1.2. Electricity futures
First traded on the NYMEX in March 1996, electricity futures contracts have the same
payoff structure as electricity forwards. However, electricity futures contracts, like other ﬁnancial
futures contracts, are highly standardized in contract speciﬁcations, trading locations, transaction
requirements, and settlement procedures. The most notable difference between the speciﬁcations of
electricity futures and those of forwards is the quantity of power to be delivered. The delivery quantity
speciﬁed in electricity futures contracts is often signiﬁcantly smaller than that in forward contracts.
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

For example, a Mid-Columbia electricity futures traded on the NYMEX speciﬁes a delivery quantity of
432 MWh of ﬁrm electricity, delivered to the Mid-Columbia hub at a rate of 1 MW per hour, 16 on-peak
hours per day during delivery month, while a corresponding forward contract has a delivery rate of
25 MW per hour for the same delivery periods in a month.
Electricity futures are exclusively traded on the organized exchanges, while electricity forwards are
usually traded over-the-counter in the form of bilateral transactions. This fact makes the futures prices
more reﬂective of higher market consensus and transparency than the forward prices. The majority of
electricity futures contracts are settled by ﬁnancial payments rather than physical delivery, which lower
the transaction costs. In addition, credit risks and monitoring costs in trading futures are much lower than
those in trading forwards, since exchanges implement strict margin requirements to ensure ﬁnancial
performance of all trading parties. The OTC transactions are vulnerable to ﬁnancial non-performance
due to counterparty defaults. The fact that the gains and losses of electricity futures are paid out daily, as
opposed to being cumulated and paid out in a lump sum at maturity time, as in trading forwards, also
reduces the credit risks in futures trading.
In summary, as compared to electricity forwards, the advantages of electricity futures lie in market
consensus, price transparency, trading liquidity, and reduced transaction and monitoring costs while the
limitations stem from the various basis risks associated with the rigidity in futures speciﬁcation and the
limited transaction quantities speciﬁed in the contracts.
2.1.3. Electricity swap
Electricity swaps are ﬁnancial contracts that enable their holders to pay a ﬁxed price for underlying
electricity, regardless of the ﬂoating electricity price, or vice versa, over the contracted time period. They
are typically established for a ﬁxed quantity of power referenced to a variable spot price at either a
generator’s or a consumer’s location. Electricity swaps are widely used in providing short- to mediumterm price certainty up to a couple of years. They can be viewed as a strip of electricity forwards with
multiple settlement dates and identical forward price for each settlement.
Electricity locational basis swaps are also commonly used to lock in a ﬁxed price at a geographic
location that is different from the delivery point of a futures contract. That is, a holder of an electricity
locational basis swap agrees to either pay or receive the difference between a speciﬁed futures contract
price and another locational spot price of interest for a ﬁxed constant cash ﬂow at the time of the
transaction. These swaps are effective ﬁnancial instruments for hedging the basis risk on the price
difference between power prices at two different physical locations.
2.2. Electricity options
The power industry had been utilizing the idea of options through embedded terms and conditions in
various supply and purchase contracts for decades, without explicitly recognizing and valuing the
options until the beginning of the electricity industry restructuring in UK, US and the Nordic countries in
the 1990s. The emergence of the electricity wholesale markets and the dissemination of option pricing
and risk management techniques have created electricity options not only based on the underlying price
attribute (as in the case with plain vanilla electricity call and put options), but also other attributes like
volume, delivery location and timing, quality, and fuel type.
Basically, a counterpart of each ﬁnancial option can be created in the domain of electricity options by
replacing the underlying of a ﬁnancial option with electricity (see [9] for introduction to various kinds of
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

ﬁnancial options). Here, we describe a sample of electricity options that are commonly utilized in risk
management applications in generation and distribution sectors. These options usually have short- to
medium maturity times such as months or a couple of years. Options with maturity times longer than
3 years are usually embedded in long-term supply or purchase contracts, which are termed as structured
transactions.
2.2.1. Plain call and put options
Electricity call and put options offer their purchasers the right, but not the obligation, to buy or sell a
ﬁxed amount of underlying electricity at a pre-speciﬁed strike price by the option expiration time. They
have similar payoff structures as those of regular call and put options on ﬁnancial securities and other
commodities. The payoff of an electricity call option is
Payoff of an electricity call option Z maxðST KK; 0Þ
(2)
where ST is the electricity spot price at time T and K is the strike price.
The underlying of electricity call and put options can be exchange-traded electricity futures or
physical electricity delivered at major power transmission inter-ties, like the ones located at California–
Oregon Border and Palo Verde in the Western US power grid. The majority of the transactions for
electricity call and put options occur in the OTC markets. Electricity call and put options are the
most effective tools available to merchant power plants and power marketers for hedging price risk
because electricity generation capacities can be essentially viewed as call options on electricity,
particularly when generation costs are ﬁxed.
2.2.2. Spark spread options
An important class of non-standard electricity options is the spark spread option (or, spark spread).
Spark spreads are cross-commodity options paying out the difference between the price of electricity
sold by generators and the price of the fuels used to generate it. The amount offuel that a generation asset
requires to produce one unit of electricity depends on the asset’s fuel efﬁciency or heat rate (Btu/kWh).
The holder of a European- spark spread call option written on fuel G at a ﬁxed heat rate KH has the right,
but not the obligation, to pay at the option’s maturity KH times the fuel price at maturity time T and
receive the price of one unit of electricity. Thus, the payoff at maturity time T is
Payoff of a spark spread call Z maxðST KKH !GT; 0Þ
(3)
where ST and GT are the electricity and fuel prices at time T, respectively.
Abstracting away the operational characteristics of a fossil fueled power generator (e.g. startup cost
and ramping constraints), the per kW beneﬁt of owning the right to use the generator is equivalent to
having 1 kW spark spread call option with a strike heat rate matching the generator’s operating heat rate.
Based on this observation, it is clear that spark spread call options play important roles in hedging the
price risk of the output electricity of fossil fueled power plants and further serve as key instruments in
valuing those generation assets [10,11].
2.2.3. Callable and putable forwards
Two interesting types of electricity derivatives termed as callable forward and putable forward are
introduced in Refs. [12,13] to mimic the interruptible supply contracts and the dispatchable independent
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

power producer contracts. In a callable forward contract, the purchaser of the contract longs one forward
contract and shorts one call option with a purchaser-selected strike price. The seller of the forward
contract holds opposite positions and can exercise the call option if the electricity price exceeds the
strike price, effectively canceling the forward contract at the time of delivery. The purchaser gets an
‘interruptibility’ discount on the forward price, which is equal to the option premium at the time of
contracting continuously compounded to the delivery time.
In a putable forward, the purchaser longs one forward contract and one put option with a sellerselected strike price. The seller holds the corresponding short positions. The purchaser exercises the put
option if the electricity price drops below the strike price at the maturity time, effectively canceling the
forward contract. At the time of contracting, the purchaser needs to pay a ‘capacity availability’
premium over the forward energy price, which equals the put option price at that time, continuously
compounded to the maturity time.
One variation of the callable forwards is proposed by adding an earlier notiﬁcation date for exercising
the call option in a callable forward before the contract matures [14,15]. This emulates an interruptible
service contract with early notiﬁcation [16].
2.2.4. Swing options
Electricity swing options are adopted from their well-known counterparts in the natural gas industry
[5]. Also known as ﬂexible nomination options, swing options have the following deﬁning features.
First, these options may be exercised daily or up to a limited number of days during the period in which
exercise is allowed. Second, when exercising a swing option, the daily quantity may vary (or, swing)
between a minimum daily volume and a maximum volume. However, the total quantity taken during a
time period such as a week or a month needs to be within certain minimum and maximum volume levels.
Third, the strike price of a swing option may be either ﬁxed throughout its life or set at the beginning of
each time period based on some pre-speciﬁed formula. Last, if the minimum-take quantity of any
contract period is missed by the buyer, then a lump sum penalty or a payment making up the seller’s
revenue shortfall needs to be paid (i.e. take-or-pay).
2.3. Structured transactions
Structured bilateral transactions are powerful tools for power market participants to share and control
a variety of risks including price and quantity risks over a potentially long time horizon.
2.3.1. Tolling contracts
Tolling is one of the most innovative structured transactions embraced by the power industry. A
tolling agreement is similar to a common electricity supply contract signed between a buyer (e.g. a
power marketer) and an owner of a power plant (e.g. an IPP) but with notable differences. For an upfront
premium paid to the plant owner, it gives the buyer the right to either operate and control the scheduling
the power plant with the ISO or simply take the output electricity during pre-speciﬁed time periods
subject to certain constraints. In addition to inherent operational constraints of the underlying power
plant, there are often other contractual limitations in the contract on how the buyer may operate the
power plant or take the output electricity. For instance, a tolling contract almost always has a clause on
the maximum allowable number of power plant restarts. These constraints make the pricing of tolling
contracts a very challenging task. The analogy between holding a tolling contract and owning
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

the underlying merchant power plant, however, leads to a numerical approach for valuing and hedging
tolling contracts [7]. Alternatively, one may use a statistical approach for benchmarking the price
reasonableness of tolling contracts based on historical electricity price and fuel costs [8].
2.3.2. Load-serving full-requirement contracts
Most large electricity consumers prefer a power supply contract with ﬂexible consumption terms.
Speciﬁcally, they desire to pay a ﬁxed rate per unit of energy for the actual consumption quantity,
regardless of the quantity being high or low. Such a contract is termed as a load-serving full-requirement
contract.
Suppose an electricity supplier (or, LSE) signs a full-requirement contract with a customer and then
utilizes futures contracts to lock in a ﬁxed quantity of electricity supply at a ﬁxed cost for hedging the
expected energy consumption of the customer [17,18]. The LSE is then at the risk of either under- or
over-hedging, as the consumption quantity of the customer will almost surely deviate from the amount
hedged by the futures contracts. When the electricity spot price is high (low), the total demand for
electricity is likely to be high (low) as well. A case in point is the periods of unusual cooling/heating
needs. Hence, if the market price of electricity is higher than the ﬁxed contract rate for serving
electricity, chances are that the customer’s energy consumption level is signiﬁcantly higher than the
hedged quantity. As a result, the LSE is under-hedged relative to its load obligation and must purchase
electricity in the open market to serve its customer at a loss because the wholesale spot price most likely
exceeds the contracted price paid by consumers. Conversely, when the electricity spot price is low, the
LSE faces the risk of being over-hedged and having to sell the surplus in the spot market or settle it
ﬁnancially at a price below its long-term contract price.
The above illustrates the under- and over-hedging exposures faced by an LSE due to the volumetric
uncertainty in customers’ load and the positive price-load correlation. To hedge the volumetric risk, the
LSE would need to buy an electricity option on the consumption quantity of its customers.
Unfortunately, such an option is usually unavailable in the marketplace. Although perfect hedging may
not be possible, weather derivatives [19,20] that exploit the correlation between load and temperature
can be used. Section 4.4 describes another approach based on an optimal hedging portfolio of standard
derivatives that exploits the positive correlation between power prices and consumption quantity [21].
2.4. Financial derivatives on electricity transmission capacity
Open access to, efﬁcient utilization of, and adequate investment in transmission networks are critical
for the electricity wholesale markets and retail competitions to be workable and efﬁcient. Intuitively,
rights are required for using transmission networks and rules are needed for rationing transmission usage
when networks become congested. There are two major proposals for using ﬁnancial instruments as
transmission rights in US: (a) the point-to-point ﬁnancial transmission rights (FTRs) [22–24]; and (b) the
ﬂowgate rights (FGRs) [25,26], as outlined in the Standard Market Design (SMD) put forth by the
Federal Energy Regulatory Commission (FERC). FTRs and FGRs are electricity derivatives, with their
values derived from the network transmission capacity.
2.4.1. FTR and FTR options
In an electricity market such as the PJM that employs locational market price (LMP), a point-topoint FTR is speciﬁed over any two locations in the power transmission grid. An FTR entitles its
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

holder to receive compensation (or pay) for transmission congestion charges that arise when the
grid is congested. The congestion charge/payment (or, payoff) associated with one unit of FTR is
equal to the difference between the two locational prices of one unit of electricity resulting from
the re-dispatch of generators out of merit order to relieve transmission congestion. The primary
markets for the FTR trading are auctions held by the independent system operators (ISOs) of power
markets.
An FTR option offers the right to the FTR settlement without the obligation to pay when that
settlement is negative. Hence, the settlement of an FTR option equals to the positive part of the
corresponding two-sided point-to-point FTR.
2.4.2. FGRs
Flowgates are deﬁned over all transmission elements such as lines, transformers, or linear
combinations of them. Each transmission element has two elemental ﬂowgates, one in each direction.
An elemental ﬂowgate has a rated capacity in megawatts in its pre-speciﬁed direction corresponding to
the capacity of an underlying transmission element. Thus, ﬂowgate rights are link-based transmission
rights for hedging transmission risks. The values of ﬂowgate rights can be established through auctions
conducted by the ISOs. The spot price upon which the settlement of ﬂowgate rights is based is given by
the real time shadow price on the corresponding constrained element, determined by the security
constrained economic dispatch algorithm employed by an ISO. Since these shadow prices are
nonnegative, FGRs are inherently deﬁned as options.
3. Pricing electricity derivatives
Since the value of electricity derivatives are based on the underlying electricity prices, modeling
electricity price is the most critical component in pricing electricity derivatives. Due to the unique
physical and operational characteristics of electricity production and transmission processes, electricity
price exhibits different behaviors than other ﬁnancial prices which can be often described by Geometric
Brownian Motion. There has been a growing literature addressing mainly two competing approaches to
the problem of modeling electricity price processes:
(a) ‘Fundamental approach’ that relies on simulation of system and market operation to arrive at market
prices; and
(b) ‘Technical approach’ that attempts to model directly the stochastic behavior of market prices from
historical data and statistical analysis.
While the ﬁrst approach provides more realistic system and transmission network modeling
under speciﬁc scenarios, it is computationally prohibitive due to the large number of scenarios that
must be considered. Such analysis may be necessary for pricing ﬁnancial transmission rights (in
particular, ﬂowgate rights) but not for the other electricity derivatives. Therefore, we shall focus
our attentions on the second approach and review the corresponding methodologies for pricing
electricity derivatives.
Approaches to characterize market prices include discrete-time time series models such as GARCH
and its variants [27–32], Markov regime-switching models [33], continuous-time diffusion models such
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

as mean-reversion [11,34,35], jump-diffusion [2,3,36], and other diffusion models [37,38]. There are
also models proposed for direct modeling of electricity forward curves [39,40].
While a straightforward application of the maximum likelihood estimation (MLE) method yields the
parameter estimates of a discrete-time time series model, it does not yield analytic expressions for
derivative prices. In fact, Monte Carlo simulation and lattice-based approaches are the only feasible
derivative pricing methods under time-series price models. For continuous-time diffusion models, model
parameters can be estimated by applying moment-based methods, such as the generalized method of
moments, which may not be as efﬁcient as the MLE method. Nonetheless, more option pricing methods
(e.g. the analytic solution approach and the partial differential equation (PDE) approach) become
applicable under the diffusion price models.
Deng [3] was the ﬁrst to employ a multifactor afﬁne jump diffusion (AJD) processes to model
electricity spot prices under several speciﬁcations, including regime switching and stochastic volatility.
Under the assumption that electricity prices follow AJD processes, an extended Fourier transform
technique developed in Ref. [41] can be applied to derive analytic expressions (up to Fourier inversion)
for a variety of derivative prices. Speciﬁcally, prices of forwards, calls/puts and spark spreads were
derived in Ref. [3] under three different electricity price models, and prices of callable forwards with an
early notiﬁcation were obtained in Ref. [14].
When there is a large set of market data available, the most appropriate approach to pricing
electricity options is to infer the risk-neutral distribution of the underlying electricity price from the
market data and then obtain the prices of the electricity derivatives based on the premise of noarbitrage. If there is not enough forward-looking market information for implementing a noarbitrage pricing model, then equilibrium models can be applied to obtain derivative prices, as in
Refs. [31,34,40,42,43] for forward prices and [44] for spark spreads. In certain cases, statistical
benchmark analysis based on historical data can provide a sense of the reasonableness on the
electricity options prices [8].
The binomial/multinomial lattice and Monte Carlo simulation methods are powerful numerical tools
for pricing electricity options with complex structures and/or under a complicated model for the
electricity price process. For instance, given the complex structure of a swing option or a tolling contract,
it is impossible to obtain prices of such contracts either in closed-forms or through PDEs. Thus, swing
options are priced by lattice models [45,46], or by approximation methods for obtaining price lower
bounds [47]. The pricing of tolling contracts requires a combination of Monte Carlo simulation with
dynamic programming [7].
4. Risk management applications
4.1. Hedging a generator’s output
Albeit having simple payoff structures, forwards, swaps, and call options are effective tools for a
generator with ﬁxed per unit cost to lock in proﬁts by selling forwards, ﬁxed-price swaps, and call
options on electricity. When the forward/swap rate or the strike price of the call options is higher than the
ﬁxed cost, the generator’s proﬁts are guaranteed.
However, if the generating costs are market-based (e.g. a natural gas ﬁred merchant power
plant that burns natural gas at market price), the selling forwards, swaps and calls will expose
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

the generator to potential fuel cost increases. In such a case, a properly constructed portfolio of
spark spread calls would be the right tool for hedging a generator’s revenue stream over a given
time period.
The operational efﬁciency of a natural gas ﬁred power plant is characterized by its operating heat rate.
Therefore, the ﬁnancial beneﬁt of owning a portfolio of spark spread calls with strike heat rates identical
to the operating heat rate of the plant is the same as owning the power plant during the time period of the
options’ maturity times. This observation leads to the valuation and hedging method for generation
capacity proposed in Refs. [10,11]. When taking into account the operational characteristics, latticebased method [48] and simulation method [35] are necessary to determine pricing and hedging strategies
of generation capacity.
In the case, where the electricity forward market at the generator’s location is not liquidly traded,
electricity forwards from adjacent trading hubs or even forwards on the input fuel, which are liquidly
traded, can be utilized to cross-hedge the electricity output price [49,50].
4.2. Ensuring generation adequacy
Oren [51,52] and Chao and Wilson [53] propose a new role for options with long maturity to address
the resource adequacy problem. They propose a scheme for ensuring generation adequacy via call
options as obligations imposed on the LSEs. Call options provide an attractive alternative to artiﬁcial
capacity products such as installed capacity (ICAP) employed in New York, New England, and PJM,
whose demand is based only on administrative requirements and which have no intrinsic value. By
requiring LSEs to purchase a proper portfolio of options, a regulator can achieve spot price volatility
reduction by implementing price insurance while using the premium to stabilize generators’ income and
enhance investment incentives.
4.3. Callable forwards and interruptible service contracts
The restructured electricity markets have shown little demand response to price spikes. The
enormous price volatility afﬁrms the need for demand responsiveness to make these markets workable.
As load curtailment can provide an efﬁcient substitute for generation capacity in meeting balancing
energy and reserves needs, ﬂexible loads are viable and valuable resources in taming price volatility.
Consider the traditional utility interruptible service contracts utilized in demand-side management
(DSM) to mitigate supply shortages. These interruptible contracts are readily implementable
through standard electricity derivatives [12–14]. For instance, a synthetic interruptible service
contract offered by an LSE is a callable forward under which the LSE sells a forward to and buys
a call option from its customer. Furthermore, with a liquid electricity derivative market, the
discounts offered to the interrupted services would be set through market trading instead of bilateral
negotiations thus making the pricing of the interruptible services more transparent and efﬁcient.
4.4. Hedging congestion risk of bilateral transactions
From the perspective of new power network transmission users, FTRs can be viewed as an
instrument for hedging their exposure to congestion cost risk. A 1-MW bilateral transaction between
two points in a transmission network is charged (or credited) the nodal price difference between
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

the point of withdrawal and the point of injection. At the same time (assuming that transmission rights
are fully funded), a 1 MW FTR between two points is an entitlement (or obligation) for the difference
between the nodal prices at the withdrawal node and the injection node. Thus regardless of how the
system is dispatched, a 1 MW FTR between two nodes is a perfect hedge against the uncertain
congestion charge between the same two nodes.
The hedging properties of FTRs make them ideal instruments for converting historical entitlements
to ﬁrm transmission capacity into tradable entitlements that hold the owners of such entitlements
harmless, while enabling them to cash out when someone else can make more efﬁcient use of the
transmission capacity covered by these entitlements. In other words, FTRs make it relatively easy to
preserve the status quo while opening up the transmission system to new and more efﬁcient use. A
word of caution is that the hedging function of FTRs may not be perfect due to changing network
operating conditions and potential inherent trading inefﬁciency [54]. Some ISOs derate FTR
settlements in order to cover congestion revenue shortfalls due to transmission contingencies not
accounted for in the FTR auction. In such cases, depending on the derating approach, FTRs may not
provide perfect hedges either.
4.5. Hedging volumetric risks
LSEs providing electricity service at regulated prices in restructured electricity markets are wary of
both price and quantity risks [17,18]. As the electricity markets are inherently incomplete, the quantity
risk cannot be perfectly hedged. Commonly proposed hedging alternatives include the implementation
of a minimal variance hedge through purchasing electricity forwards [18] and the utilization of weather
derivatives.
Recent work reported in Ref. [21] addresses the problem of hedging volumetric risks by risk-averse
LSEs, whose hedging objective is to maximize a concave utility function. Exploiting the correlation
between consumption quantities and spot prices, the authors developed an optimal, zero-cost hedging
function described by a payoff function of spot price. They also demonstrate how such a hedging strategy
can be implemented through a portfolio of standard forwards and a spectrum of call and put options with
various strike prices.
5. Conclusion
In electricity market restructuring, electricity derivatives play an important role in establishing price
signals, providing price discovery, facilitating effective risk management, inducing capacity investments
in generation and transmission, and enabling capital formation. Custom design of electricity ﬁnancial
instruments and structured transactions can provide energy price certainty, hedge volumetric risk,
synthesize generation and transmission capacity, and implement interruptible service contracts.
Admittedly, many exotic forms of electricity options can meet speciﬁc needs for hedging and
speculation. However, we emphasize the importance of standardization. Future research should focus on
identifying standardized electricity derivatives and utilization of ﬁnancial engineering tools to
synthesize and replicate alternative contracts using standardized instruments. Such standardization
will reduce transaction costs and produce liquidity, which in turn will improve the efﬁciency of risk
management practices.
S.J. Deng, S.S. Oren / Energy 31 (2006) 940–953

Acknowledgements
This research is supported in part by the NSF Grant ECS-0134210 (Deng), National Science
Foundation Grant ECS-0224779 (Oren) and a Power System Engineering Research Center (PSERC)
grant (Deng and Oren).
