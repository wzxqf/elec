# Aggarwal 等 - 2009 - Electricity price forecasting in deregulated markets A review and evaluation

## Metadata

- source_pdf: 参考文献/Aggarwal 等 - 2009 - Electricity price forecasting in deregulated markets A review and evaluation.pdf
- extraction_method: pymupdf
- extraction_status: partial
- title: 
- doi: 

## Abstract



## Body

1. Introduction
Under restructuring of electric power industry, different participants namely generation companies and consumers of electricity
need to meet in a marketplace to decide on the electricity price
[1]. In the current deregulated scenario, the forecasting of electricity demand and price has emerged as one of the major research
ﬁelds in electrical engineering [2]. A lot of researchers and academicians are engaged in the activity of developing tools and algorithms for load and price forecasting. Whereas, load forecasting
has reached advanced stage of development and load forecasting
algorithms with mean absolute percentage error (MAPE) below
3% are available [3,4], price-forecasting techniques, which are
being applied, are still in their early stages of maturity. In actual
electricity markets, price curve exhibits considerably richer structure than load curve [5] and has the following characteristics: high
frequency, nonconstant mean and variance, multiple seasonality,
calendar effect, high level of volatility and high percentage of unusual price movements. All these characteristics can be attributed to
the following reasons, which distinguish electricity from other
commodities: (i) non-storable nature of electrical energy, (ii) the
requirement of maintaining constant balance between demand
and supply, (iii) inelastic nature of demand over short time period,
and (iv) oligopolistic generation side. In addition to these, market
equilibrium is also inﬂuenced by both load and generation side
uncertainties [2]. Therefore price-forecasting tools are essential
for all market participants for their survival under new deregulated
environment. Even accurate load forecasts cannot guarantee profits and the market risk due to trading is considerable because of extreme volatility of electricity prices.
Seeing the importance of price forecasting, authors felt that
there is a need of comprehensive survey at one place so that future
researchers in this area can easily get information about the current state of the research. Although, a few attempts have already
been made in this direction [5–9], but only qualitative aspects of
price forecasting have been addressed, and there is no such work
that has presented the quantitative analysis of the papers published so far. Authors of [5] documented the key issues of electricity price modeling and forecasting, and reviewed the price models
adapted from ﬁnancial assets. Importance of the price-forecasting
problem, key issues and some techniques developed so far have
been reported in [6]. Various price-forecasting techniques from input and output variables perspective have been discussed and
comparison of results of ﬁve different techniques has been presented in [7]. Time series forecasting procedures have been discussed in [8]. An overview of price-forecasting papers and
general procedure for price forecasting has been presented in [9].
So an overall assessment of the price-forecasting algorithms is still
required.
In order to investigate the state of price-forecasting methodologies, a review of 47 papers published during 1997 to November
2006, has been done based on the following parameters: (i) type
of model, (ii) time horizon for prediction, (iii) input variables
used, (iv) output variables, (v) analysis of results, (vi) data points
used for analysis, (vii) preprocessing employed, and (viii) model
0142-0615/$ - see front matter  2008 Elsevier Ltd. All rights reserved.
doi:10.1016/j.ijepes.2008.09.003
* Corresponding author. Tel.: +91 9416366091; fax: +91 1744 238050.
E-mail addresses: vasusanjeev@yahoo.co.in (S.K. Aggarwal), lmsaini@rediffmail.
com (L.M. Saini), ashwa_ks@yahoo.co.in (A. Kumar).
1 Tel.: +91 9416828819/9416137773.
Electrical Power and Energy Systems 31 (2009) 13–22
Contents lists available at ScienceDirect
Electrical Power and Energy Systems
journal homepage: www.elsevier.com/locate/ijepes

architecture. All the papers have been classiﬁed in three main
categories. It has been observed that forecasting errors are still
high from risk management perspective and test results are
difﬁcult to compare with each other. Most of the electricity
markets are at the stage of infancy, so the researchers have to
make predictive analysis based on the small data set available with
them. Little work has been done in the direction of price spike prediction. All the work has been quantiﬁed and put in the form of
tables.
This paper is organized as follows: In Section 2, a short introduction to price-forecasting methodologies is given. In Section 3,
factors affecting electricity prices, as considered by different
authors in their respective models, have been classiﬁed and discussed. The various features of the time series and causal models
have been outlined in Section 4. Artiﬁcial intelligence (AI) based
methods have been elaborated in Section 5. Data-mining models
are discussed in Section 6. Section 7 deals with locational marginal
price (LMP) forecasting models. Section 8 deals with the different
techniques as they are applied by the researchers to different electricity markets. Discussion and key issues are given in Section 9.
Section 10 concludes the review.
2. Price-forecasting methodologies
Numerous methods have been developed for electricity price
forecasting and most of these algorithms are same as used for load
forecasting and especially short-term load forecasting (STLF). Time
horizon varies from hour ahead to a week ahead forecasting. The
price-forecasting models have been classiﬁed in three sets [10]
and these three sets have been further divided into subsets as
shown in Fig. 1.
2.1. Game theory models
The ﬁrst group of models is based on game theory. It is of great
interest to model the strategies (or gaming) of the market participants and identify solution of those games. Since participants in
oligopolistic electricity markets shift their bidding curves from
their actual marginal costs in order to maximize their proﬁts,
these models involve the mathematical solution of these games
and price evolution can be considered as the outcome of a power
transaction game. In this group of models, equilibrium models
take the analysis of strategic market equilibrium as a key point.
There are several equilibrium models available like Nash equilibrium, Cournot model, Bertrand model, and supply function equilibrium model. Study of game theory models in itself is a major
area of research and has been kept outside the scope of this paper. A detailed discussion on game theory models can be found
in [11].
2.2. Simulation models
These models form the second class of price-forecasting techniques, where an exact model of the system is built, and the solution
is
found
using
algorithms
that
consider
the
physical
phenomenon that governs the process. Then, based on the model
and the procedure, the simulation method establishes mathematical models and solves them for price forecasting. Price forecasting
by simulation methods mimics the actual dispatch with system
operating requirements and constraints. It intends to solve a security constrained optimal power ﬂow (SCOPF) with the entire system range. Two kinds of simulation models have been analyzed
in this paper. One is market assessment and portfolio strategies
(MAPS) algorithm developed by GE Power Systems Energy Consulting [12] and the other is UPLAN software developed by LCG Consulting [13].
MAPS is used to capture hour-by-hour market dynamics while
simulating the transmission constraints on the power system. Inputs to MAPS are detailed load, transmission and generation units’
data. Where as the outputs are complete unit dispatch information,
LMP prices at generator buses, load buses and transmission ﬂow
information. UPLAN, a structural multi-commodity, multi-area
optimal power ﬂow (MMOPF) type model, performs Monte Carlo
simulation to take into account all major price drivers. UPLAN is
used to forecast electricity prices and to simulate the participants’
behavior in the energy and other electricity markets like ancillary
service market, emission allowance market. The inputs to MMOPF
are competitive bidding behavior, generation units’ data, the transmission network data, hydrological conditions, fuel prices and demand forecasts. These are almost comparable to the input
variables of MAPS. The outputs are forecast of prices and their
probability distribution across different energy markets. The dynamic effect of drivers on market behavior has also been captured.
Both UPLAN and MAPS may be used for long as well as short range
planning.
Simulation methods are intended to provide detailed insights
into system prices. However, these methods suffer from two drawbacks. First, they require detailed system operation data and second, simulation methods are complicated to implement and their
computational cost is very high.
Electricity Price Forecasting
Models
Game Theory
Models
Time Series
Models
Simulation Models
Parsimonious
Stochastic Models
Artificial Intelligence
based models
Regression or Causal
Models
Neural Network
based Models
Data-mining Models
Fig. 1. Classiﬁcation of price-forecasting models.
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

2.3. Time series models
Time series analysis is a method of forecasting which focuses on
the past behavior of the dependent variable [14]. Sometimes exogenous variables can also be included within a time series framework. Based on time series, there are further three types of models.
2.3.1. Parsimonious stochastic models
Many stochastic models are inspired by the ﬁnancial literature
and a desire to adapt some of the well known and widely applied
in practice approaches. In this paper, univariate discrete type models like autoregressive (AR), moving average (MA), autoregressive
moving average (ARMA), autoregressive integrated moving average (ARIMA), and generalized autoregressive conditional heteroskedastic (GARCH) have been considered. These are discrete time
counterparts corresponding to the continuous-time stochastic
models. Purely ﬁnance-inspired stochastic models involving certain characteristics of electricity prices, like price spikes and mean
reversion, have been kept outside the scope of this review. A discussion on these models can be seen in [5].
Stochastic time series can be divided into stationary process
and non-stationary process. The basic assumption of stationarity
on the error terms includes zero mean and constant variance. In
AR, MA and ARMA models conditions of stationarity are satisﬁed;
therefore they are applicable only to stationary series. ARIMA model tries to capture the incremental evolution in the price instead of
the price value. By the use of a difference operator, transformation
of a non-stationary process into a stationary process is performed.
The class of models where the constant variance assumption does
not need to hold is named heteroskedastic. Thus GARCH model
considers the conditional variance as time dependent. In all these
models price is expressed in terms of its history and a white noise
process. If other variables are affecting the value of price, the effect
of these variables can be accounted for using multivariate models
like TF (transfer function) and ARMA with exogenous variables
(ARMAX) models. As electricity price is a non-stationary process,
which exhibits daily, weekly, yearly and other periodicities. Therefore, a different class of models that have this property, designated
as seasonal process model, is used.
2.3.2. Regression or causal models
Regression type forecasting model is based on the theorized
relationship between a dependent variable (electricity price) and
a number of independent variables that are known or can be estimated [15]. The price is modeled as a function of some exogenous
variables. The explanatory variables of this model are identiﬁed on
the basis of correlation analysis on each of these independent variables with the price (dependent) variable.
2.3.3. Artiﬁcial intelligence (AI) models
These may be considered as nonparametric models that map
the input–output relationship without exploring the underlying
process. It is considered that AI models have the ability to learn
complex and nonlinear relationships that are difﬁcult to model
with conventional models. These models can be further divided
into two categories: (i) artiﬁcial neural network (ANN) based models and (ii) data-mining models.
2.3.3.1. ANN based models. ANNs are able to capture the autocorrelation structure in a time series even if the underlying law
governing the series is unknown or too complex to describe.
Since quantitative forecasting is based on extracting patterns
from observed past events and extrapolating them into the future, thus ANN may be assumed to be good candidates for this
task [16]. The available NN models are: (i) multilayer feed forward NN (FFNN), (ii) radial basis function network (RBF), (iii)
support vector machine (SVM), (iv) self-organizing map (SOM),
(v) committee machine of NNs, and (vi) recurrent neural network
(RNN).
2.3.3.2. Data-mining models. Recently, data-mining techniques like
Bayesian categorization method, closest k-neighborhood categorization, reasoning based categorization, genetic algorithm (GA)
based categorization, have gained popularity for data interpretation and inferencing. All those models using data-mining techniques have been covered in the category of data-mining models
in this work.
3. Factors inﬂuencing electricity prices
The factors inﬂuencing spot prices may be classiﬁed on the basis
of: C1 – market characteristics, C2 – nonstrategic uncertainties, C3
– other stochastic uncertainties, C4 – behavior indices, and C5 –
temporal effects. The different input variables, along with the class
they belong to, used by different researchers are presented in
Table 1. There are as many as 40 variables used by different
researchers. Most of the researchers have utilized past experience
in selecting the input variables for their respective model and
choice of best input variables for a particular model is still an open
area of research.
The widely used input variable is the electricity price of previous days. Researchers have used as much as past 1–7, 14, 21, 28,
Table 1
Factors inﬂuencing electricity prices
Class
Input variable
Time period whose data is used as input
C1
(1) Historical load
f(load); (d  m, t), m = 1, 2, 3, 4, 7, 14, 21, 28
(2) System load rate, (3) imports/exports, (4) capacity excess/shortfall
(d, t), (d, t  1), (d  1, t), (d  2, t), (d  7, t)
(5) Historical reserves
(d, t  2), (d, t  1), (d, t)
(6) Nuclear, (7) thermal, (8) hydro generation, (9) generation capacity, (10) net-tie ﬂows, (11) MRR, (12)
system’s binding constraints, (13) line limits
(d, t)
(14) Past MCQ (market-clearing quantity)
(d  1, t)
C2
(15) Forecast load
(d, t  2), (d, t  1), (d, t)
(16) Forecast reserves, (17) temperature, (18) dew point temperature, (19) weather, (20) oil price, (21) gas
price, (22) fuel price
(d, t)
C3
(23) Generation outages, (24) line status, (25) line contingency information, (26) congestion index
(d, t)
C4
(27) Historical prices
f(price); (d  m, t  n), m = 0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 21,
28, 364 and n = 0, 1, 2, 3, 4.
(28) Demand elasticity, (29) bidding strategies, (30) spike existence index, (31) ID ﬂag
(d, t)
C5
(32) Settlement period, (33) day type, (34) month, (35) holiday code, (36) Xmas code, (37) clock change,
(38) season, (39) summer index, (40) winter index
(d, t)
C1 – market characteristics, C2 – nonstrategic uncertainties, C3 – other stochastic uncertainties, C4 – behavior indices, C5 – temporal effects, d – day, t – settlement period
number of the day.
Note: The serial number of input variables given here are used in the input variable column of Tables 2 and 4 for respective input variables used by different researchers.
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

364 days price lags to capture the complete seasonal/calendar
variations namely daily, weekly and yearly variations. As price is
strongly correlated with demand, next most often used input variable is demand. Most authors have used the projected demand of
independent system operator (ISO), of the concerned electricity
market, as an input variable, a few have predicted the demand ﬁrst
and then used it as input variable for the price-forecasting model
[17,18]. Many researchers have also used historical load data as input variables. Authors of Ref. [19] have included functions of price
and forecasted load in their model. Instead of load, Li and Wang
[20] have used system load rate (SLR) as input variable so as to
take the effect of the rate at which load is changing on the output.
Capacity excess or surplus is total available capacity minus the required capacity at peak hour. This has been used by most of the
researchers as input variable, because it may affect the price signiﬁcantly in case surplus goes below certain threshold level and
thereby prompting some major participants to utilize this period
as an opportunity to exercise their market power. Since temperature is the main exogenous variable that affects the system load,
authors of Ref. [18,21,22] have used temperature as input variable
in their respective models. To take the effect of inﬂation and cost
of fuel prices on electricity price, fuel and oil prices have also been
used as input variables in [21–24]. Must run ratio (MRR) is the
generation concentration index (an indicator of oligopolistic
nature of the market), which has been used as an input variable
in [25,26] and has been shown to have considerable impact on
market price.
In order to understand the market state, instead of overall generation capacity, Gonzalez et al. [10] have used hourly production
capacity of different technologies like hydro, thermal and nuclear
as input variables and also reported the use of input variables like
participants’ pricing strategies, production costs, aggregated supply functions, generation companies’ shares etc. in their model
but no improvement in the accuracy was observed. ID ﬂag, used
in [27], is an indicator for presence of peak price (price volatility)
in the neighborhood of the predicted settlement period. Multiple
seasonalities related to daily, weekly, monthly and yearly periodicities have also been utilized as input variables as is evident from
the Table 1. Kian and Keyhani have used demand elasticity and
bidding strategies as explanatory variables of a regression-based
model [28].
4. Methodologies based on stochastic time series and causal
models
Twelve research papers can be covered in this category, three
are causal models [19,28,29] and nine are stochastic time series
models [30–38]. In Ref. [28], a regression-based model for electricity price was derived based on the assumption that power consumption and market prices are stochastic processes. Statistical
results for price model coefﬁcients were shown. Vucetic et al.
[19] have assumed a piece-wise stationary price time series having
multiple regimes with stable price–load relationship in each regime. These regimes, in the price series, were discovered with
the help of a regime discovery algorithm and price was modeled
by applying separate regression model for each regime. In Ref.
[29,35,36] the price series has been decomposed into detailed
and approximation parts using wavelet transform (WT). Future
behavior of decomposed series was predicted by applying appropriate model in the wavelet domain and ﬁnally inverse WT was
used to generate price prediction in the time domain.
In Ref. [29], second order regression polynomial of forecasted demand was applied to predict detailed components. In Ref. [35], the
future behavior of all the constitutive series has been predicted by
applying ARIMA to each of the series. In Ref. [36], both load and price
series were decomposed. Price and historical load data’s approximate part was used by multivariate time series for price approximate coefﬁcients prediction and price detail coefﬁcient part used
by univariate time series to forecast price detail coefﬁcients. Nogales et al. [30] have developed two models. The ﬁrst one was a dynamic regression (DR) model, which relates spot price to its own
lagged values and actual demand values. In the TF model, the relationship between price and demand has been established through
a TF term and a disturbance term that follows an ARMA process.
In Ref. [31], an ARIMA based forecasting model was presented. Cuaresma et al. [32] have demonstrated a comparison of the performance of 50 linear univariate time series AR and ARMA models.
Seasonal process ARIMA model has been proposed in Ref.
[33,34]. In both these papers, a stationary price time series has
been obtained by ﬁltering out non-periodic trend component and
periodical component and then price proﬁle has been predicted
by applying ARIMA. Further accuracy improvement has been
achieved through successive error correction method. In Ref.
Table 2
Main characteristics of time series models
Paper
Model type
Input variables (serial numbers as per
column 2 of Table 1)
Variable
segmentation
Preprocessing employed
Model identiﬁcation
and validation
Parameter
estimation
[29]
Second order
polynomial
27, 15
SS
WT
[32]
AR, ARMA
SS, 24 hourly
Series
[37]
ARMA, ARMAX,
AR, ARX
27, 1, 15
24 hourly
series
LT, Normalization
MLF
[38]
GARCH
27, 1
SS
LT
ACF, PACF
MLF
[30]
(1) DR, (2) TF
1, 27
SS
LT, outliers have been removed
ACF, PACF
MLF
[31]
ARIMA
1, 27, 8
SS
LT
ACF, PACF
MLF
[33]
Seasonal process
SS
Elimination of periodic and nonperiodic trend component
ACF, PACF
RA
[34]
Seasonal process
SS
Elimination of periodic and nonperiodic trend component
ACF, PACF
RA
[35]
ARIMA
SS
WT
ACF, PACF
MLF
[36]
Multivariate
ARMA
27, 15
SS
WT
[28]
MLR
17, 22, 27, 15, 9, 28, 29
SS
Statistical tools
LSE
[19]
Nonlinear
regression
27, 15, f(price), f(load)
SS
LSE
DR, dynamic regression; LT, log transformation; LSE, least square estimation; MLF, maximum likelihood function; MLR, multiple linear regression; RA, regression analysis; SS,
single series; TF, transfer function; WT, wavelet transform.
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

[34], in addition to price proﬁle, conﬁdence interval (CI) of price for
that period was also predicted. ARMA and ARMAX models were
used in [37]. In Ref. [38], univariate ARMA model with GARCH error
components was utilized and GARCH model with demand as exogenous variable was also developed.
Main characteristics of different time series models are given in
Table 2 and forecasting performance comparison has been presented in Table 3. It can be observed that log transformation of
the price time series has been adopted as a preprocessing technique in most of the stochastic time series models. This has been
done to obtain more stable variance. The idea of variable segmentation, i.e., framing the model as 24 separate hourly series, has been
applied in Ref. [32,37]. It has been observed that an hour-by-hour
modeling strategy for electricity spot prices improves signiﬁcantly
the forecasting abilities of linear univariate time series models
[32]. Autocorrelation function (ACF) and partial autocorrelation
function (PACF) are the preferred choice of researchers for model
identiﬁcation
and
estimation.
Maximum
likelihood
function
(MLF) is the most widely used parameter estimation technique.
5. Neural network-based models
In this category, 17 researchers have forecasted the price pro-
ﬁle, while six have made point prediction like maximum price or
average price and in one paper [39] the parameters of a chaos model have been forecasted. Authors of [17,40,41] have used 24 output
nodes and all other papers have used one output node. Information
for NN models is given in Tables 4–6. In Table 4, information
regarding model used, preprocessing employed and input variables
used by the different researchers has been presented. Forecasting
performance comparison has been given in Table 5. NN models’
architecture information and data used in different models, has
been compared in Table 6. It is evident from the Table 4 that the
FFNN architecture, which is also known as multilayer perceptron
(MLP), along with back propagation (BP) as the learning algorithm
is the most popular choice among researchers for a price-forecasting problem.
Authors of Ref. [42] initially reported the use of FFNN in price
proﬁle forecasting that tried 12 different combinations. In Ref.
[27], the raw price data was pre-processed by a front-end processor (based on fuzzy logic) representing the features of Saturday,
Sunday and public holidays. The predictor was a FFNN trained by
BP that predicted price proﬁles corresponding to weekends and
public holidays. Szkuta et al. have also used BP trained FFNN
[43]. Different FFNN models for weekdays and weekends were presented in [23]. Yao et al. [44] initially utilized WT for the decomposition of price and load data series into detailed and approximation
parts and then RBF network was used for predicting the approximate part and whereas, the detailed part was predicted by a
weighted average method. In Ref. [45], NN model was used for predicting the price and fuzzy model predicted price ranges using linear programming. Two models, applicable only to working days,
were compared in Ref. [41]. First one is a RNN model and second
is a k-weighted nearest neighbor (kWNN) algorithm based model,
which utilized a weighted-Euclidian norm to ﬁnd days, which are
nearest, in certain characteristics, to the forecast day. Genetic algorithm (GA) was used for estimating the weights corresponding to
different nearest neighbors. Zhang et al. [46] have implemented a
cascaded NN structure using a non-cascaded FFNN with the predicted input (load and weather) expressed as the measured input
plus an additional error term representing the associated uncertainties. Gaussian RBF networks approximate input–output relationships by building localized clusters and since unimportant
input factors may mislead local learning of RBF networks and
thereby poor generalization, therefore, a two-step training method
based on the inverses of standard deviations to identify and eliminate unimportant input factors was developed in Ref. [22].
Chaos theory was applied to construct a phase space from past
data of electric price and load in Ref. [47] and RNN was used for
prediction. Rodriguez and Anders [48] have proposed a hybrid of
NN and fuzzy logic known as adaptive-network-based fuzzy inference system (ANFIS) in which the output was obtained as a linear
combination of the input membership values of the input variables
and the inputs. An adaptively trained NN has been proposed whose
architecture can be changed during learning phase [40]. Input factors for the NN were obtained using a price simulation method.
Authors in [25,26] have used NN model for short-term forecasting
and a linear regression model for long term forecasting. The prediction of spot price was done in Ref. [39] using the method of nonlinear
auto-correlated
chaotic
model,
whose
parameters
were
predicted based on a wavelet NN (WNN) having hidden layer with
wavelet function. To overcome the inadequacy of a single network,
committee machine consisting of RBF and MLP networks has been
presented in Ref. [21]. Instead of simple averaging the outputs of
different networks, the method used the current input data and
the historical data to calculate weighting coefﬁcients, for combining predictions of different networks, in a weight calculator. Gonzalez et al. [10] have proposed a switching model based on the
input–output hidden Markov model (IOHMM) framework. The
model was based on the premises that each market may be represented by two states, one of them is hidden state, (characterized by
the interaction among resources, demand, and participants strategies), and the visible state, the power price series. NN has been
used to model state subnetwork and a dynamic regression process
of input variables has been used to implement output subnetwork.
Extended Kalman ﬁlter (EKF) learning has been used to train
MLP networks by treating weights of a network as the state of an
unforced nonlinear dynamic system in Ref. [24]. By ignoring the
interdependencies of mutually exclusive weights from different
neurons, a signiﬁcantly lower computational complexity and storage per training instance was achieved. Rough set theory (RST) has
been applied to the input data pattern in order to group and
combine the similar data patterns in Ref. [49] and the resulting
Table 3
Forecasting performance comparison of time series models
Paper
Data
used
(days)
Predicted period
Level of accuracy
Time
horizon
Output
[29]
1 week
DMAPE 2.5–11.11%
1 DA
PP
[32]
45 days
WMAE 3–7%.
1–7 DA
PP
[37]
4 weeks
WMAPE 3–11.1%
1 DA
PP
[38]
147, 105
12 months
WMAPE 9–11%
1 DA
PP
[30]
81, 135,
2 weeks, 1 week
DMAPE 3–5%
1 DA
PP
[31]
145, 85,
73, 92
3 and 11 weeks,
1 and 3 weeks
WMAPE 8–20%
(average 11%)
1 DA
PP
[33]
50, 50
2 sets of 10 days
MaxAE 1.21–4.36, 36–
1 DA
AvP
[34]
2 different days
DPE 1.5%. Hourly PE
0.1–5.23%.
1 h
ahead
PP, CI
[35]
4 weeks of 4
seasons
WMAPE 5–27%
1 DA
PP
[36]
1 week
DAPE min 0.1–5.3%,
DAPE max 52.2–98.7%
1 DA
PP
[28]
3 days
1, 2, 3
DA
PP
[19]
MSE 53.7–93.9, R2 0.8–
0.68
1 DA
PP
AE, absolute error; MAE, mean absolute error; DMAE, daily MAE; WMAE, weekly
MAE; MAPE, mean absolute percentage error; DMAPE, daily MAPE; WMAPE, weekly
MAPE; PE, percentage error; DPE, daily PE; APE, absolute percentage error; DAPE,
daily APE; MPE, mean percentage error; MSE, mean square error; RMSE, root mean
square error; AvP, average price; CI, conﬁdence interval; DA, day ahead; PP, price
proﬁle; R2, coefﬁcient of determination.
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

patterns were used to train the NN. A fuzzy neural network (FNN)
having higher learning capability has been proposed in Ref. [50]. In
FNN, the fuzzyﬁed classiﬁcation process (internal decomposition of
price series) of the input space has been performed in hidden layer
and defuzziﬁcation process in the single node of the output layer.
In Ref. [18], historical days that are similar in nature to a forecast
Table 4
Neural network models’ input variables and preprocessing employed
Paper
NN model
Learning
algorithm
Input variables (serial numbers as per
column 2 of Table 1)
Total number of input
factors
Preprocessing technique
[43]
MLP
BP
27, 15, 16, 32, 33, 35, 34, 37, 36
[48]
(1) MLP, (2)
FMLP
(i) BP, (ii) LM
15, 23, 4, 3
Outliers removed
[27]
MLP
BP
27, 15, 32, 31
9, 9, 6
Feature extraction for different days. SR [0,1]
[42]
MLP
BP
27, 15, 4, 32, 33
[23]
MLP
BP (CG)
27, 15, 3, 14, 22, 19, 32, 33, 38
SR [0,1], outliers removed
[51]
MLP
BP
27, 33, 34
Preprocessing using NN to forecast max, min,
medium values of prices
[17]
MLP
BP
27, 1, 15
[18]
MLP
BP
27, 1, 15, 17, 32, 33
Similar days data using Euclidian norm
[40]
MLP
BP
27, 1, 15, 5, 16, 32, 33, 24, 26, 13
Outliers removed
[25]
MLP
BP
27, 15, 11
[26]
MLP
BP
27, 15, 11
[41]
MLP
BP
[20]
MLP
AFSA
27, 2
WT, variable segmentation
[10]
MLP, DRM
IOHMM
27, 1, 15, 6, 7, 8
MLP – 5, DRM – 4
[49]
MLP
BP
27, 1
Similar patterns found using RST
[50]
MLP
GDR
[39]
MLP
BP
Noise ﬁltration using Fourier wave ﬁlter
[47]
RNN
27, 1, 15
[44]
RBF
27, 1, 33
WT, different model for each weekday
[45]
MLP
LM
SR [0.1–0.9]
[52]
MLP
BP
ACF
[21]
CM
27, 1, 15, 4, 17, 20, 21
RBF – 23, MLP – 55
SR [0,1]
[46]
MLP
BP, QN for CI
27, 1, 15, 4
[24]
MLP
EKF
27, 15, 4, 20, 21, 33, 35
[22]
RBF
2 stage training
15, 4, 17, 20, 21, 33
SR [0,1]
AFSA, artiﬁcial ﬁsh swarm algorithm; ACF, autocorrelation function; BP, back propagation (ﬁrst order gradient learning algorithm); CG, conjugate gradient; CI, conﬁdence
interval; CM, committee machine; DRM, dynamic regression model; EKF, extended Kalman ﬁlter; FMLP, fuzzy MLP; GDR, generalized delta rule; IOHMM, input–output
hidden Markov model; LM, Levenberg Marquardt algorithm; MLP, multilayer perceptron; QN, quasi-Newton; RBF, radial basis function; RST, rough set theory; SR, scaling
range; WT, wavelet transform.
Table 5
Forecasting performance comparison of neural network models
Paper
Output
Training data (days)
Predicted period
Time horizon
Level of accuracy
[43]
PP
1 week
1 time period ahead
Daily AvE 2.18–11.09
[48]
PP
14/28
1 day, 30 days
1 DA
DMAPE 20–38%.
[27]
PP
180, 180, 2
60 days, 60 days
1 DA
DMAPE 8.93–12.19%
[42]
PP
1 week
1 DA
Average DMAPE 11.57–12.86%
[23]
PP, QP
363, 404, 131
1 month
1 DA
DMAE 1.19–1.76 (training and validation only)
[51]
PP
1095, 730
2 sets of 2 days
1, 2, 3 DA
Error less than 1c€ in 85% cases
[17]
LP, PP
2 different weeks
1 DA
MAPE without spike 8.44%, with spike 15.87%
[18]
LP, PP
1 week, 1 month
1–6 h ahead
WMAPE 10.69–25.77%, monthly MAPE 9.75–20.03%.
[40]
PP, zonal PP, PDF
7–56
1 week
Short term
WMAPE 11–13%
[25]
PP, CI
1 week
1 DA
DMAPE 10–20%
[26]
PP, CI
1 week
1 DA
WMAPE 15.5%
[41]
PP
2 sets of 3 months
1 DA
RNN: AvPE 12–15%, kNN: AvPE 9–11%
[20]
PP
1 week
1 DA
DMAPE 3.5–5.16%
[10]
PP, PDF
1 week, 92 days
1 h ahead
WMAPE 15.83%
[49]
PP
1 month
1 h ahead
DMAPE 6.04%
[50]
PP
4 weeks of 4 seasons
1 DA
Average weekly MAPE = 7.5%
[39]
PCM, PP
10 days
1 time period ahead
APE 8%
[47]
PP
2 days
1, 25, 49 h ahead
DMPE 2.22–8%
[44]
PP
1 week
1 DA
Average AE 4–7.5%
[45]
Max Price, Range
1 DA
Overall RMSE 9.23%
[52]
AvP
228, 221, 214, 207, 144
7–91 days
m – ahead, m = 7, 14, 21, 28,
Average MAPE 8.22–9.12%
[21]
OPHAP
6 months
1 DA
Monthly MAPE 7.74–19.85%
[46]
OPHAP, C.I.
5 months
1 DA
Average monthly MAPE 8.8%, one-sigma CI coverage
66.6%
[24]
OPHAP, C.I.
11 and 2 months
1 DA
MAPE 11.1%, one-sigma CI coverage 68%.
[22]
OPHAP
12 months
1 DA
Average MAPE 11.9%
AvP, average price; AvE, average error; AvPE, average percentage error; DA, day ahead; LP, load proﬁle; OPHAP, on-peak hour average price; PP, price proﬁle; PCM, parameters
of chaotic model; PDF, probability density function; QP, quantity proﬁle; SDE, standard deviation of error; WMPE, weekly mean percentage error.
Note: Abbreviations of all accuracy criterion are same as Table 3.
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

day were identiﬁed based on a weighted-Euclidian norm method
and then a NN, trained with this similar days’ input data, forecast
the price by modifying the price curve obtained by averaging three
similar price days. A regression model has been applied to determine weighted factors in weighted-Euclidian norm method.
In Ref. [51], preprocessing was done to forecast maximum, minimum, medium values of the price using three auxiliary NNs and
then ﬁve principal NNs were used to forecast hourly prices.
Georgilakis [17] has reported an adaptively trained MLP-BP, in
which main NN predicted the hourly prices using forecasted load
information of an auxiliary NN. In Ref. [20], WT has been used to extract approximate price signals and then these signals fed to an arti-
ﬁcial ﬁsh swarm algorithm (AFSA) based NN to map inﬂuences of
nonlinear factors. In Ref. [52], ACF has been applied to the price
time series to ﬁnd out correlation between different periods of
the series and NN was used for price prediction. A moving cross validation method was employed for ﬁnding the best architecture of
the FFNN.
6. Data-mining models
Five papers have been considered in this section. Two working
day models have been proposed in Ref. [53]. One of them is kWNN
algorithm combined with GA and the other is dynamic regression
model in which least square estimation (LSE) method has been
used for estimation of coefﬁcients. A hybrid of Bayesian-based classiﬁcation and AR method that does not need any training has been
presented in [54]. In this, a clustering algorithm, other than the kmeans and the convergent k-means, has been used to predict output probability density function (PDF) and an AR model captures
the output change trend. Monthly MAPE in this method varies
from 9.96% to 13.69%. In Ref. [55], normal price has been predicted
using wavelet and NN based model and price spike using a datamining framework. Bayesian classiﬁcation and similarity searching
techniques have been used to mine the database. The preprocessing module separates the two price signals. The NN-wavelet module predicts the normal price and the possibility of price spikes at
speciﬁc occasions. If a speciﬁc occasion is forecasted to have price
spike then spike-forecasting module is activated. The range of price
spike can be predicted through data-mining techniques such as
categorization algorithm. The value of price spike is estimated
using k-nearest neighboring approach. Hourly PE (percentage error) varied from 1% to 31% in most of the test cases, whereas in
one case it was reported to be 49%. In [56], SVM and probability
based classiﬁcation algorithm was combined with normal priceforecasting method to determine probability of price spike. Then
a Bayesian classiﬁer was used for range of the forecasted price
spike and k-nearest neighboring approach for value of price spike.
A spike existence index was used as an input variable to utilize the
characteristic of price spikes that tend to occur together in a short
period. Forecasted hourly PE varies from 5.47% to 20% in most of
the cases, whereas in one case it is reported to be 49%. A hybrid numeric method that integrates a Bayesian statistical method and a
Bayesian expert (BE) has been proposed in Ref. [57] for price spike
prediction. Price series was classiﬁed into three classes of price
spikes, normal price and lower price using Bayesian classiﬁcation
approach and a BE combined with SVM has been used to forecast
electricity price spikes, normal price and lower price.
7. Methods for LMP prediction
In a power system, when the available least-cost energy cannot
be delivered to load in a transmission-constrained area, higher-cost
generation units have to be dispatched to meet that load. In this situation, the price of energy in the constrained area is higher than the
unconstrained market-clearing price (MCP). LMP is deﬁned as the
price of lowest-cost resources available to meet the load, subject
to delivery constraints of the physical network and is made up of
three components, (i) energy cost component, (ii) transmission
congestion component, and (iii) marginal loss component. The congestion and loss components are different for different locations
and the energy cost component is identical for all the nodes. For a
market, there is only one system marginal price (SMP), whereas,
there is an LMP involving the line ﬂow constraints and other security constraints at each node/area in a market. Three works pertaining to LMP forecasting have been considered [58–60]. In Ref. [58],
an EKF learning based NN model for forecasting zonal LMPs has
been reported. Congestion components have been estimated by
forecasting differences between zonal LMPs and hub LMPs. Quanti-
ﬁcation of transmission outages has been done based on a heuristic
method to feed into NN as input. On-peak average day-ahead and
on-peak real-time LMPs have been predicted in log form. Overall
MAPE varies from 8% to 20%. A fuzzy reasoning and RNN based
method has been proposed in Ref. [59]. Quantiﬁcation of contingencies has been done based on fuzzy rules to feed into NN as input in
the form of a variable called lmp (ratio of the LMP over the hub
LMP). Three different RNN models for weekday, Saturday and Sunday have been used. In [58,59], one transaction period ahead LMP
for an area has been predicted. In [60], a Fuzzy-c-means (FCM)
and RNN based method has been used. FCM is used to classify the
transaction periods into three clusters according to load levels:
peak, medium and off-peak load. In total nine RNNs were developed
to forecast nine different combinations of three clusters according
to load and three classes based on type of day.
8. Forecasting methodologies from electricity markets
perspective
Researchers have developed various forecasting tools covering
most of the deregulated markets. In [61], model has not been
Table 6
Neural network models’ architecture
Paper
No. of neurons
Activation
function
No. of parameters
Settlement
periods
[43]
[15-15-1]
[48]
[(1,2)-(4,8,12)-(1)]
S/L, FMF/L
ANN – 13 to 49,
ANFIS – 24
[27]
[9-7-4-1], [9-7-4-1], [6-
4-1]
S/S
107, 107, 33
[42]
[12-8-5-1]
S/S
[23]
[4-6-2]
TS/L
[51]
[1-2:4-1:14], [10-3-1]
[17]
[72-15-24], [48-15-24]
1479, 1119
[18]
[5/10-*-1], [4/9-*-1]
[40]
[25-40-24], [73-100-
24], [121-150-24]
2024, 9824,
21924
[25]
[3-2-1]
[26]
[3-2-1]
[41]
[24-24-24]
[20]
[10]
[5-*-1]
S/L
[49]
[50]
[19-19-1]
FC/L
[39]
WF/*
[47]
[44]
[12-*-1]
GRBF
[45]
[7-7-1]
L/L
[52]
[4-7-1]
S/L
[21]
6 clusters, [55-8-1]
283, 457
[46]
[56-8-1]
[24]
[50-*-1]
[22]
6 clusters
, not reported; DA, day ahead; FC, fuzziﬁed classiﬁed function; FMF, fuzzy membership function; GRBF, Gaussian radial basis function; L, linear function; S, sigmoid
function; TS, tan sigmoid; WF, wavelet function.
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

applied to any market but the work is signiﬁcant because it explores the possibility of signal-processing techniques (Fourier
and Hartley transform) as a preprocessing and ﬁltering tool to
bring out hidden patterns in the price signal. Authors of Ref.
[30,31,38,58] have tested their models on more than one electricity
market, whereas, others have conﬁned themselves to only one
market. An analysis of power prices across 14 different electricity
markets has been outlined in Ref. [62] and shown how price
evolution is different in different markets and therefore large
variations exist in price-forecasting accuracy achieved by different
models across different electricity markets.
Spanish electricity market [63], PJM [64] and New England electricity market [65] are the markets, which have caught the attention of most of the researchers. These are based on standard
market design (SMD) structure, which is basically a two-settlement market comprising a day-ahead market and a real-time intraday market. Most of the statistical models have been applied on
market data from these markets. Whereas; Ontario electricity market [66] and National electricity market of Australia (NEM) [67] follow a single settlement real-time structure. Only a few researchers
have applied their models on data from these markets. Apart from
that, California electricity market [68] is also one of the largely
studied markets in the world for the well-known problems that
it faced in the second half of 2000. The information regarding
status of research in different electricity markets is presented in
Table 7.
9. A discussion and key issues in designing a price-forecasting
system
Designing a price-forecasting model is a complex task as is evident from the literature review presented in previous sections.
Variations in input variable selection, forecasting horizon, preprocessing to be used, model selection, parameter estimation and
accuracy assessment have been reported, but few guidelines to
help the new designer. The key issues involved in formulating a
price-forecasting problem are as follows:
The electricity markets are highly volatile in nature and the reasons for spot price volatility are: (i) at any particular point in time,
plants of different technologies with different heat rate curves are
in operation making the aggregate supply–price curve complex
and intraday variable in nature, (ii) oligopolistic supply side, and
(iii) inelastic nature of electricity demand over short term. Authors
in Ref. [10] have tried to incorporate the effect of ﬁrst reason on the
market’s hidden states by using generation levels of different technologies as input variables. In [25,26], MRR has been included as
an input variable to capture oligopolistic nature of the market.
Garcia et al. [38] have shown the effect of market volatility on
the performance of ARIMA and GARCH models, whereas; effect of
volatility on the performance of the other models has yet not been
reported adequately in literature.
The selection of input feature is a key issue for the success of
any forecasting technique. Principal component analysis (PCA),
correlation analysis, genetic algorithm (GA), sensitivity analysis,
spectrum analysis techniques can be used for this purpose. Authors
in Ref. [24,40,43] have performed sensitivity analysis to show the
effect of input variable variation on output. Ref. [48] has performed
feature selection using correlation analysis. An analytical method,
which can select minimum number of effective input features,
has yet not been reported.
Most of the time series models are univariate models, whereas
few models involving structural approach are available. The problem with the time series models is the assumption of stationarity,
whereas price series exhibits a high degree of non-stationarity. It
can be observed from Section 4 that, research is moving in the
direction of development of more sophisticated hybrid and nonstationary models involving some kind of preprocessing of data
in order to attain stable mean and variance for price series. Garcia
et al. [38] have developed a non-stationary time series model
based on GARCH with reasonable degree of success.
From risk management perspective, distribution of prices is
more important than the point prediction; six papers have covered
this aspect as shown in Tables 2 and 5.
There is no benchmark for checking the continued out-performance of a single model over other models. Most of the results reported
by
different
researchers
cannot
be
put
in
a
single
framework because of diversity in their presentations (Tables 3
and 5). Authors of [50] have compared the performance of FNN
price-forecasting model with 6 other models and proved its
superiority. Persistence method [17,40] and naïve method [35]
are some of the reported methods, which can be used as benchmark
for
testing
the
effectiveness
of
any
new
forecasting
methodology.
A price signal exhibits much richer structure that load series,
although signal-processing techniques, like WT, are good candidates for bringing out hidden patterns in price series after decomposing price series into better behaved signals, probability of loss
of valuable information remains. An analytical method for noise removal is yet to be reported.
No single available model has been applied across data from larger number of markets. There is little systematic evidence as yet
that one model may explain the behavior of price signal in different
electricity markets, which is an indicator of participants’ collective
response to uncertainties, on a consistent basis.
Four papers, [23,30,40,48] have presented the results of their
respective models with and without removing spikes and observed
that prediction quality was improved by removing the outliers. On
the other hand, price spike prediction is relatively a new and
important area because price spikes have the capacity to signiﬁ-
cantly affect the proﬁtability of both suppliers and customers. In
three papers [55–57], emphasis has been given to predict price
spike.
Most of the researchers have concentrated on price forecasting
in day-ahead markets following SMD structure and case for realtime electricity markets is relatively under investigated. There is
a need to make more research efforts in other markets as well; this
will help in interpreting and understanding the price evolution in
different electricity markets in a better perspective.
Table 7
Price-forecasting research and electricity markets
Serial
no.
Market
Total
papers
Paper no.
PJM electricity
market
[12,20,36,58–60]
California electricity
market
[13,17,19,23,28,30,31,33,34,37,38,40,45]
New England
electricity market
[21,22,24,46,47,54,58]
Ontario electricity
market
[48]
Spanish electricity
market
[10,30,31,35,38,41,50,53]
Victoria electricity
market, NEM
[19,43]
Queensland
electricity market
[55,56]
UK power pool
[27,29,42,44]
European energy
exchange (Leipzig)
[32,51,52]
Electricity markets of
China
[25,26,39,57]
Korean power
exchange
[34]
S.K. Aggarwal et al. / Electrical Power and Energy Systems 31 (2009) 13–22

10. Conclusions
An overview of different price-forecasting methodologies is presented and key issues have been analyzed. Quantiﬁcation of various features of research papers pertaining to price forecasting
has been done. Broadly short-term price-forecasting models concentrate on any of three output variables – average price, peak
price, and price proﬁle. Of these three types, price proﬁle forecasting is more common and has been reported across different electricity markets using different models. A mathematical model of
price Y(t) can be represented in the form of the following equation:
YðtÞ ¼ ypðtÞ þ yðtÞ þ ysðtÞ, where, yp(t) is a component which depends primarily on the time of day and on normal working conditions of load and supply pattern for the particular day. The term
y(t) is an additive residual term describing inﬂuences due to load
and supply pattern deviations from normal and random correlation
effects. Usually such effects are small compared to time of day
component when deviations are moderate from the normal conditions. The third component ys(t) depends upon the complex strategies adopted by the participants and the most difﬁcult to model.
Different price-forecasting methodologies can be categorized in
three major categories. Among these categories, a statistical model
tries to capture the effect of price-inﬂuencing factors on price by
analyzing the past data. Univariate models like ARIMA predict
the future price values based on only price series data itself,
whereas; multivariate linear models like DR, TF and nonlinear
models like ANN can consider the effect of exogenous variables
as well. The DR and TF algorithms have been found to be more
effective than the ARIMA models. Although DR and TF models have
also shown good performance over nonlinear models in some case
studies, but in some cases ANN models have given better results.
Moreover, recent variations achieved in ANN and ARIMA using fuzzy logic and WT hold promise as well. In conclusion, there is no
systematic evidence of out-performance of one model over the
other models on a consistent basis. This may be attributed to the
reason that history of electricity markets is relatively short and
large differences in price developments exist in different power
markets. It is hoped that with better computational tools at the disposal of researchers price evolution in electricity markets will be
better understood over a period of time.
