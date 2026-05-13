# records.jsonl 与源 PDF 抽样对应性检查

- 检查日期: 2026-05-13
- 检查对象: `文本提取/outputs/pdf_extract/jsonl/records.jsonl`
- 源 PDF 根目录: `参考文献/`
- 抽样方式: 固定随机种子 `20260513`，分层抽取 `success=8`、`partial=4`
- 记录总数: `53`；状态分布: `partial=16`, `success=37`
- 源 PDF 缺失: `0`
- 样本记录通过: `12/12`；片段命中: `30/30`

## 判定口径

本次检查重新读取每个样本的源 PDF 内置文字层，将 JSONL 中的摘要/正文片段做 Unicode 规范化、大小写统一并移除空白和标点后，在 PDF 全文文字层中定位。正文每篇最多检查前、中、后三个片段；摘要存在时检查摘要片段。若整段连续匹配失败，再用多个子片段窗口复核，以兼容提取清洗阶段删除页码行造成的非连续差异。

## 样本结果

| 序号 | JSONL行 | 状态 | 页数 | 片段命中 | 源 PDF |
|---:|---:|---|---:|---:|---|
| 1 | 2 | `partial` | 26 | 3/3 | `参考文献/Artzner 等 - 1999 - COHERENT MEASURES OF RISK.pdf` |
| 2 | 3 | `partial` | 12 | 3/3 | `参考文献/Bao - 2021 - Risk assessment and management of electricity markets A review with suggestions.pdf` |
| 3 | 5 | `partial` | 10 | 3/3 | `参考文献/Cabero 等 - 2005 - A Medium-Term Integrated Risk Management Model for a Hydrothermal Generation Company.pdf` |
| 4 | 17 | `partial` | 25 | 3/3 | `参考文献/Poli 等 - 2007 - Particle swarm optimization An overview.pdf` |
| 5 | 10 | `success` | 14 | 4/4 | `参考文献/Deng和Oren - 2006 - Electricity derivatives and risk management.pdf` |
| 6 | 26 | `success` | 6 | 2/2 | `参考文献/中国电力市场建设路径优选及设计_葛睿.pdf` |
| 7 | 29 | `success` | 7 | 2/2 | `参考文献/南方( 以广东起步) 电力现货市场机制设计.pdf` |
| 8 | 30 | `success` | 9 | 2/2 | `参考文献/售电公司购售电业务决策与风险评估_王林炎.pdf` |
| 9 | 34 | `success` | 11 | 2/2 | `参考文献/基于能源转型的中国特色电力市场建设的分析与思考_陈国平.pdf` |
| 10 | 38 | `success` | 10 | 2/2 | `参考文献/新电改背景下售电公司的购售电策略及风险评估_罗舒瀚.pdf` |
| 11 | 46 | `success` | 8 | 2/2 | `参考文献/电力市场中需求响应市场与需求响应项目研究_赵鸿图.pdf` |
| 12 | 50 | `success` | 10 | 2/2 | `参考文献/计及风险规避的售电公司平衡市场优化交易策略研究_杨萌.pdf` |

## 命中片段证据

### 1. Artzner 等 - 1999 - COHERENT MEASURES OF RISK.pdf
- JSONL 行: `2`；状态: `partial`；方法: `pymupdf`；PDF 页数: `26`
- 摘要片段: JSONL 中为空，未做摘要文字命中检查。
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`Address correspondence to P. Artzner, Institut de Recherche Math´ematique Avanc´ee, Universit´e Louis Pasteur,`
- 正文片段 2: 命中，页码 `11`，模式 `exact`；`p. 8133) and SEC rule 15c3-1a,(11) as requiring no margin (no “deduction”). No justiﬁcation is given for this speciﬁcation. We shall use the paper ...`
- 正文片段 3: 命中，页码 `24`，模式 `exact`；`For a dense set of random variables X on the ﬁnite state space Ä we have, by Proposition 5.3, the equality E [−X/r | X ≤q · r] = WCEα(X); hence the...`

### 2. Bao - 2021 - Risk assessment and management of electricity markets A review with suggestions.pdf
- JSONL 行: `3`；状态: `partial`；方法: `pymupdf`；PDF 页数: `12`
- 摘要片段: JSONL 中为空，未做摘要文字命中检查。
- 正文片段 1: 命中，页码 `11`，模式 `exact`；`Minglei Bao received the Ph.D. degree from Zhejiang University and B.S. degree in Electric Engineering from Shandong University in 2021 and`
- 正文片段 2: 命中，页码 `11`，模式 `exact`；`Hangzhou, China. His current research interests include power systems reliability/performance analysis incorporating renewable energy resources, smart`
- 正文片段 3: 命中，页码 `11`，模式 `exact`；`China. His research interests include reliability analysis, short-circuit current limitation, and the power`

### 3. Cabero 等 - 2005 - A Medium-Term Integrated Risk Management Model for a Hydrothermal Generation Company.pdf
- JSONL 行: `5`；状态: `partial`；方法: `pymupdf`；PDF 页数: `10`
- 摘要片段: JSONL 中为空，未做摘要文字命中检查。
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`proposed methodology includes three steps: the generation of scenarios for these random parameters, the approximation of these`
- 正文片段 2: 命中，页码 `4`，模式 `exact`；`of modeling approaches, Cournot model of equilibrium has received much attention. In this framework each generation company`
- 正文片段 3: 命中，页码 `10`，模式 `exact`；`The authors are grateful to the Iberdrola’s Analysis and Processes Unit (ANPRO) for their signiﬁcant support. We have also`

### 4. Poli 等 - 2007 - Particle swarm optimization An overview.pdf
- JSONL 行: `17`；状态: `partial`；方法: `pymupdf`；PDF 页数: `25`
- 摘要片段: JSONL 中为空，未做摘要文字命中检查。
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`Abstract Particle swarm optimization (PSO) has undergone many changes since its introduction in 1995. As researchers have learned about the techniq...`
- 正文片段 2: 命中，页码 `12`，模式 `exact`；`method, an individual changed its stage after 50 iterations with no improvement. The population was initialized as PSO particles; the “LifeCycle” a...`
- 正文片段 3: 命中，页码 `21`，模式 `exact`；`is quite illuminating in that sense. As show in Table 1, particle swarm optimization is exponentially growing.6 So, clearly, we are still looking a...`

### 5. Deng和Oren - 2006 - Electricity derivatives and risk management.pdf
- JSONL 行: `10`；状态: `success`；方法: `pymupdf`；PDF 页数: `14`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`attributes of electricity production and distribution. Uncontrolled exposure to market price risks can lead to`
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`Electricity spot prices are volatile due to the unique physical attributes of electricity such as nonstorability, uncertain and inelastic demand an...`
- 正文片段 2: 命中，页码 `5`，模式 `exact`；`replacing the underlying of a ﬁnancial option with electricity (see [9] for introduction to various kinds of`
- 正文片段 3: 命中，页码 `12`，模式 `exact`；`signals, providing price discovery, facilitating effective risk management, inducing capacity investments`

### 6. 中国电力市场建设路径优选及设计_葛睿.pdf
- JSONL 行: `26`；状态: `success`；方法: `pymupdf`；PDF 页数: `6`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`在新一轮电力体制改革中，电力市场建设是重中之重，设计一条合理的电力市场建设路径对 于从计划模式向市场模式平稳过渡、最终达成目标市场模式尤为重要。文中基于对中国电力系统 运营现状、电力市场基本模式、国外电力市场建设经验教训等的分析，建立了适应中国电力市场建 设路径优选的评价指标，在此基础上论证了...`
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`中国电力市场建设路径优选及设计 葛 睿１，陈龙翔１，王轶禹１，刘敦楠２ （１．国家电力调度控制中心，北京市１０００３１；２．新能源电力系统国家重点实验室（华北电力大学），北京市１０２２０６） 摘要：在新一轮电力体制改革中，电力市场建设是重中之重，设计一条合理的电力市场建设路径对 于从计划模式向...`

### 7. 南方( 以广东起步) 电力现货市场机制设计.pdf
- JSONL 行: `29`；状态: `success`；方法: `pymupdf`；PDF 页数: `7`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`遵循市场经济基本规律和电力工业运行客观规律，综合考虑广东电网的实际情况，本文设计了南方( 以广东起 步) 电力现货市场机制，提出了以基于差价合约的中长期交易规避风险和全电量集中竞争现货交易发现价格的电力市 场交易体系，设计了基于节点电价的两部制结算方法。将原有中长期物理交易变更为中长期差价合约...`
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`引言 2015 年3 月15 日，中共中央、国务院发布的 《关于进一步深化电力体制改革的若干意见》( 中发 〔2015〕9 号) ［1］宣告了新一轮电力体制改革的开始， 意见及其配套文件中提出构建有效竞争的电力市 场。在此基础上，国家发展改革委、国家能源局相 继发布了《电力中长期交易基本规则(...`

### 8. 售电公司购售电业务决策与风险评估_王林炎.pdf
- JSONL 行: `30`；状态: `success`；方法: `pymupdf`；PDF 页数: `9`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`购售电业务的决策和风险评估是售电公司适应电力市场的关键。为丰富售电公司购售电业 务，在中长期加现货交易的市场模式下，对售电公司可能的购售电业务进行分类，组合得到不同购 售电业务模式。采用场景法对模型中的现货价格和用电需求等风险随机变量进行模拟，以条件风 险利润作为风险评估指标，以售电公司风险和...`
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`ｈｔｔｐ：／／ｗｗｗ． ａｅｐｓ－ｉｎｆｏ． ｃｏｍ 售电公司购售电业务决策与风险评估 王林炎１，张粒子１，张 凡２，金东亚１ （１．华北电力大学电气与电子工程学院，北京市１０２２０６；２．国网能源研究院有限公司，北京市１０２２０９） 摘要：购售电业务的决策和风险评估是售电公司适应电力市场的关...`

### 9. 基于能源转型的中国特色电力市场建设的分析与思考_陈国平.pdf
- JSONL 行: `34`；状态: `success`；方法: `pymupdf`；PDF 页数: `11`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`With the continuous deepening of China’s energy clean transition and the acceleration of power market construction, it is necessary to combine the ...`
- 正文片段 1: 命中，页码 `1`，模式 `window 2/4`；`第40 卷 第2 期 中 国 电 机 工 程 学 报 Vol.40 No.2 Jan. 20, 2020 2020 年1 月20 日 Proceedings of the CSEE ©2020 Chin.Soc.for Elec.Eng. DOI：10.13334/j.0258-8013.pc...`

### 10. 新电改背景下售电公司的购售电策略及风险评估_罗舒瀚.pdf
- JSONL 行: `38`；状态: `success`；方法: `pymupdf`；PDF 页数: `10`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`随着我国电力体制改革的深入和能源利用技术的发 展，售电公司日渐成为多能量市场的主体。开展多时间尺度 购电、合理配置各类能源购电比例及提供差异化售电合同是 基金项目：国家电网公司科技项目(5217L017000M)。 Project Supported by Science and Techno...`
- 正文片段 1: 命中，页码 `5`，模式 `exact`；`Tab. 1 Distribution scale of energy purchasing in different energy markets under different risk aversion factor`

### 11. 电力市场中需求响应市场与需求响应项目研究_赵鸿图.pdf
- JSONL 行: `46`；状态: `success`；方法: `pymupdf`；PDF 页数: `8`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`The effective way to implement demand response is the demand response markets and demand response programs, and the drive is demand response benefi...`
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`(1. School of Computer Science and Technology, Henan Polytechnic University, Jiaozuo 454000, Henan Province, China;`

### 12. 计及风险规避的售电公司平衡市场优化交易策略研究_杨萌.pdf
- JSONL 行: `50`；状态: `success`；方法: `pymupdf`；PDF 页数: `10`
- 摘要片段 1: 命中，页码 `1`，模式 `exact`；`由于市场价格具有波动性和不确定性，售电公司在平 衡市场的直接交易面临较大风险。引入用户侧负荷作为平衡 资源，提出了包含可中断负荷/电量收购和关键负荷电价两 类需求响应项目参与的平衡市场优化交易策略，以规避市场 风险。采用条件风险价值度量交易策略风险，并建立了基于 随机规划的非线性数学模型。利用...`
- 正文片段 1: 命中，页码 `1`，模式 `exact`；`第40 卷 第11 期 电 网 技 术 Vol. 40 No. 11 2016 年11 月 Power System Technology Nov. 2016 文章编号：1000-3673（2016）11-3300-09 中图分类号：TM 73 文献标志码：A 学科代码：470·40 计及风险...`

## 结论

本次分层样本未发现 JSONL 与源 PDF 文字层错配、串文件或明显缺页问题。`partial` 样本的主要特征是摘要为空，正文片段仍可在源 PDF 中命中。
