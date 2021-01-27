# Quick References for Fairness Measures

##  Fairness Metrics <a id = "metric_quickref"></a>

| Category | Metric | Definition | Weakness | References |
|------|------|------|------|------|
| Group Fairness |**Demographic Parity**| A model has **Demographic Parity** if the predicted positive rates (selection rates) are approximately the same for all protected attribute groups.<br> $$\dfrac{P(\hat{y} = 1 \lvert unprivileged)} {P(\hat{y} = 1 \rvert privileged)}$$ | Historical biases present in the data are not addressed and may still bias the model. | [Zafar *et al* (2017)](#zafar2017_ref) |
||**Equalized Odds**| Odds are equalized if $P(+)$ is approximately the same for all protected attribute groups.<br>  **Equal Opportunity** is a special case of equalized odds specifying that $$P(+ \rvert y = 1)$$ is approximately the same across groups. | Historical biases present in the data  are not addressed and may still bias the model. | [Hardt *et al* (2016)](#hardt2016_ref) |
||**Predictive Parity**| This parity exists where the Positive Predictive Value and Negative Predictive Value are each approximately the same for all protected attribute groups. | Historical biases present in the data are not addressed and may still bias the model.  | [Zafar *et al* (2017)](#zafar2017_ref) |
||||||
| Similarity-Based Measures |**Individual Fairness**| Individual fairness exists if "similar" individuals (ignoring the protected attribute) are likely to have similar predictions. | The appropriate metric for similarity may be ambiguous. |[Dwork (2012)](#dwork2012_ref), [Zemel (2013)](#zemel2013_ref), [Kim *et al* (2018)](#kim2018_ref) |
| &nbsp; |**Unawareness** | A model is unaware if the protected attribute is not used. | Removal of a protected attribute may be ineffectual due to the presence of proxy features highly correlated with the protected attribute.| [Zemel *et al* (2013)](#zemel2013_ref), [Barocas and Selbst (2016)](#barocas2016_ref) |
||||||
| Causal Reasoning |**Counterfactual Fairness** \*| Counterfactual fairness exists where counterfactual replacement of the protected attribute does not significantly alter predictive performance. This counterfactual change must be propogated to correlated variables. | It may be intractable to develop a counterfactual model.  | [Russell *et al* (2017)](#russell2017_ref) |
||||||

<br><br>
## Interpretations of Common Measures <a id = "measure_quickref"></a>
| Metric | Measure | Equation | Interpretation |
|:----|:----:|:----:|:----|
|**General Measures**|Base Rate| $$\sum_{i = 0}^N(y_i)/N$$  | - |
| |Selection Rate| $$\sum_{i = 0}^N(\hat{y}_i)/N$$ | - |
|**Group Fairness Measures**|Demographic (Statistical) Parity Difference| $$P(\hat{y} = 1\ \left\lvert\ unprivileged) - P(\hat{y} = 1\ \right\rvert\ privileged) $$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
| |Disparate Impact Ratio (Demographic Parity Ratio)| $$\dfrac{P(\hat{y} = 1\ \rvert\ unprivileged)}{P(\hat{y} = 1\ \rvert\ privileged)} = \dfrac{selection\_rate(\hat{y}_{unprivileged})}{selection\_rate(\hat{y}_{privileged})}$$ | 1 indicates fairness <br>  < 1 favors privileged group <br>  > 1 favors unprivileged group |
| |Positive Rate Difference| $$ precision(\hat{y}_{unprivileged}) - precision(\hat{y}_{unprivileged})$$ |
| |Average Odds Difference| $$\dfrac{(FPR_{unprivileged} - FPR_{privileged}) + (TPR_{unprivileged} - TPR_{privileged})}{2}$$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
| |Average Odds Error| $$\dfrac{\left\lvert FPR_{unprivileged} - FPR_{privileged}\right\rvert + \left\lvert TPR_{unprivileged} - TPR_{privileged}\right\rvert}{2}$$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
| |Equal Opportunity Difference| $$recall(\hat{y}_{unprivileged}) - recall(\hat{y}_{privileged})$$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
| |Equalized Odds Difference| $$max( (FPR_{unprivileged} - FPR_{privileged}), (TPR_{unprivileged} - TPR_{privileged}) )$$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
| |Equalized Odds Ratio| $$min( \dfrac{FPR_{smaller}}{FPR_{larger}}, \dfrac{TPR_{smaller}}{TPR_{larger}} )$$ | 1 indicates fairness <br>  < 1 favors privileged group <br>  > 1 favors unprivileged group |
|**Individual Fairness Measures**|Consistency Score| $$ 1 - \frac{1}{n\cdot\text{n_neighbors}}\sum_{i = 1}^n |\hat{y}_i - \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|$$ | 1 indicates consistency <br> 0 indicates inconsistency |
| |Generalized Entropy Index| $$ GE = \mathcal{E}(\alpha) = \begin{cases} \frac{1}{n \alpha (\alpha-1)}\sum_{i = 1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\ \frac{1}{n}\sum_{i = 1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha = 1,\\ -\frac{1}{n}\sum_{i = 1}^n\ln\frac{b_{i}}{\mu},& \alpha = 0. \end{cases} $$ | - |
| |Generalized Entropy Error| $$GE(\hat{y}_i - y_i + 1) $$ | - |
| |Between-Group Generalized Entropy Error| $$GE( [N_{unprivileged}*mean(Error_{unprivileged}), N_{privileged}*mean(Error_{privileged})] ) $$ | 0 indicates fairness<br>(+) indicates unfairness |


<br><br>
|Metric |Statistical Criteria |Definition |Description |
|-|-|-|-|
|Demographic Parity|Statistical Independence |R ⊥ G |sensitive attributes (A) are statistically independent of the prediction result (R) |
|Equalized Odds| Statistical Separation |R ⊥ A\|Y |sensitive attributes (A) are statistically independent of the prediction result (R) given the ground truth (Y) |
|Predictive Parity |Statistical Sufficiency |Y ⊥ A\|R |sensitive attributes (A) are statistically independent of the ground truth (Y) given the prediction (R) |


<br><br>
## Fairness Measures Defined

|Name | Definition | Description | About |  Aliases | Criteria|
|-|-|-|-|-|-|
|**Demographic Parity**| P( ŷ=1\|G=u)=P(ŷ=1\|G=p ) | Predictions must be statistically independent from the sensitive attributes. Subjects in all groups should have equal probability of being assigned to the positive class. | P(y=1) may differ among groups | Statistical Parity, Equal Acceptance Rate, Benchmarking | Statistical Independence |
|**Conditional Statistical Parity**| P(ŷ=1\|L=l,G=u) =P(ŷ=1\|L=l,G=p ) | Subjects in all groups should have equal probability of being assigned to the positive class conditional upon legitimate factors (L) | &nbsp; | &nbsp; | Statistical Separation|
|**False positive error rate (FPR) balance**| P(ŷ=1\|Y=0,G=u)=P(ŷ=1\|Y=0,G=p) | Equal probabilities for subjects in the negative class to have positive predictions. <br> Mathematically equivalent to equal TNR: P(d=0\|Y=0,G=m)=P(d=0\|Y =0,G=f) | &nbsp; | Predictive Equality | &Statistical Separation|
|**False negative error rate (FNR) balance**| P(ŷ=0\|Y=1,G=u)=P(ŷ=0\|Y=1,G=p ) | Equal probabilities for subjects in the positive class to have negative predictions. <br> Mathematically equivalent to equal TPR: P(d=1\|Y=1,G=m)=P(d=1\|Y=1,G=f). |     |  Equal Opportunity | Statistical Separation|
|**Equalized Odds**| P(ŷ=1\|Y=c, G=u)=P(ŷ=1\|Y=c, G=p ), c ∈ 0, 1. | Equal TPR and equal FPR |Mathematically equivalent to the conjunction of FPR balance and FNR balance|  Disparate mistreatment, Conditional procedure accuracy equality | Statistical Separation|
|**Predictive Parity**| P(Y=1\|ŷ=1,G=u)=P(Y=1\|ŷ=1,G=p) | All groups have equal PPV (probability that a subject with a positive prediction actually belongs to the positive class. <br> Mathematically equivalent to equal False Discovery Rate (FDR): P(Y=0\|d=1,G=m)=P(Y=0\|d=1,G=f ) | &nbsp; |  Outcome test | Statistical Sufficiency|
|**Conditional use accuracy equality**| (P(Y=1\|ŷ=1,G=u)=P(Y=1\|ŷ=1,G=p)) ∧ (P(Y=0\|ŷ=0,G=u)=P(Y=0\|ŷ=0,G=p)) | &nbsp; | &nbsp; |  &nbsp; | &nbsp; | Statistical Sufficiency|
|**Overall Accuracy Equity**| P(ŷ=Y,G=m)=P(ŷ=Y,G=p) | &nbsp; | Use when True Negatives are as desirable as True Positives| &nbsp; | &nbsp;|
|**Treatment Equality**| FNu/FPu=FNp/FPp | Groups have equal ratios of False Negative Rates to False Positive Rates | &nbsp; | &nbsp;| &nbsp; |
|**Calibration**| P(Y=1\|S=s,G=u)=P(Y=1\|S=s,G=p ) | For a predicted probability score S, both groups should have equal probability of belonging to the positive class | &nbsp; | &nbsp;| Test-fairness, matching conditional frequencies |  Statistical Sufficiency
|**Well-calibration**| P(Y=1\|S=s,G=u)=P(Y=1\|S=s,G=p )=s | For a predicted probability score S, both groups should have equal probability of belonging to the positive class, and this probability is equal to S | &nbsp; | &nbsp;| Statistical Sufficiency|
|**Balance for positive class**| E(S\|Y=1,G=u)=E(S\|Y=1,G=p ) | Subjects in the positive class for all groups have equal average predicted probability score S | &nbsp;| &nbsp;| Statistical Separation|
|**Balance for negative class**| E(S\|Y=0,G=u)=E(S\|Y=0,G=p) | Subjects in the negative class for all groups have equal average predicted probability score S |     | &nbsp; | &nbsp; | Statistical Separation|
||||||||
|**Causal discrimination**| (Xp=Xu ∧ Gp != Gu) → ŷu=ŷp | Same classification produced for any two subjects with the exact same attributes| &nbsp; | &nbsp; |&nbsp; |
|**Fairness through unawareness**| Xi=Xj → ŷi=ŷj | No sensitive attributes are explicitly used in the decision-making process| &nbsp; |Unawareness| &nbsp; |
|**Fairness through awareness (Individual Fairness)**| for a set of applicants V , a distance metric between applicants k : V Å~V → R, a mapping from a set of applicants to probability distributions over outcomes M : V → δA, and a distance D metric between distribution of outputs, fairness is achieved iff D(M(x),M(y)) ≤ k(x,y). | Similar individuals (as defined by some distance metric) should have similar classification | &nbsp; | Individual Fairness | &nbsp; |
|**Counterfactual fairness**| A causal graph is counterfactually fair if the predicted outcome d in the graph does not depend on a descendant of the protected attribute G. | &nbsp; |  &nbsp; |  &nbsp; |  &nbsp; |
||||||||


<br><br>
----
## References
<a id = "agniel2018biases"></a>
Agniel D, Kohane IS, & Weber GM (2018). Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. Bmj, 361. Retrieved from [https://www.bmj.com/content/361/bmj.k1479](https://www.bmj.com/content/361/bmj.k1479)

<a id = "bantilan2018_ref"></a>
Bantilan N (2018). Themis-ml: A fairness-aware machine learning interface for end-to-end discrimination discovery and mitigation. Journal of Technology in Human Services, 36(1), 15-30. Retrieved from [https://www.tandfonline.com/doi/abs/10.1080/15228835.2017.1416512](https://www.tandfonline.com/doi/abs/10.1080/15228835.2017.1416512)

<a id = "barocas2016_ref"></a>
Barocas S, & Selbst AD (2016). Big data's disparate impact. California Law Review, 104, 671. Retrieved from [http://www.californialawreview.org/wp-content/uploads/2016/06/2Barocas-Selbst.pdf](http://www.californialawreview.org/wp-content/uploads/2016/06/2Barocas-Selbst.pdf)

Bellamy RK, Dey K, Hind M, Hoffman SC, Houde S, Kannan K, ... & Nagar S (2018). AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. arXiv Preprint. [arXiv:1810.01943.](https://arxiv.org/abs/1810.01943). See Also [AIF360 Documentation](http://aif360.mybluemix.net/)

Bird S, Dudík M,  Wallach H,  & Walker K (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Research. Retrieved from [https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf). See Also [FairLearn Reference](https://fairlearn.github.io/).

<a id = "dwork2012_ref"></a>
Dwork C, Hardt M, Pitassi T, Reingold O, & Zemel R (2012, January). Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference (pp. 214-226). Retrieved from [https://arxiv.org/pdf/1104.3913.pdf](https://arxiv.org/pdf/1104.3913.pdf)

<a id = "fourfifths_ref"></a>
Equal Employment Opportunity Commission, & Civil Service Commission, Department of Labor & Department of Justice (1978). Uniform guidelines on employee selection procedures. Federal Register, 43(166), 38290-38315. Retrieved from [http://uniformguidelines.com/uniformguidelines.html#18](http://uniformguidelines.com/uniformguidelines.html#18)

<a id = "hardt2016_ref"></a>
Hardt M, Price E, & Srebro N (2016). Equality of opportunity in supervised learning. In Advances in neural information processing systems (pp. 3315-3323). Retrieved from [http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf](http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)

<a id = "hcup_ref"></a>
Healthcare Cost and Utilization Project (HCUP) (2017, March). HCUP CCS. Agency for Healthcare Research and Quality, Rockville, MD. Retrieved from [www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)

Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, & Mark RG (2016). Scientific Data. MIMIC-III, a freely accessible critical care database. DOI: 10.1038/sdata.2016.35. Retrieved from [http://www.nature.com/articles/sdata201635](http://www.nature.com/articles/sdata201635)

<a id = "kim2018_ref"></a>
Kim M, Reingol O, & Rothblum G (2018). Fairness through computationally-bounded awareness. In Advances in Neural Information Processing Systems (pp. 4842-4852). Retrieved from [https://arxiv.org/pdf/1803.03239.pdf](https://arxiv.org/pdf/1803.03239.pdf)

<a id = "naacp"></a>
National Association for the Advancement of Colored People (NAACP) (2012). Criminal Justice Fact Sheet. NAACP. Retrieved from http://www.naacp.org/pages/criminal-justice-fact-sheet.

<a id = "romei2014_ref"></a>
Romei, A., & Ruggieri, S. (2014). A multidisciplinary survey on discrimination analysis. The Knowledge Engineering Review, 29(5), 582-638. Retrieved from [https://www.cambridge.org/core/journals/knowledge-engineering-review/article/multidisciplinary-survey-on-discrimination-analysis/D69E925AC96CDEC643C18A07F2A326D7](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/multidisciplinary-survey-on-discrimination-analysis/D69E925AC96CDEC643C18A07F2A326D7)

<a id = "russell2017_ref"></a>
Russell C, Kusner MJ, Loftus J, & Silva R (2017). When worlds collide: integrating different counterfactual assumptions in fairness. In Advances in Neural Information Processing Systems (pp. 6414-6423). Retrieved from [https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf](https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf)

<a id = "shorrocks_ref"></a>
Shorrocks AF (1980). The class of additively decomposable inequality measures. Econometrica: Journal of the Econometric Society, 613-625. Retrieved from [http://www.vcharite.univ-mrs.fr/PP/lubrano/atelier/shorrocks1980.pdf](http://www.vcharite.univ-mrs.fr/PP/lubrano/atelier/shorrocks1980.pdf)

<a id = "speicher2018_ref"></a>
Speicher T, Heidari H, Grgic-Hlaca N, Gummadi KP, Singla A, Weller A, & Zafar M B (2018, July). A unified approach to quantifying algorithmic unfairness: Measuring individual &group unfairness via inequality indices. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2239-2248). Retrieved from [https://arxiv.org/pdf/1807.00787.pdf](https://arxiv.org/pdf/1807.00787.pdf)

<a id = "zemel2013_ref"></a>
Zemel R, Wu Y, Swersky K, Pitassi T, & Dwork C (2013, February). Learning fair representations. International Conference on Machine Learning (pp. 325-333). Retrieved from [http://proceedings.mlr.press/v28/zemel13.pdf](http://proceedings.mlr.press/v28/zemel13.pdf)

<a id = "zafar2017_ref"></a>
Zafar MB, Valera I, Gomez Rodriguez, M, & Gummadi KP (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).  https://arxiv.org/pdf/1610.08452.pdf


