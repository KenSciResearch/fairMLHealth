# Quick Reference for Fairness Measures
----
##  Fairness Metrics <a id="metric_quickref"></a>

### Metrics by Category

There are three common categories of metrics for determining whether a model is considered "fair": Group Fairness, which compares the statistical similarities of predictions relative to known and discrete protected groupings; Similarity-Based Measures, which evaluate predictions without those discrete protected groups; and Causal Reasoning measures, which evaluate fairness through the use of causal models.


|**Category** |**Metric** |**Definition** |**Weakness** |**References** |
|------|------|------|------|------|
|Group Fairness |**Demographic Parity**| A model has **Demographic Parity** if the predicted positive rates (selection rates) are approximately the same for all protected attribute groups: <br> <img src="https://render.githubusercontent.com/render/math?math=\dfrac{P(\hat{y}=1\lvert%20unprivileged)}{P(\hat{y}=1\rvert%20privileged)}"> <br> <br> Harms Addressed: Allocative| Historical biases present in the data are not addressed and may still bias the model. | [Zafar *et al* (2017)](#zafar2017_ref) |
||**Equalized Odds**| Odds are equalized if $P(+)$ is approximately the same for all protected attribute groups.<br>  **Equal Opportunity** is a special case of equalized odds specifying that $P(+ \rvert y = 1)$ is approximately the same across groups. <br> <br> Harms Addressed: Allocative, Representational | Historical biases present in the data  are not addressed and may still bias the model. | [Hardt *et al* (2016)](#hardt2016_ref) |
||**Predictive Parity**| This parity exists where the Positive Predictive Value is approximately the same for all protected attribute groups. <br> <br> Harms Addressed: Allocative, Representational | Historical biases present in the data are not addressed and may still bias the model.  | [Zafar *et al* (2017)](#zafar2017_ref) |
|------|------|------|------|------|
| Similarity-Based Measures |**Individual Fairness**| Individual fairness exists if "similar" individuals (ignoring the protected attribute) are likely to have similar predictions. <br> <br> Harms Addressed: Representational | The appropriate metric for similarity may be ambiguous. |[Dwork (2012)](#dwork2012_ref), [Zemel (2013)](#zemel2013_ref), [Kim *et al* (2018)](#kim2018_ref) |
| &nbsp; |**Entropy-Based Indices**| Measures of entropy, particularly existing inequality indices from the field of economics, are applied to evaluate either individuals or groups <br> <br> Harms Addressed: Representational |  |[Speicher (2018)](#speicher2018_ref) |
| &nbsp; |**Unawareness** | A model is unaware if the protected attribute is not used. <br> <br> Harms Addressed: Allocative, Representational | Removal of a protected attribute may be ineffectual due to the presence of proxy features highly correlated with the protected attribute.| [Zemel *et al* (2013)](#zemel2013_ref), [Barocas and Selbst (2016)](#barocas2016_ref) |
|------|------|------|------|------|
| Causal Reasoning |**Counterfactual Fairness** \*| Counterfactual fairness exists where counterfactual replacement of the protected attribute does not significantly alter predictive performance. This counterfactual change must be propogated to correlated variables. <br><br>Harms Addressed: Allocative, Representational | It may be intractable to develop a counterfactual model.  | [Russell *et al* (2017)](#russell2017_ref) |
|------|------|------|------|------|
### Statistical Definitions of Group Fairness
|Metric |Statistical Criteria |Definition |Description |
|------|------|------|------|------|
|Demographic Parity|Statistical Independence |$R{\perp}G$ |sensitive attributes (A) are statistically independent of the prediction result (R) |
|Equalized Odds| Statistical Separation |$R{\perp}A\rvert{Y}$ |sensitive attributes (A) are statistically independent of the prediction result (R) given the ground truth (Y) |
|Predictive Parity |Statistical Sufficiency |$Y{\perp}A\rvert{R}$ |sensitive attributes (A) are statistically independent of the ground truth (Y) given the prediction (R)

From: [Verma & Rubin, 2018](#vermarubin)

## Fairness Measures

|Name |Definition |About |Aliases |
|------|------|------|------|
|**Demographic Parity** | <img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}\lvert{G=u})=P(\hat{y}\lvert{G=p})"> |Predictions must be statistically independent from the sensitive attributes. Subjects in all groups should have equal probability of being assigned to the positive class. Note: may fail if the distribution of the ground truth justifiably differs among groups <br>Criteria: Statistical Independence |Statistical Parity, Equal Acceptance Rate, Benchmarking |
|**Conditional Statistical Parity** |<img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}=1\lvert{L=l,G=u})=P(\hat{y}=1\lvert{L=l,G=p})"> | Subjects in all groups should have equal probability of being assigned to the positive class conditional upon legitimate factors (L). <br>Criteria: Statistical Separation |&nbsp; |
|**False positive error rate (FPR) balance** |<img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}=1\lvert{Y=0,G=u})=P(\hat{y}=1\lvert{Y=0,G=p})"> |Equal probabilities for subjects in the negative class to have positive predictions. <br> Mathematically equivalent to equal TNR: P(d=0\lvert{Y=0,G=m})=P(d=0\lvert{Y =0,G=f}) <br>Criteria: Statistical Separation | Predictive Equality |
|**False negative error rate (FNR) balance**| <img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}=0\lvert{Y=1,G=u})=P(\hat{y}=0\lvert{Y=1,G=p})"> | Equal probabilities for subjects in the positive class to have negative predictions. <br> Mathematically equivalent to equal TPR: $P(d=1\lvert{Y=1,G=m})=P(d=1\lvert{Y=1,G=f})$. <br>Criteria: Statistical Separation | Equal Opportunity |
|**Equalized Odds**| <img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}=1\lvert{Y=c,G=u})=P(\hat{y}=1\lvert{Y=c,G=p}),{c}\in{0,1}"> | Equal TPR and equal FPR. Mathematically equivalent to the conjunction of FPR balance and FNR balance <br>Criteria: Statistical Separation|  Disparate mistreatment, Conditional procedure accuracy equality |
|**Predictive Parity**| <img src="https://render.githubusercontent.com/render/math?math=P(Y=1\lvert{\hat{y}=1,G=u})=P(Y=1\lvert{\hat{y}=1,G=p})"> | All groups have equal PPV (probability that a subject with a positive prediction actually belongs to the positive class. <br> Mathematically equivalent to equal False Discovery Rate (FDR): $P(Y=0\lvert{d=1,G=m})=P(Y=0\lvert{d=1,G=f})$ <br>Criteria: Statistical Sufficiency |Outcome Test |
|**Conditional use accuracy equality**| <img src="https://render.githubusercontent.com/render/math?math=(P(Y=1\lvert{\hat{y}=1,G=u})=P(Y=1\lvert{\hat{y}=1,G=p}))"> <img src="https://render.githubusercontent.com/render/math?math=\wedge (P(Y=0\lvert{\hat{y}=0,G=u})=P(Y=0\lvert{\hat{y}=0,G=p}))"> | Criteria: Statistical Sufficiency | &nbsp; |
|**Overall Accuracy Equity**| <img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}=Y,G=m)=P(\hat{y}=Y,G=p)"> |Use when True Negatives are as desirable as True Positives |&nbsp; |
|**Treatment Equality**| <img src="https://render.githubusercontent.com/render/math?math=FNu/FPu=FNp/FPp"> | Groups have equal ratios of False Negative Rates to False Positive Rates |&nbsp; |
|**Calibration**| <img src="https://render.githubusercontent.com/render/math?math=P(Y=1\lvert{S=s,G=u})=P(Y=1\lvert{S=s,G=p})"> | For a predicted probability score S, both groups should have equal probability of belonging to the positive class <br>Criteria: Statistical Sufficiency |Test-fairness, matching conditional frequencies |
|**Well-calibration**| <img src="https://render.githubusercontent.com/render/math?math=P(Y=1\lvert{S=s,G=u})=P(Y=1\lvert{S=s,G=p})=s"> | For a predicted probability score S, both groups should have equal probability of belonging to the positive class, and this probability is equal to S <br>Criteria: Statistical Sufficiency |&nbsp; |
|**Balance for positive class**| <img src="https://render.githubusercontent.com/render/math?math=E(S\lvert{Y=1,G=u})=E(S\lvert{Y=1,G=p})"> | Subjects in the positive class for all groups have equal average predicted probability score S <br>Criteria: Statistical Separation |&nbsp; |
|**Balance for negative class**| <img src="https://render.githubusercontent.com/render/math?math=E(S\lvert{Y=0,G=u})=E(S\lvert{Y=0,G=p})"> | Subjects in the negative class for all groups have equal average predicted probability score S <br>Criteria: Statistical Separation |&nbsp; |
||||||
|**Causal discrimination**| <img src="https://render.githubusercontent.com/render/math?math=(X_p=X_u\wedge%20G_p!=G_u)\rightarrow\hat{y}_u=\hat{y}_p"> | Same classification produced for any two subjects with the exact same attributes |&nbsp; |
|**Fairness through unawareness**| <img src="https://render.githubusercontent.com/render/math?math=X_i=X_j\rightarrow\hat{y}_i=\hat{y}_j"> | No sensitive attributes are explicitly used in the decision-making process <br>Criteria: Unawareness | &nbsp; |
|**Fairness through awareness (Individual Fairness)**| for a set of applicants V , a distance metric between applicants k : V Å~V → R, a mapping from a set of applicants to probability distributions over outcomes M : V → δA, and a distance D metric between distribution of outputs, fairness is achieved iff <img src="https://render.githubusercontent.com/render/math?math=D(M(x),M(y))≤k(x,y)"> | Similar individuals (as defined by some distance metric) should have similar classification |Individual Fairness |
|**Counterfactual fairness**| A causal graph is counterfactually fair if the predicted outcome d in the graph does not depend on a descendant of the protected attribute G. |&nbsp; |&nbsp; |
||||||


<br><br>
## Interpretation of Common Measures <a id="measure_quickref"></a>

|Group Measure Type|Examples| "Fair" Range |
|----|----|----|
|Statistical Ratio|Disparate Impact Ratio, Equalized Odds Ratio| 0.8 <= "Fair" <= 1.2|
|Statistical Difference (Binary Classification) |Equalized Odds Difference, Predictive Parity Difference| -0.1 <= "Fair" <= 0.1|
|Statistical Difference (Regression) | MAE Difference, Mean Prediction Difference | Problem Specific |

| Metric | Measure | Equation | Interpretation |
|:---- |:---- |:---- |:---- |
|**Group Fairness Measures - Binary Classification**  |Selection Rate|<img src="https://render.githubusercontent.com/render/math?math=\dfrac{\sum_{i=0}^N%20\hat{y}_i}{N}"> | - |
|&nbsp;|Demographic (Statistical) Parity Difference | <img src="https://render.githubusercontent.com/render/math?math=P(\hat{y}=1\lvert%20unprivileged)-P(\hat{y}=1\rvert%20privileged)"> |(-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp; |Disparate Impact Ratio (Demographic Parity Ratio)| <img src="https://render.githubusercontent.com/render/math?math=\dfrac{P(\hat{y}=1\%20\rvert%20u)}{P(\hat{y}=1\%20\rvert%20p)}=\dfrac{selection\_rate(\hat{y}_{u})}{selection\_rate(\hat{y}_{p})}"> |< 1 favors privileged group <br>  > 1 favors unprivileged group |
|&nbsp; |Positive Rate Difference| <img src="https://render.githubusercontent.com/render/math?math=precision({\hat{y}}_{u})-precision({\hat{y}}{u})"> |(-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp; |Average Odds Difference| <img src="https://render.githubusercontent.com/render/math?math=\dfrac{(FPR_{u}-FPR_{p})+(TPR_{u}-TPR_{p})}{2}"> |(-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp; |Average Odds Error| <img src="https://render.githubusercontent.com/render/math?math=\dfrac{\left\lvert%20FPR_{u}-FPR_{p}\right\rvert+\left\lvert%20TPR_{u}-TPR_{p}\right\rvert}{2}"> |(-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp; |Equal Opportunity Difference| <img src="https://render.githubusercontent.com/render/math?math=recall({\hat{y}}_{u})-recall({\hat{y}}_{p})"> |(-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp; |Equal Odds Difference| <img src="https://render.githubusercontent.com/render/math?math=max((FPR_{u}-FPR_{p}),(TPR_{u}-TPR_{p}))"> |(-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp; |Equal Odds Ratio| <img src="https://render.githubusercontent.com/render/math?math=min(\dfrac{FPR_{u}}{FPR_{p}},\dfrac{TPR_{u}}{TPR_{p}})"> |< 1 favors privileged group <br>  > 1 favors unprivileged group |
|**Group Fairness Measures - Regression**| Mean Prediction Ratio| <img src="https://render.githubusercontent.com/render/math?math=mean\_prediction\_ratio=\dfrac{\mu(\hat{y}_{u})}{\mu(\hat{y}_{p})}"> | < 1 favors privileged group <br>  > 1 favors unprivileged group |
|&nbsp;  | Mean Prediction Difference| <img src="https://render.githubusercontent.com/render/math?math=mean\_difference=\mu(\hat{y}_{u})-\mu(\hat{y}_{p})"> | (-) favors privileged group <br> (+) favors unprivileged group |
|&nbsp;  | MAE Ratio|<img src="https://render.githubusercontent.com/render/math?math=MAE\_ratio=\dfrac{MAE_u}{MAE_p}" >| < 1 favors privileged group <br>  > 1 favors unprivileged group |
|&nbsp;  | MAE Difference| <img src="https://render.githubusercontent.com/render/math?math=MAE\_difference=MAE_u-MAE_p"> | (-) favors privileged group <br> (+) favors unprivileged group |
|**Individual Fairness Measures** |Consistency Score | <img src="https://render.githubusercontent.com/render/math?math=1-\frac{1}{n\cdot{N_{n_neighbors}}}*\sum_{i=1}^n\lvert\hat{y}_i-\sum_{j\in\mathcal{N}_{neighbors}(x_i)}\hat{y}_j\rvert"> | 1 is consistent <br> 0 is inconsistent |
|&nbsp; |Generalized Entropy Index| <img src="../img/generalized_entropy_equation.png"> | - |
|&nbsp; |Generalized Entropy Error| <img src="https://render.githubusercontent.com/render/math?math=GE(\hat{y}_i-y_i+1)"> | - |
|&nbsp; |Between-Group Generalized Entropy Error| <img src="https://render.githubusercontent.com/render/math?math=GE([N_{u}*mean(Error_{u}),N_{p}*mean(Error_{p})])"> | 0 is fair <br>(+) is unfair |

<br><br>
----
## References
<a name="Agarwal2018"></a>
Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In International Conference on Machine Learning (pp. 60-69). PMLR. Available through [arXiv preprint:1803.02453](https://arxiv.org/pdf/1803.02453.pdf).

<a id="barocas2016_ref"></a>
Barocas, S., & Selbst AD (201). Big data's disparate impact. California Law Review, 104, 671. Retrieved from [https://www.cs.yale.edu/homes/jf/BarocasDisparateImpact.pdf](https://www.cs.yale.edu/homes/jf/BarocasDisparateImpact.pdf)

<a id="dwork2012_ref"></a>
Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012, January). Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference (pp. 214-226). Retrieved from [https://arxiv.org/pdf/1104.3913.pdf](https://arxiv.org/pdf/1104.3913.pdf)

<a id="hardt2016_ref"></a>
Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In Advances in neural information processing systems (pp. 3315-3323). Retrieved from [http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf](http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)

<a id="kim2018_ref"></a>
Kim, M., Reingol, O., & Rothblum, G. (2018). Fairness through computationally-bounded awareness. In Advances in Neural Information Processing Systems pp. 4842-4852). Retrieved from [https://arxiv.org/pdf/1803.03239.pdf](https://arxiv.org/pdf/1803.03239.pdf)

<a id="russell2017_ref"></a>
Russell, C., Kusner, M.J., Loftus, J., & Silva, R. (2017). When worlds collide: integrating different counterfactual assumptions in fairness. In Advances in Neural Information Processing Systems (pp. 6414-6423). Retrieved from [https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf](https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf)

<a id="vermarubin"></a>
Verma, S., & Rubin, J. (2018, May). Fairness definitions explained. In 2018 ieee/acm international workshop on software fairness (fairware) (pp. 1-7). IEEE.

<a id="zemel2013_ref"></a>
Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013, February). Learning fair representations. International Conference on Machine Learning (pp. 325-333). Retrieved from [http://proceedings.mlr.press/v28/zemel13.pdf](http://proceedings.mlr.press/v28/zemel13.pdf)

<a id="zafar2017_ref"></a>
Zafar, M.B., Valera, I., Gomez Rodriguez, M., & Gummadi, K.P. (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).  https://arxiv.org/pdf/1610.08452.pdf
