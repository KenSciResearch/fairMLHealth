# Evaluating Fairness in Machine Learning


## About
All machine learning (ML) models can be assumed to hold biases, just as all humans hold biases. Building models that will work for all patients starts with understanding that bias through measurement and evaluation.

This reference introduces concepts, methods, and libraries for measuring fairness in ML as it relates to problems in healthcare. This is a revamped version of the tutorial presented at the [KDD 2020 Tutorial on Fairness in Machine Learning for Healthcare](publications/KDD2020-FairnessInHealthcareML-Slides.pptx), the notebook for which can be found here: [docs/publications/KDD2020-FairnessInHealthcareML-TutorialNotebook.ipynb](publications/KDD2020-FairnessInHealthcareML-TutorialNotebook.ipynb).

There are abundant other publications covering the theoretical basis for fairness metrics, and many resources both online and academic covering the details of specific fairness measures. See [References](#references) and [Additional Resources](#additional_resources) below, or [Our References and Resources Page](References_and_Resources.md) for just a few. Many of these otherwise excellent references stop short of discussing the practical and philosophical considerations raised when evaluating real models for real customers. Here we attempt to bridge that gap.


## Contents
[Part 1](#part1) - Framing the Problem
  * Background: Disparity & Harms

[Part 2](#part2) - Defining and Measuring Fairness (in Python)
  * Metrics of Fairness
  * Example Model
  * Group Fairness Measures
  * Individual Fairness Measures

[Part 3](#part3) - Comparing Models and Mitigating Disparity
  * Choosing Appropriate Measure(s)
  * Comparing Models
  * Fairness-Aware ML Algorithms

[Part 4](#part4) - Conclusion
  * How Fair is Fair Enough?


----
# Framing the Problem <a class = "anchor" id = "part1"></a>

## Context
In issues of social justice, *"discrimination"* refers to the unjustified, differential treatment of individuals based on their sociodemographic status [[Romei and Ruggieri 2014]](#romei2014_ref). The "socially salient" sociodemographic groups [[Speicher 2018]](#speicher2018_ref) about which discrimination is measured are known as ***protected attributes***, *sensitive attributes*, or *protected features*. So, a "fair" model could be considered one that does not discriminate relative to any protected attributes.

## Disparity  <a id="disparity_def"></a>

The term "discrimination" typically evokes direct or deliberate action to disadvantage one race, religion, or ethnicity. This kind of disparity is known as *disparate treatment*. However, a more insidious form of discrimination can occur when ostensibly unbiased practices result in the – perhaps unconscious – unfair treatment of a socially disadvantaged group. This is known as *disparate impact*.

Disparate impact in a machine learning model originates from bias in either the data or the algorithms. A popular example is the prejudicially biased data used for recidivism prediction. Due to disparate socioeconomic factors and systemic racism in the United States, blacks have historically been (and continue to be) incarcerated at higher rates than whites [[NAACP]](#naacp).  Not coincidentally, blacks are also exonerated due to wrongful accusation at a considerably higher rate than whites [[NAACP]](#naacp).  A recidivism model that fails to adjust for circumstances such as these will predict a higher rate of recidivism among blacks.

Machine learning models can also be a source of disparate impact in their implementation, through unconscious human biases that affect the fair interpretation or use of the model's results. This reference does not cover measurement of fairness at implementation. However, if you are interested in fair implementation, we recommend looking at Google's [Fairness Indicators](https://www.tensorflow.org/tfx/guide/fairness_indicators).

## Harms

In evaluating the potential impact of an ML model, it can be helpful to first clarify what specific harm(s) can be caused by the model's failures. In the context of fairness, machine learning "harms" are commonly observed to fall into one of two categories.

* **Allocative Harm:** functionality promoting unfair allocation of finite resources
* **Representational Harm:** functionality promoting the continued marginalization of some groups
  * Examples include:
    * Quality of Service: allocating higher insurance payouts for males than for females
    * Stereotyping: service more likely to show advertising for bail bonds to dark skinned men
    * Under-Representation: image search for "doctor" returning mostly images of white men
    * Recognition: facial recognition mistakenly and offensively labeling a person as an animal


References:
[The Trouble with Bias](https://youtu.be/fMym_BKWQzk) Kate Crawford, NIPS2017


----
# Defining and Measuring Fairness <a id="part2"></a>

The following section defines common fairness measures that are used elsewhere. Skip ahead to [Part 3](#part3) for an example of how these measures are applied.

### [Convenient Charts of Fairness Measures](Measures_QuickReference.md)

## Metrics of Fairness  <a id="metric_quickref"></a>

There are six common metrics for determining whether a model is considered "fair": Equal Treatment ("**Unawareness**"), **Demographic Parity**, **Equalized Odds**, **Predictive Parity**, **Individual Fairness**, and **Counterfactual Fairness**.


<h3 style="text-align: center"><u>  Statistical Criteria for Fairness Metrics </u> </h3>

|Metric |Statistical Criteria |Definition |Description |
|-|-|-|-|
|Demographic Parity|Statistical Independence |R ⊥ G |sensitive attributes (A) are statistically independent of the prediction result (R) |
|Equalized Odds| Statistical Separation |R ⊥ A\|Y |sensitive attributes (A) are statistically independent of the prediction result (R) given the ground truth (Y) |
|Predictive Parity |Statistical Sufficiency |Y ⊥ A\|R |sensitive attributes (A) are statistically independent of the ground truth (Y) given the prediction (R) |


<h3 style="text-align: center"><u>  Definitions of Fairness </u> </h3>

| Category | Metric | Definition | Weakness | References |
|------|------|------|------|------|
| Group Fairness |**Demographic Parity**| A model has **Demographic Parity** if the predicted positive rates (selection rates) are approximately the same for all protected attribute groups. <img src="https://render.githubusercontent.com/render/math?math=\dfrac{P(\hat{y}=1\lvert%20unprivileged)}{P(\hat{y}=1\rvert%20privileged)}"> <br> Harms Addressed: Allocative| Historical biases present in the data are not addressed and may still bias the model. | [Zafar *et al* (2017)](#zafar2017_ref) |
||**Equalized Odds**| Odds are equalized if P(+) is approximately the same for all protected attribute groups.<br>  **Equal Opportunity** is a special case of equalized odds specifying that P(+\|y=1) is approximately the same across groups. <br> Harms Addressed: Allocative, Representational | Historical biases present in the data  are not addressed and may still bias the model. | [Hardt *et al* (2016)](#hardt2016_ref) |
||**Predictive Parity**| This parity exists where the Positive Predictive Value is approximately the same for all protected attribute groups. <br> Harms Addressed: Allocative, Representational | Historical biases present in the data are not addressed and may still bias the model.  | [Zafar *et al* (2017)](#zafar2017_ref) |
||||||
| Similarity-Based Measures |**Individual Fairness**| Individual fairness exists if "similar" individuals (ignoring the protected attribute) are likely to have similar predictions. <br>Harms Addressed: Representational | The appropriate metric for similarity may be ambiguous. |[Dwork (2012)](#dwork2012_ref), [Zemel (2013)](#zemel2013_ref), [Kim *et al* (2018)](#kim2018_ref) |
| &nbsp; |**Unawareness** | A model is unaware if the protected attribute is not used.  | Removal of a protected attribute may be ineffectual due to the presence of proxy features highly correlated with the protected attribute.| [Zemel *et al* (2013)](#zemel2013_ref), [Barocas and Selbst (2016)](#barocas2016_ref) |
||||||
| Causal Reasoning |**Counterfactual Fairness** \*| Counterfactual fairness exists where counterfactual replacement of the protected attribute does not significantly alter predictive performance. This counterfactual change must be propagated to correlated variables. <br>Harms Addressed: Allocative, Representational | It may be intractable to develop a counterfactual model.  | [Russell *et al* (2017)](#russell2017_ref) |
||||||

\* *Note that this reference will not elaborate the details of Counterfactual Fairness since the libraries used do not have built-in functionality for it. For an example of Counterfactual Fairness, see "ThemisML" by [Bantilan (2018)](#bantilan2018_ref).*

## Example Model


For the following explanations of specific measures, we'll use the results of a hypothetical predictive model shown in the table below. Our example shows performance and fairness measures relative to language as a protected attribute (i.e., whether or not a given patient is an English speaker). Also shown are stratified tables that will aid us in better understanding the source of any apparent disparity. Curious readers can walk through the generation of the model and this table of fairness measures in the [Evaluating Fairness in Binary Classification Tutorial Notebook](../examples_and_tutorials/Tutorial-EvaluatingFairnessInBinaryClassification.ipynb).


<a id="fairness_report"></a>
<h3 style="text-align: center"><u> Example FairMLHealth Fairness Report </u> </h3>
<p style="text-align: center"><img src="./img/evaluation_example_report.png"
     alt="Example FairMLHealth Fairness Report"
     /></p>

<h3 style="text-align: center"><u> Example FairMLHealth Stratified Data Table </u> </h3>
<p style="text-align: center"><img src="./img/evaluation_stratified_data.png"
     alt="Example FairMLHealth Stratified Data Table"
     /></p>

<h3 style="text-align: center"><u> Example FairMLHealth Stratified Performance Table </u> </h3>
<p style="text-align: center"><img src="./img/evaluation_example_stratified_performance.png"
     alt="Example FairMLHealth Stratified Performance Table"
     /></p>

<h3 style="text-align: center"><u> Example FairMLHealth Stratified Fairness Table </u> </h3>
<p style="text-align: center"><img src="./img/evaluation_example_stratified_fairness.png"
     alt="Example FairMLHealth Stratified Fairness Table"
     /></p>


----
## Group Fairness Measures

### Demographic Parity  <a id="dem_parity"></a>
A model has **Demographic Parity** if the predicted positive rates (selection rates) are approximately the same for all groups of the protected attribute. Two common measures are the Statistical Parity Difference and the Disparate Impact Ratio.

The *Statistical Parity Difference* is the difference in the probability of prediction between the two groups. A difference of 0 indicates that the model is perfectly fair relative to the protected attribute (it favors neither the privileged nor the unprivileged group). Values between -0.1 and 0.1 are considered reasonably fair.
> <img src="https://render.githubusercontent.com/render/math?math=statistical\_parity\_difference = P(\hat{y} = 1\ |\ unprivileged) - P(\hat{y} = 1\ |\ privileged)">

The *Disparate Impact Ratio* is the ratio between the probability of positive prediction for the unprivileged group and the probability of positive prediction for the privileged group. A ratio of 1 indicates that the model is fair relative to the protected attribute (it favors neither the privileged nor the unprivileged group).  Values between 0.8 and 1.2 are considered reasonably fair.
> <img src="https://render.githubusercontent.com/render/math?math=disparate\_impact\_ratio = \dfrac{P(\hat{y} = 1\ |\ unprivileged)}{P(\hat{y} = 1\ |\ privileged)} = \dfrac{selection\_rate(\hat{y}_{unprivileged})}{selection\_rate(\hat{y}_{privileged})}">

### Equal Odds
Odds are equalized if P(+) is approximately the same for all groups of the protected attribute.

The *Equalized Odds Difference* is the greater between the difference in TPR and the difference in FPR. This provides a comparable measure to the Average Odds Difference found in [AIF360](https://github.com/IBM/AIF360). A value of 0 indicates that all groups have the same TPR, FPR, TNR, and FNR, and that the model is "fair" relative to the protected attribute.
> <img src="https://render.githubusercontent.com/render/math?math=equalized\_odds\_difference = max( (FPR_{unprivileged} - FPR_{privileged}), (TPR_{unprivileged} - TPR_{privileged}) )">

The *Equalized Odds Ratio* is the smaller between the TPR Ratio and FPR Ratio, where the ratios are defined as the ratio of the smaller of the between-group rates vs the larger of the between-group rates. A value of 1 means that all groups have the same TPR, FPR, TNR, and FNR. This measure is comparable to the Equal Opportunity Difference (found in [AIF360](https://github.com/IBM/AIF360)).
> <img src="https://render.githubusercontent.com/render/math?math=equalized\_odds\_ratio = min( \dfrac{FPR_{smaller}}{FPR_{larger}}, \dfrac{TPR_{smaller}}{TPR_{larger}} )">

*Equal Opportunity Difference (or Ratio)* compares the recall scores (TPR) between the unprivileged and privileged groups.
> <img src="https://render.githubusercontent.com/render/math?math=equal\_opportunity\_difference = recall(\hat{y}_{unprivileged}) - recall(\hat{y}_{privileged})">


### Measures of Disparate Performance
These measures evaluate whether model performance is similar for all groups of the protected attribute.

The *Positive Predictive Parity Difference (or Ratio)* compares the Positive Predictive Value (PPV, aka. precision), between groups.
> <img src="https://render.githubusercontent.com/render/math?math=positive\_predictive\_parity\_difference = precision(\hat{y}_{unprivileged}) - precision(\hat{y}_{privileged})">

The *Balanced Accuracy Difference (or Ratio)* compares the Balanced Accuracy between groups, where balanced accuracy is the mean of the sensitivity and specificity. **Since many models are biased due to data imbalance, this can be an important measure.**
> <img src="https://render.githubusercontent.com/render/math?math=balanced\_accuacy\_difference = (Sensitivity_{unprivileged} + Specificity_{unprivileged})/2 - (Sensitivity_{privileged} + Specificity_{privileged})/2">

## Comparing Group Fairness (Statistical) Measures <a id="comparing_group_measures"></a>
The highlighted rows in our example FairMLHealth Fairness Report [above](#fairness_report) indicates that the Disparate Impact ratio is out of range; but what is that range and how is it determined? In 1978, the United States Equal Employment Opportunity Commission adopted the "Four-Fifths Rule", a guideline stating that, "A selection rate for any race, sex, or ethnic group which is less than four-fifths (4/5) (or eighty percent) of the rate for the group with the highest rate will generally be regarded... as evidence of adverse impact."[EOC (1978)](#fourfifths_ref) This rubric has since been adopted for measures of fairness in ML. This translates to a "fair" range of selection rate ratios that are between 0.8 and 1.2.

The four-fifths rule works well when comparing prediction performance metrics whose values are above 0.5. However, the rule fails when comparing small values, as is the case in this example and which is as shown in the example stratified table. The ratios between two such small values can easily be well above 1.2, even though the true difference is only a few percentage points. For this reason it's useful to compare both the ratios and the differences when evaluating group measures.

Returning to the example: the Disparate Impact Ratio and Statistical Parity Difference are two related measures that compare the selection rates between the protected and unprotected groups. Although the Disparate Impact Ratio in our example is outside of the "fair" range for ratios (it's above 1.2), the Statistical Parity Difference is well within range for differences. We can see why more clearly by examining the Stratified Performance Table (also above). Here we see that the selection rates (shown as: "POSITIVE PREDICTION RATES") are actually quite close. The same is true for the Equalized Odds Ratio, which also appears outside of the "fair" range. The Equalized Odds Difference is actually quite small, which we can understand more clearly by looking at the True Positive Rates and False Positive Rates (shown as TPR and FPR) in the Stratified Table.

|Group Measure Type |Examples |"Fair" Range |Favored Group |
|- |- |- |- |
|Statistical Ratio |Disparate Impact Ratio, Equalized Odds Ratio | 0.8 <= "Fair" <= 1.2 | < 1 favors privileged group, > 1 favors unprivileged |
|Statistical Difference |Equalized Odds Difference, Predictive Parity Difference | -0.1 <= "Fair" <= 0.1 | < 0 favors privileged group, > 0 favors unprivileged |

### Problems with Group Fairness Measures
Although these statistically-based measures make intuitive sense, they are not applicable in every situation. For example, Demographic Parity is inapplicable where the base rates significantly differ between groups. Also, by evaluating protected attributes in pre-defined groups, these measures may miss certain nuance. For example, a model may perform unfairly for certain sub-groups of the unprivileged class (e.g., black females), but not for the unprivileged group as a whole.


### The Impossibility Theorem of Fairness <a id="impossibility"></a>
Another drawback of these statistically-based measures is that they are mathematically incompatible. No machine learning model can be perfectly fair according to all three metrics at once. People + AI Research (PAIR) posted an [excellent visual explanation of the Impossibility Theorem](https://pair.withgoogle.com/explorables/measuring-fairness/).


----
## Similarity-Based Measures and Individual Fairness
Measures of individual fairness determine if "similar" individuals are likely to have similar predictions. This is independent of any specific protected attribute unless otherwise stated.

### Consistency Scores <a id="consistency_score"></a>
Consistency scores measure the similarity between specific predictions and the predictions of like individuals. They are not specific to a particular attribute, but rather they evaluate the generally equal treatment of equal individuals. In [AIF360](https://github.com/IBM/AIF360), the consistency score is calculated as the compliment of the mean distance to the score of the mean nearest neighbor, using Scikit-Learn's Nearest Neighbors algorithm (default: 5 neighbors determined by the Ball Tree algorithm). For this measure, values closer to 1 indicate greater consistency, and those closer to zero indicate less consistency. More information about consistency scores is available in [[Zemel (2013)]](#zemel2013_ref).
> <img src="https://render.githubusercontent.com/render/math?math=consistency\_score%20=%201%20-%20\frac{1}{n\cdot\text{n_neighbors}}\sum_{i%20=%201}^n%20|\hat{y}_i%20-%20\sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)}%20\hat{y}_j|">


### The Generalized Entropy Index and Related Measures
The *Generalized Entropy (GE) Index* was proposed as a metric for income inequality [[Shorrocks (1980)]](#shorrocks_ref)), although it originated as a measure of redundancy in information theory. In 2018, [Speicher *et al.*](#speicher2018_ref) proposed its use for ML models. These measures are dimensionless, and therefore are most useful in comparison relative to each other. Values closer to zero indicate greater fairness, and increasing values indicating decreased fairness.
> <img src="./img/generalized_entropy_equation.png">

#### Special Cases
The *Theil Index* occurs where the <img src="https://render.githubusercontent.com/render/math?math=GE"> alpha is equal to one. Although it is dimensionless like other indices of generalized entropy, it can be transformed into an Atkinson index, which has a range between 0 and 1.
> <img src="https://render.githubusercontent.com/render/math?math=Theil%20Index%20=%20GE(\alpha%20=%201)">

The *Coefficient  of  Variation* is two times the square root of the <img src="https://render.githubusercontent.com/render/math?math=GE"> where alpha is equal to 2.
> <img src="https://render.githubusercontent.com/render/math?math=Coefficient%20of%20Variation%20=%202*\sqrt{GE(\alpha%20=%202)}">

#### Generalized Entropy of Error
*Generalized Entropy Error* is the Generalized Entropy Index of the prediction error. Like the Consistency Score above, this measure is dimensionless; however, it does not provide specific information to allow discernment between groups.
> <img src="https://render.githubusercontent.com/render/math?math=GE(Error,%20\alpha%20=%202)%20=%20GE(\hat{y}_i%20-%20y_i%20+%201)">

*Between Group Generalized Entropy Error* is the Generalized Entropy Index for the weighted means of group-specific errors. More information is available in [Speicher (2013)](#speicher2018_ref).
> <img src="https://render.githubusercontent.com/render/math?math=GE(Error_{group},%20\alpha%20=%202)%20=%20GE(%20[N_{unprivileged}*mean(Error_{unprivileged}),%20N_{privileged}*mean(Error_{privileged})]%20)">


## Comparing Similarity-Based Measures
Some measures of Individual Fairness are dimensionless, and for that reason they are most useful when comparing multiple models as we will see [below](#part3). However, some measures such as the Consistency Score and Between-Group Generalized Entropy Error exist on scales from 0 to 1. The directions of these scales can differ between measures (i.e., perfect fairness may lie at either 0 or at 1 depending upon the measure), so you will want to make a note of which applies. For example, for the Consistency Score shown above, a score of 1 is considered perfectly "fair". Adapting the four-fifths rule, we can say that a model should be consistent for at least 80% of predictions. By this measure, our example model above is out of range.

### Problems with Similarity-Based Fairness Measures
Similarity-based measures are not without their own drawbacks. The Consistency Score, for example, uses Scikit-Learn's standard K-Nearest Neighbors (KNN) algorithm to define similarity, which may need additional (separate) parameter tuning, can be sensitive to irrelevant features, and may not be appropriate in cases of high dimensionality, sparse data or missingness. This then begs the question: *is the Consistency Score out of range because our prediction model is unfair, or because we haven't properly tuned the KNN algorithm?* Without significant additional work we cannot rule out the latter. Even supposing that a properly fit KNN model is possible, the results still may not be the most appropriate measure of similarity. For example, although diseases and procedures may be predictive, can it be said that all cardiac arrest survivors who received an Echocardiogram should be predicted to spend the same amount of time in the ICU?


## See Also

### [Summary Tables:  Convenient Charts of Fairness Measures](Measures_QuickReference.md)


# Comparing Models and Mitigating Disparity <a class = "anchor" id = "part3"></a>


##  Choosing Appropriate Measure(s)
Our choice of measure is informed both by the use cases for each particular measure, and also by the problem context and by the preferences of the community(ies) affected by the model. Unfortunately this means that Unfortunately no one "correct" way to measure fairness. This also means that there is no one "correct" way to demonstrate that fairness. The burden is on the Data Scientist to transparently document their process and prove that they've taken reasonable steps to develop and to measure a model that is as fair as reasonably possible.

Although no model can be perfectly fair according to all metrics per the [Impossibility Theorem (above)](#impossibility), ideally a model will be at least within the range of fairness across the measures. From there, it's a matter of optimization for the specific measure(s) that is most applicable to the problem at hand. Thus the process begins with a clear understanding of the stakeholders and how they will view the potential outcomes. For healthcare models, the stakeholders are typically the patients, care providers, and the community(ies) being served, although it is likely that the care providers will represent the interests of the other two. It can also be helpful to create a table of outcomes, similar to the one below, to clearly document the harms, benefits, and preferences involved.

See Also: [Value Sensitive Design](https://en.wikipedia.org/wiki/Value_sensitive_design)


<h3 style="text-align: center"><u> Example Table of Outcomes </u> </h3>

|Prediction |Outcomes | Preference |
|-|-|-|
|**TP** |Benefit: Deserving patient receives help |high importance |
|**TN** |Benefit: Community resources saved |less important |
|**FP** |Harm: community resources wasted on an individual without need |less important (to avoid) |
|**FN** |Harm: reduced likelihood of recovery |high importance (to avoid) |

<p style="text-align: center"> <b>P</b> = "long length of stay expected (refer to counseling)"  </p>
<br>

### Useful Questions to Ask when Choosing the Appropriate Measure(s)
**1)** What ethical frameworks are held by the stakeholders? How do they weigh the costs and benefits of different outcomes?

**2)** Which among all available measures are out of range?
    **2b)** Why are they out of range? Is is it due to the data, the model, the measure, or some combination?

**3)** Can the sources of unfairness be sufficiently addressed through changes to either the data or the model?
    **3b)** If the model remains unfair, is it still more fair than the current decision-making process?


## Comparing Models
In this section we will compare the results of multiple models using FairMLHealth's **compare** tool. For this purpose we trained three new models: an "unaware" version of our baseline model (one that excludes the protected attribute LANGUAGE_ENGL), a fairness-aware Grid Search model constrained by demographic parity (available through [Fairlearn](https://github.com/fairlearn/fairlearn), and a basic Random Forest model using our baseline data. We compare the fairness measures of all four prediction sets to see how the model bias is affected across the spectrum of measures. Again, those who are inclined can walk through the generation of these models and this model comparison table in the [Evaluating Fairness in Binary Classification Tutorial Notebook](../examples_and_tutorials/Tutorial-EvaluatingFairnessInBinaryClassification.ipynb).

<h3 style="text-align: center"><u> Example FairMLHealth Model Comparison Report </u> </h3>
<p style="text-align: center"><img src="./img/evaluation_model_comparison.png"
     alt="Example FairMLHealth Model Comparison Report"
     /></p>

The table shows that relative fairness or unfairness of our predictions differs across the available measures, while model performance remains similar. Looking at the second row for example, we can see that the three altered model predictions have reduced Disparate Impact Ratio relative to our baseline. However, surprisingly, the Fair GridSearch result has so dramatically shifted the results that the model is now *less* fair by that measure. While the baseline model has a 20% bias in favor of the *unprivileged* group, the fairness-aware model has nearly a 40% bias favoring the *privileged* group. Yet, using the Balanced Accuracy Difference as a second example, we can see that both the unaware model and the GridSearch model are less fair by this measure.

These specific results represent one set of models on a toy example, so the results may differ for other problems. For example, the GridSearch algorithm available in Fairlearn has been shown to be effective on some datasets ([Agarwal *et al.* (2019)](#Agarwal2019)). And unawareness does not always reduce Disparate Impact; in fact it can increase it, as we showed in the [KDD 2020 Tutorial](publications/KDD2020-FairnessInHealthcareML-TutorialNotebook.ipynb) which compared fairness relative to gender using this same general setup. This goes to say that the field has not yet found a panacea which can correct all fairness issues for every model, so it's important to test different approaches. Also remember to consider the effects of biased data collection processes or biased application of model results before deciding whether to make changes to the model or the input data.


## Fairness-Aware ML Algorithms <a id="mitigation"></a>

More than a dozen fairness-aware machine learning algorithms have been developed, although as shown above they may not be necessary to improve your model. However, if necessary the following fairness-aware machine learning algorithms that are readily available through the [AIF360](https://github.com/IBM/AIF360) and [Fairlearn](https://github.com/fairlearn/fairlearn) libraries, which also have helpful documentation.

<h3 style="text-align: center"><u> Fairness-Aware Algorithms </u> </h3>

|Algorithm| AIF360 | Fairlearn| Reference|
|:----|:----|:----|:----|
|Optimized Preprocessing | Y | - | Calmon et al. (2017) |
|Disparate Impact Remover | Y | - | Feldman et al. (2015) |
|Equalized Odds Postprocessing (Threshold Optimizer) | Y | Y | Hardt et al. (2016) |
|Reweighing | Y | - | Kamiran and Calders (2012) |
|Reject Option Classification | Y | - | Kamiran et al. (2012) |
|Prejudice Remover Regularizer | Y | - | Kamishima et al. (2012) |
|Calibrated Equalized Odds Postprocessing | Y | - | Pleiss et al. (2017) |
|Learning Fair Representations | Y | - | [Zemel (2013)](#zemel2013_ref) |
|Adversarial Debiasing | Y | - | Zhang et al. (2018) |
|Meta-Algorithm for Fair Classification | Y | - | Celis et al. (2018) |
|Rich Subgroup Fairness | Y | - | [Kearns, Neel, Roth, & Wu (2018)](#kearns) |
|Exponentiated Gradient | - | Y | [Agarwal, Beygelzimer, Dudik, Langford, & Wallach (2018)](#Agarwal2018) |
|Grid Search | - | Y | [Agarwal, Dudik, & Wu (2019)](#Agarwal2019); [Agarwal, Beygelzimer, Dudik, Langford, & Wallach (2018)](#Agarwal2018) |


# Conclusion
## How fair is fair enough?

While this specific solution may not always be available, there will likely always be options for potential improvement. Yet, we know from the [Impossibility Theorem](#impossibility) that we cannot produce a model that is perfectly fair by all measures. So how do we know when to stop?

**The ultimate metric for the fairness of our model is whether our results meet the expectations of the people who are affected by it.** Can we justify our results to them. Will they stand up to the standards of the community, the healthcare practitioners, and most importantly, the patients?


## Final Remarks

Just as data and model performance can drift over time, so too can prediction fairness. We recommend integrating fairness evaluation with your modeling pipeline as a form of continuous process improvement. By regularly evaluating multiple measures of fairness at once you can ensure that it continues to meet the expectations of the stakeholders.

For more examples of fairness measurement using the FairMLHealth tool, see the [Evaluating Fairness in Binary Classification Tutorial Notebook](../examples_and_tutorials/Tutorial-EvaluatingFairnessInBinaryClassification.ipynb), [ Example-BinaryClassificationTemplate Notebook](../examples_and_tutorials/ Example-BinaryClassificationTemplate.ipynb) in our examples_and_tutorials section. There are also a number of additional references at the bottom of this page, as well as in our [Documentation Folder](README.md).




# References


## Additional Resources and Tutorials <a id="additional_resources"></a>

[Fairness Indicators](https://www.tensorflow.org/tfx/guide/fairness_indicators) - A TensorFlow-enabled suite designed for simulation of long-term effects caused by the implementation of a machine learning model.

["A Tutorial on Fairness in Machine Learning" by Ziyuan Zhong](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb) - A popular reference with additional background on fairness metrics.

["How to define fairness to detect and prevent discriminatory outcomes in Machine Learning" by Valeria Cortez](https://towardsdatascience.com/how-to-define-fairness-to-detect-and-prevent-discriminatory-outcomes-in-machine-learning-ef23fd408ef2#:~:text=Demographic%20Parity%20states%20that%20the,%E2%80%9Cbeing%20shown%20the%20ad%E2%80%9D) - Another source for background on fairness metrics.

## Citations
<a id="agniel2018biases"></a>
Agniel D, Kohane IS, & Weber GM (2018). Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. Bmj, 361. Retrieved from [https://www.bmj.com/content/361/bmj.k1479](https://www.bmj.com/content/361/bmj.k1479)

<a id="Agarwal2018"></a>
Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In International Conference on Machine Learning (pp. 60-69). PMLR. Available through [arXiv preprint:1803.02453](https://arxiv.org/pdf/1803.02453.pdf).

<a id="Agarwal2019"></a>
Agarwal, A., Dudik, M., & Wu, Z. S. (2019, May). Fair regression: Quantitative definitions and reduction-based algorithms. In International Conference on Machine Learning (pp. 120-129). PMLR. Available through https://arxiv.org/pdf/1905.12843.pdf

<a id="bantilan2018_ref"></a>
Bantilan N (2018). Themis-ml: A fairness-aware machine learning interface for end-to-end discrimination discovery and mitigation. Journal of Technology in Human Services, 36(1), 15-30. Retrieved from [https://www.tandfonline.com/doi/abs/10.1080/15228835.2017.1416512](https://www.tandfonline.com/doi/abs/10.1080/15228835.2017.1416512)

<a id="barocas2016_ref"></a>
Barocas S, & Selbst AD (2016). Big data's disparate impact. California Law Review, 104, 671. Retrieved from [https://www.cs.yale.edu/homes/jf/BarocasDisparateImpact.pdf](https://www.cs.yale.edu/homes/jf/BarocasDisparateImpact.pdf)

Bellamy RK, Dey K, Hind M, Hoffman SC, Houde S, Kannan K, ... & Nagar S (2018). AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. arXiv Preprint. [arXiv:1810.01943.](https://arxiv.org/abs/1810.01943). See Also [AIF360 Documentation](http://aif360.mybluemix.net/)

Bird S, Dudík M,  Wallach H,  & Walker K (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Research. Retrieved from [https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf). See Also [FairLearn Reference](https://fairlearn.github.io/).

<a id="dwork2012_ref"></a>
Dwork C, Hardt M, Pitassi T, Reingold O, & Zemel R (2012, January). Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference (pp. 214-226). Retrieved from [https://arxiv.org/pdf/1104.3913.pdf](https://arxiv.org/pdf/1104.3913.pdf)

<a id="fourfifths_ref"></a>
Equal Employment Opportunity Commission, & Civil Service Commission, Department of Labor & Department of Justice (1978). Uniform guidelines on employee selection procedures. Federal Register, 43(166), 38290-38315. Retrieved from [http://uniformguidelines.com/uniformguidelines.html#18](http://uniformguidelines.com/uniformguidelines.html#18)

<a id="hardt2016_ref"></a>
Hardt M, Price E, & Srebro N (2016). Equality of opportunity in supervised learning. In Advances in neural information processing systems (pp. 3315-3323). Retrieved from [http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf](http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)

<a id="hcup_ref"></a>
Healthcare Cost and Utilization Project (HCUP) (2017, March). HCUP CCS. Agency for Healthcare Research and Quality, Rockville, MD. Retrieved from [https://www.hcup-us.ahrq.gov](https://www.hcup-us.ahrq.gov)

Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, & Mark RG (2016). Scientific Data. MIMIC-III, a freely accessible critical care database. DOI: 10.1038/sdata.2016.35. Retrieved from [http://www.nature.com/articles/sdata201635](http://www.nature.com/articles/sdata201635)

<a id="kearns"></a>
Kearns, M., Neel, S., Roth, A., & Wu, Z. S. (2018, July). Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In International Conference on Machine Learning (pp. 2564-2572). PMLR. Available through http://proceedings.mlr.press/v80/kearns18a.html

<a id="kim2018_ref"></a>
Kim M, Reingol O, & Rothblum G (2018). Fairness through computationally-bounded awareness. In Advances in Neural Information Processing Systems (pp. 4842-4852). Retrieved from [https://arxiv.org/pdf/1803.03239.pdf](https://arxiv.org/pdf/1803.03239.pdf)

<a id="naacp"></a>
National Association for the Advancement of Colored People (NAACP) (2012). Criminal Justice Fact Sheet. NAACP. Retrieved from https://naacp.org/resources/criminal-justice-fact-sheet.

<a id="romei2014_ref"></a>
Romei, A., & Ruggieri, S. (2014). A multidisciplinary survey on discrimination analysis. The Knowledge Engineering Review, 29(5), 582-638. Retrieved from [https://www.cambridge.org/core/journals/knowledge-engineering-review/article/multidisciplinary-survey-on-discrimination-analysis/D69E925AC96CDEC643C18A07F2A326D7](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/multidisciplinary-survey-on-discrimination-analysis/D69E925AC96CDEC643C18A07F2A326D7)

<a id="russell2017_ref"></a>
Russell C, Kusner MJ, Loftus J, & Silva R (2017). When worlds collide: integrating different counterfactual assumptions in fairness. In Advances in Neural Information Processing Systems (pp. 6414-6423). Retrieved from [https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf](https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf)

<a id="shorrocks_ref"></a>
Shorrocks AF (1980). The class of additively decomposable inequality measures. Econometrica: Journal of the Econometric Society, 613-625. Retrieved from [http://www.vcharite.univ-mrs.fr/PP/lubrano/atelier/shorrocks1980.pdf](http://www.vcharite.univ-mrs.fr/PP/lubrano/atelier/shorrocks1980.pdf)

<a id="speicher2018_ref"></a>
Speicher T, Heidari H, Grgic-Hlaca N, Gummadi KP, Singla A, Weller A, & Zafar M B (2018, July). A unified approach to quantifying algorithmic unfairness: Measuring individual &group unfairness via inequality indices. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2239-2248). Retrieved from [https://arxiv.org/pdf/1807.00787.pdf](https://arxiv.org/pdf/1807.00787.pdf)

<a id="zemel2013_ref"></a>
Zemel R, Wu Y, Swersky K, Pitassi T, & Dwork C (2013, February). Learning fair representations. International Conference on Machine Learning (pp. 325-333). Retrieved from [http://proceedings.mlr.press/v28/zemel13.pdf](http://proceedings.mlr.press/v28/zemel13.pdf)

<a id="zafar2017_ref"></a>
Zafar MB, Valera I, Gomez Rodriguez, M, & Gummadi KP (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).  https://arxiv.org/pdf/1610.08452.pdf

## Other Fairness Libraries of Note
* [Aequitas](https://github.com/dssg/aequitas)
* [AIF360](https://github.com/IBM/AIF360)
* [Awesome Fairness in AI](https://github.com/datamllab/awesome-fairness-in-ai)
* [Dalex](https://dalex.drwhy.ai/)
* [Fairlearn](https://github.com/fairlearn/fairlearn)
* [Fairness Comparison](https://github.com/algofairness/fairness-comparison)
* [FAT Forensics](https://github.com/fat-forensics/fat-forensics)
* [ML Fairness Gym](https://github.com/google/ml-fairness-gym)
* [Themis ML](https://themis-ml.readthedocs.io/en/latest/)


