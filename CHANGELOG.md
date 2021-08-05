# Changelog

All notable changes to this project will be documented in this file. Please do follow the format according to https://keepachangelog.com/en/1.0.0/.

_“Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.”_ - Rumi



## [0.1.26] - 2021-08-05
### Report Update
- Allow flexible return type for any report.compare result
- Update tests for report.py
- Update method for testing if a model predicts probabilities in FairCompare
- Allow user to skip performance measures when using report.compare


## [0.1.25] - 2021-08-05
### Custom Boundary Bug
- Fix bug that prevented custom boundaries from updating

## [0.1.24] - 2021-08-03
### Undefined Ratios and Fairness Metric Testing
- Added code to return NaN values in cases where a fairness ratio has a zero denominator
- Updates to test__fairnessmetrics to increase robustness.
- Ceiling placed on the version number for AIF360 in setup.py

## [0.1.23] - 2021-08-03
### Correct Column References
- Bug: private names appearing in measure tables
- Updated references to private columns

## [0.1.22] - 2021-08-03
### Report Update
- report.measure_model removed in favor of using single function report.compare (neé compare_models) for all reports


## [0.1.21] - 2021-08-03
### Flag-Cohort Correction
- Updates to flagging function to facilitate use for cohorted tables

## [0.1.20] - 2021-08-03
### Analytical Update
- Bootstrap_significance updated to accept any function that returns a p-value, and supporting functions for kruskal and chisquare were added
- Cohorting enabled for measure.data and measure.performance
- Fixed bugs uncovered during testing

## [0.1.19] - 2021-08-03
### Testing Update
- New tests added
- Testing utilities renamed for clarity
- Fixed bugs uncovered during testing


## [0.1.18] - 2021-07-31
### Regression Measure Flagging
- Updated flagging feature to enable custom boundaries and flagging for regression measures
-  Related updates to the __validation module. This includes the addition of class-related validation and a first pass at improved type-hinting throughout the tool.
- Minor changes to text displays.

## [0.1.17] - 2021-07-31
### Library Reorganization

- Major changes to the structure of the library intended to make it more intuitive:
    - fairMLHealth modules were renamed to clarify scope and purpose. This will facilitate adding new features in upcoming PRs:
        - **reports.py** is now named **measure.py**
        - **model_comparison** is now named **report.py**
        - utils.py is now split into stat_utils.py and __utils.py
            - __utils.py contains back-end functions for which validation is assumed to have been run
            - stat_utils.py contains supplemental functions that may be useful in statistical analysis or data munging
    - newly-renamed measure.py has also been reorganized so that functions are in alphabetical order and names are more intuitive. (this should have been a separate PR. apologies in advance).
    - load_mimic_data.py is now "__mimic_data.py"
    - the "tutorials_and_examples" folder has been renamed "examples_and_tutorials" so that it will be easier to find (e.g. seen adjacent to the "docs" folder)


## [0.1.16] - 2021-07-30
### Documentation Update

- Doctstings on public-facing functions given added detail components of the code.
- The "feature_table" function was moved into utils.py so that it can be used publicly.

## [0.1.15] - 2021-06-25
### Cohorting Feature
- Enables regression versions of analytical functions, with basic measures added for the regression version of bias analysis table.
- Adds tutorial, template, and example notebooks for regression features
- Improves management of significant figures as required to enable flagging of regression outputs

## [0.1.14] - 2021-06-25
### Cohorting Feature
- Adds cohorting wrapper which iterates to create separate analysis tables by group

## [0.1.13] - 2021-06-23
### Flagging Updates
- Add flagging functionality for stratified tables (bias analysis table, performance analysis table, data analysis table)
- Update flagging function s.t. it can be called as an argument in the model_comparison or bias analysis table functions

## [0.1.12] - 2021-06-17
### Stratified Table Updates
- data analysis table now accepts y as a data frame, enabling stratified analysis across multiple targets.
- "Overview" column evaluating all features as one unit ("ALL_FEATURES") is now optional via add_overview argument

## [0.1.11] - 2021-06-17
### Validation Updates
- Validation and preprocessing are now in individual modules to reduce redundancy and improve readability
- Bugs in validation addressed

## [0.1.10] - 2021-06-08
### Testing Updates
- Validation added to check http & https URLs in notebooks and markdown documents.
- Windows added as agent to CI pipeline (to test on multiple OS)

## [0.1.10] - 2021-06-08
### Testing Updates
- Validation added to check http & https URLs in notebooks and markdown documents.
- Windows added as agent to CI pipeline (to test on multiple OS)

## [0.1.9] - 2021-04-19
### Reporting Updates
- Both standard and stratified table functions are now contained in a single module, and new APIs have been put in place to facilitate both classification and regression analysis tables


## [0.1.8] - 2021-04-19
### Installation Improvements
- Fairlearn removed as backend requirement (still required for tutorials)
- Version ranges updated in setup.py (wider range of dependency versions now allowed)
- Troubleshooting documentation added
- Bug fixes implemented

## [0.1.7] - 2021-04-14
### Testing Update
- Add integrated notebook testing
- Enhance contribution documentation

## [0.1.6] - 2021-04-14
### Flexibility Update
- Add method to model comparison allowing comparison of either predictions or models
- Improved validation and exception handling

## [0.1.5] - 2021-02-08
### Documentation Update
- Add "docs" folder containing background information on measuring fairness in ML, quick reference tables for the different metrics and measures, and our current list of recommended outside references and resources
- Update tutorial and examples. Content introducing background on fairness has been moved to the docs folder
- Add development documentation (docs folder): issue template and pull request template
- Update READMEs and resources
- Correct typos

## [0.1.4] - 2020-12-09
### Added and fixed
- Add validations in the model comparison methods
- Allow multiple data inputs to model comparison (one per model), in addition to previous method allowing a single dataset to be used for all models
- Add unit tests
- Fix installation issues (Issue #25)

## [0.1.3] - 2020-12-01
### Added
- Change from "dev" to "integration" as staging branch

## [0.1.2] - 2020-10-28
### Added
- Add CI.

## [0.1.1] - 2020-10-27
### Hotfix
- Corrected critical bug: typo in test for number of protected attributes was rejecting all inputs
- Notebook typos corrected

## [0.1.0] - 2020-10-23
### Permissions
- First public-facing version of repo

