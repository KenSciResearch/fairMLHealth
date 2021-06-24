# Changelog

All notable changes to this project will be documented in this file. Please do follow the format according to https://keepachangelog.com/en/1.0.0/.

_“Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.”_ - Rumi


## [0.1.10] - 2021-06-08
### Testing Updates
- Validation added to check http & https URLs in notebooks and markdown documents.
- Windows added as agent to CI pipeline (to test on multiple OS)

## [0.1.9] - 2021-04-19
### Reporting Updates
- Both standard and stratified reporting functions are now contained in a single module, and new APIs have been put in place to facilitate both classification and regression reports


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

