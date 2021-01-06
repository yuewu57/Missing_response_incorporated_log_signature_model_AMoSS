# Missing_response_incorporated_log_signature_model_AMoSS

This repository stores the resources to rework on ASRM/QIDS/EQ-5D/GAD-7 data collected from AMoSS study through the use of log-signature features. <sup>1<sup>

Basically, the analysis is log-signature-based and integrated missing responses. <sup>2<sup>  

To focous on our proposed method, comparision methods were not included in this repository. But request can be made through email.
  
  
Getting Started
---------------

Once the repo has been cloned locally, setup a python environment with ``python==3.6`` and run ``pip install -r requirements.txt``.

As the data were collected pre-GDPR and contained sensitive personal data, it cannot be placed into a publicly accessible repository.

Patient Group
---------------
| Diagnosis Group   |  Class|
|------------|--------|
|Borderline|0|
|Healthy|1|
|Bipolar|2|


Structure
---------------
| File    | Task| Section in manuscript|
|----------|------------|--------|
|``src/features/data_preprocessing_general.py``| cleaning/Aligning raw data|Data|
|``notebooks/hists_plotting.ipynb``| histograms for data |Data|
|``src/features/feature_extracting_general.py ``| encoding missingness and output log-signature features|The workflow|
|``src/models/weeklydata_MRLSM.ipynb``| results (include plots) for MRLSM |Results|


References
---------------
  1. Tsanas A, Saunders KE, Bilderbeck AC, Palmius N, Osipov M, Clifford GD, Goodwin GΜ, De Vos M. Daily longitudinal self-monitoring of mood variability in bipolar disorder and borderline personality disorder. *Journal of affective disorders*. 2016 Nov 15;205:225-33. doi:10.1016/j.jad.2016.06.065
 
  2. Lyons T. Rough paths, signatures and the modelling of functions on streams. *arXiv preprint arXiv:1405.4537*. 2014 May 18.

