---
title: 'Tyssue: an epithelium simulation library'
tags:
  - Python
  - developmental biology
  - epithelium
  -
  -
authors:
  - name: Guillaume Gay
    orcid:
    affiliation: "1"
  - name: Sophie Theis
    orcid: 0000-0003-4206-8153
    affiliation: "1, 2"
affiliations:
 - name: Morphogénie Logiciels, 32110 St Martin d’Armagnac, France.
   index: 1
 - name: LBCMCP, Centre de Biologie Intégrative (CBI), Université de Toulouse, CNRS, UPS, Toulouse 31062, France.
   index: 2
date: 2 october 2020
bibliography: paper.bib

---
# Summary
<div align="justify">
The tyssue library seeks to provide a unified interface to implement bio-mechanical models of living tissues. It's main focus is on vertex based epithelium models. tyssue allows to model 2D, apical 3D or full 3D epihelium based on two kind of resolutions : quasi-static equilibrium of a potential energy or a viscous solver following Euler resolution. Tissue is a modular library. Starting with the same tissue geometry, the choice of constraints/models/parameters increase the possibility to answer on different biological question.

[**Figure avec un exemple de tissue 2D, 3D apical et 3D ?**]
</div>

# Statement of Need
<div align="justify">
Studying tissue morphogenesis is complicated. Such as those process are placed at the embryonic stage of development, it can be complicated to perturb them with genetic tools, or even to capture when the process takes only few minutes. To execute complex morphogenetic movements, epithelia are driven by in-plane forces, like constriction of apical cell surface [@Heer:2017], and/or out-plane forces, like the apico-basal cable in apoptotic cell [@Monier:2015]. Modeling those process  help to understand how tissue acquires their shape. Several vertex models have been developed in the past few years to describe the physics of epithelia [@Alt:2017].

Cell elimination or division is the most commun processus which takes part in epithelia remodelling. Those cellular process need to be included in many model. Okuda is one on the first author to propose a model which include cell elimination in 3D vertex model [@Okuda:2012].

In an other way, Swcharz focused his works on tissue rigidity which allows or not cell deplacement in an epithelium, based on relation between area and perimeter on a cell [@Bi:2015]. This kind of relation permit to calibrate epithelium degree of freedom to permit/facilitate to one cell to move in the tissue or not.

tyssue combine all those piece of puzzle to propose the most versatile model to answer to different biological question.


~~Vertex models is used to understand the bridge between cellular shape, the fores generated inside epithelial cells and mechanical constraints externally imposed on the tissue to lead tissue morphogenesis.~~
</div>


# Mention
<div align="justify">
Tyssue library has been used in several studies of epithelium morphogenesis, such as leg folding and mesoderm invaginaion in *Drosophila melanogaster* ([@Monier:2015], [@Gracia:2019], [@Martin:2020]).
</div>

# Acknowledgements
<div align="justify">
The work of this paper was supported by grants from the European Research Council (ERC) under the European Union Horizon 2020 research and innovation program (grant number EPAF: 648001), and from the Association Nationale de la recherche et de la Technologie (ANRT).
</div>
# Correspondence
Please contact guillaume@morphogenie.fr

# Code
<div align="justify">
tyssue is written in Python 3. Code and detailed installation instructions can be found [here](https://github.com/DamCB/tyssue/blob/master/INSTALL.md). Continuous integration is performed with [Travis](**adresse**). The associated code coverage can be found at [CodeCov](**adresse**).
</div>

