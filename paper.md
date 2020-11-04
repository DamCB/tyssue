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

</div>


<center>
![figure1](doc/illus/figure1.jpg  "figure1")
</center>


# Statement of Need
<div align="justify">

Studying tissue morphogenesis is complicated. Such as those process are placed at the embryonic stage of development, it can be complicated to perturb them with genetic tools, or even to capture when the process takes only few minutes. To execute complex morphogenetic movements, epithelia are driven by in-plane forces, like constriction of apical cell surface [@Heer:2017], and/or out-plane forces, like the apico-basal cable in apoptotic cell [@Monier:2015]. Modeling those process help to understand how tissue acquires their shape and overstep biological limit. Several vertex models have been developed in the past few years to describe the physics of epithelia [@Alt:2017].

Tyssue is a python library which provide solution to model tissue as a vertex geometry. A vertex model define a tissue as an assembly of vertex and edges, which can form polygonal face (in 2D) or polyhedron (in 3D). Now we assume that cell junction are straight lines, and there is only one edge between two neighbouring cell. The way we construct our model, each edge is defined by two half-edges which belong to one of the two neighbouring cell (**figure2 A**). The library implements concepts and mechanisms common to all vertex models:

### Topological aspect
Few basic cellular process are implemented in our library such as cell elimination, division or cell neighbouring change. We implemented those process based on previous works.

Cell neighbouring change - also called T1 transition - is based on the junction length. When a junction length goes below a threshold length, cell can swap neighbouring cell. According to Finegan et al. work, cell does not swap as soon as they have junction which reach the threshold rather than according with a probability [@Finegan:2019].

Cell division is a relative simple process to implement as we suppose that the two daughters cell will have equal area which is half the area of the mother cell. This process occurs according to a plane division axis, which can be aleatory or constraint. The division plane cross two edges of the mother cell and the new junction will be created at the middle of those two junctions [@Brodland:2002].

Cell elimination append when a cell area reached a small area threshold. When it appens, cell start its elimination process by reduce its number of neighbors until it remains three using T1 transition. Then this cell is eliminated and the three vertex are merged to created a new vertex. [@Okuda:2015]

~~Swap occurs according to a certain number of parameters are completed, which we can simplify with a probability of swapping. ~~


### Mechanical aspect

Several previous works lay the foundations for more recent models.
Farhadifar et al. characterize qualitatively and quantitatively the importance of cell proliferation and elimination in the number of neighbouring cell distribution by using energy function with cell elasticity and junctional forces.

**honda ??**

Bi focused his works on tissue rigidity which allows or not cell deplacement in an epithelium, based on relation between area and perimeter on a cell [@Bi:2015]. This kind of relation permit to calibrate epithelium degree of freedom to permit/facilitate to one cell to move in the tissue or not. tyssue combine all those piece of puzzle to propose the most versatile model to answer to different biological question.


<center>
![figure2](doc/illus/figure2.jpg  "figure2")
</center>


**[Comment c'est implémenté, exemple de déclaration de modèle ou lien vers un nb de la doc]**

tyssue library has already been used in several studies with different context of epithelia morphogenesis, such as leg folding and mesoderm invagination in *Drosophila melanogaster* ([@Monier:2015], [@Gracia:2019], [@Martin:2020]).

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
