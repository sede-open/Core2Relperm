<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#how-to-cite">How to Cite</a></li>
  </ol>
</details>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sede-open/Core2Relperm">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Core2Relperm</h3>

  <p align="center">
    A Python library for interpreting core flooding experiments
    <br />
    <a href="https://github.com/sede-open/Core2Relperm"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/sede-open/Core2Relperm">View Demo</a>
    ·
    <a href="https://github.com/sede-open/Core2Relperm/issues">Report Bug</a>
    ·
    <a href="https://github.com/sede-open/Core2Relperm/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/sede-open/Core2Relperm)

For modelling studies of underground storage of carbon dioxide and hydrogen, transport in the <A HREF="https://en.wikipedia.org/wiki/Vadose_zone">vadoze zone</a>, contaminant hydrology as well as hydrocarbon recovery, it is important to have a consistent set of relative permeability and capillary pressure-saturation functions as inputs for numerical reservoir models in order to assess risks and uncertainties and provide forward-models for different scenarios.
Such relative permeability and capillary-pressure saturations functions are typically obtained in <A HREF="https://en.wikipedia.org/wiki/Special_core_analysis">Special Core Analysis (SCAL)</a> where core flooding experiments are a central element (see also <A HREF="https://www.scaweb.org/">The Society of Core Analysts</a>). Interpreation of such core flooding experiments by analytical approximations has several disadvantages and instead, interpretation by inverse modelling is the preferred approach.
This project has been created to provide a standalone Python tool for the interpretation of such core flooding experiments. 
It contains 
<ul>
  <li>a 1D numerical flow solver (Darcy fractional flow solver with capillarity in 1D) and</li>
  <li>an inverse modelling framework which is utilizing the optimization package called <A HREF="https://lmfit.github.io/lmfit-py/">lmfit</a> from Python</li>
</ul>
The inverse modelling framework is in its default version a least-squares fit using the Levenberg-Marquardt algorithm. It essentially performs a least-squares fit of the numerical solution of a set of partial differential equations (which are numerically solved by the flow solver) to numerical data. The Jacobian is automatically computed numerically in the background by the lmfit package. 
The flow solver is accelerated with the <A HREF="https://numba.pydata.org/">numba</a> just-in-time compiler which makes the flow solver code run in just about 50 ms. 
For a few tens of iterations required for a typical inverse modelling with least-squares fit, the code runs just in a few seconds. One can also change an option in the lmfit package (only a single line) to using the <A HREF="https://emcee.readthedocs.io/en/stable/">emcee</a> <A HREF="https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo">Markov chain Monte Carlo (MCMC)</a> package. About 10,000-20,000 iterations will run in a few hours in single-threaded mode. The advantage of using the MCMC approach is that one can address problems non-uniqueness and <A HREF="https://emcee.readthedocs.io/en/stable/tutorials/line/">non-Gaussian errors</a>. 

Flow simulator code and inverse modelling framework are research code. The 1D flow code has been validated against benchmarks developed by <A HREF="http://jgmaas.com/">Jos Maas</a> and respective benchmark examples are included as examples. The inverse modelling framework has been validated in a series of publications

1. S. Berg, E. Unsal, H. Dijk, Non-Uniqueness and Uncertainty Quantification of Relative Permeability Measurements by Inverse Modelling, <A HREF="https://www.sciencedirect.com/science/article/pii/S0266352X20305279?dgcid=author">Computers and Geotechnics 132, 103964, 2021.</a>

2. S. Berg, E. Unsal, H. Dijk, Sensitivity and uncertainty analysis for parameterization of multi phase flow models, <A HREF="https://doi.org/10.1007/s11242-021-01576-4">Transport in Porous Media 140(1), 27-57, 2021.</a>

3. S. Berg, H. Dijk, E. Unsal, R. Hofmann, B. Zhao, V. Ahuja, Simultaneous Determination of Relative Permeability and Capillary Pressure from an Unsteady-State Core Flooding Experiment ? <A HREF="https://doi.org/10.1016/j.compgeo.2024.106091">Computers and Geotechnics 168, 106091, 2024.</a> 

4. R. Lenormand, K. Lorentzen, J. G. Maas and D. Ruth
COMPARISON OF FOUR NUMERICAL SIMULATORS FOR SCAL EXPERIMENTS 
<A HREF="https://www.jgmaas.com/SCA/2016/SCA2016-006.pdf">SCA2016-006</a>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Read the paper to get some background info. Then install your favorite Python distribution of you don't already have one (we used Anaconda),
install required libraries, download the code and run the examples.


### Dependencies

The code and examples can be run from most modern Python distributions such as Anaconda. You may want to choose a distribution that has `matplotlib`, `numpy` and other standard packages pre-installed. There are a few extra libraries to install:

* pandas (using internally pandas data frames, but also to import/export data)
* lmfit (the engine for the least squares fits)
* emcee (Markov chain Monte Carlo sampler, optional)
* numba (Just In Time compiler)
* seaborn (for statistical data visualization)

### Installation

Quick installation by replicating the environment in Anaconda:

1. Clone the repo
   ```sh
   git clone https://github.com/sede-open/core2relperm.git
   ```
2. Configure conda
   ```sh
   conda update conda
   conda config --set ssl_verify false
   ```
3. Replicate environment using either of the following commands:
   ```sh
   conda env create -f environment.yml 
   ```
4. Activate the environment
   ```sh
   conda activate relperm
   ```

Alternatively, if you face issues with above mentioned quick installtion, you can create the environment and install the Python packages manually as shown below:

1. Create new environment and install required Python libraries
   ```sh
   conda create -n relperm numpy matplotlib numba scipy seaborn pandas lmfit emcee
   ```
2. For rendering in VSCode install the ipykernel package
   ```sh
   conda install ipykernel
   ```
   


<!-- USAGE  -->
## Usage

### Solver Benchmarks

We included 4 SCAL benchmarks from <A HREF="https://www.jgmaas.com">https://www.jgmaas.com</A> 

```
  benchmark_scores_Case1.ipynb
  benchmark_scores_Case2.ipynb
  benchmark_scores_Case3.ipynb
  benchmark_scores_Case4.ipynb
```


that are benchmarking the 2-phase 1D flow solver defined in 
R. Lenormand, K. Lorentzen, J. G. Maas and D. Ruth, COMPARISON OF FOUR NUMERICAL SIMULATORS FOR SCAL EXPERIMENTS, <A HREF="https://www.jgmaas.com/SCA/2016/SCA2016-006.pdf">SCA2016-006</a>




### Inverse Modelling Examples from latest Computers & GeoTechnics Paper

We include 2 examples from the paper <b>S. Berg, H. Dijk, E. Unsal, R. Hofmann, B. Zhao, V. Ahuja, Simultaneous Determination of Relative Permeability and Capillary Pressure from an Unsteady-State Core Flooding Experiment ? <A HREF="https://doi.org/10.1016/j.compgeo.2024.106091">Computers and Geotechnics 168, 106091, 2024.</a> </b>

* Fig. 09
  ```sh
  example_Fig09_USS_dpw+dpo+noSwz.py
  ```
* Fig. 17
  ```sh
  example_Fig17_USS_dpw+dpo+Swz_bumpfloods.py
  ```

The `.py` files are also available as `.ipynb` Jupyter notebooks (generated with <A HREF="https://jupytext.readthedocs.io/en/latest/using-cli.html">jupytext</a>). Respective markdown tags are included in the .py files to generate the formatting e.g. headers in the Jupyter notebooks.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Interpretation of <b>Unsteady-state</b> drainage and imbibition experiments
We include 4 Jupyter notebooks with synthetic data for drainage and imbibition which are based on the example_Fig17_USS_dpw+dpo+Swz_bumpfloods.ipynb. In the "_to-Excel" notebooks the "simulated" experimental data is written to Excel files, then loaded again and interpreted by inverse modelling. In the "_from-Excel" notebooks only the data from the Excel sheets is loaded and then interpreted by inverse modelling.

* Drainage
  ```sh
  USS_synthetic_data_drainage_to-Excel.ipynb
  ```
  which generates <b>expdataHISUSSdrainage.xlsx</b>
  ```sh
  USS_synthetic_data_drainage_from-Excel.ipynb
  ```
* Imbibition
  ```sh
  USS_synthetic_data_imbibition_to-Excel.ipynb
  ```
  which generates <b>expdataHISUSSimbibition.xlsx</b>
  ```sh
  USS_synthetic_data_imbibition_from-Excel.ipynb
  ```

The Excel files and "_from-Excel" notebooks can be used to interpret user experimental data sets. In principle only the Excel sheets have to be modified with the user experimental data sets. But it is important to maintain the format given in the Excel sheets. 

In these notebooks two alternative methods for assessing the uncertainty regions have been added"
1. for each varied fit parameter a normal distribution is generated with 1000 samples and then from these samples mean and standard deviation are determined and plotted as uncertainty ranges
2. Making use of the 
<A HREF="https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html">numpy.random.Generator.multivariate_normal</a> function the covariance matrix from lmfit is direclty used to draw random samples. With a few modifiations i.e. not allowing negative values for fit parameters that should not become negative respective 95% confidence intervals are shows, which is the cleanest way.

Interestingly, the uncertainty ranges from the proper 95% confidence intervals are not dramatically different that the guestimate used in the previous examples.

Also, plotting the correlation matrix has been improved where the varied parameters are automatically determined from the lmfit report. 




### Interpretation of <b>Steady-state</b> imbibition experiments
We include 2 Jupyter notebooks with synthetic data for  imbibition for steady-state flow derived from examples in <A HREF="https://doi.org/10.1007/s11242-021-01576-4">Transport in Porous Media 140(1), 27-57, 2021.</a>
In the "_to-Excel" notebooks the "simulated" experimental data is written to Excel files, then loaded again and interpreted by inverse modelling. In the "_from-Excel" notebooks only the data from the Excel sheets is loaded and then interpreted by inverse modelling.


* Imbibition
  ```sh
  SS_synthetic_data_imbibition_to-Excel.ipynb
  ```
  which generates <b>expdataHISSSimbibition.xlsx</b>
  ```sh
  SS_synthetic_data_imbibition_from-Excel.ipynb
  ```

The Excel files and "_from-Excel" notebooks can be used to interpret user experimental data sets. In principle only the Excel sheets have to be modified with the user experimental data sets. But it is important to maintain the format given in the Excel sheets. 



<!-- ROADMAP -->
## Roadmap

- [ ] Add Changelog
- [ ] Add more examples from previous papers
    - [ ] steady-state experiments
    - [ ] matching real data

<!-- 
See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

It would be great if you could contribute to this project. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Steffen Berg - <A HREF="https://www.linkedin.com/in/steffen-berg-5409a672">LinkedIn</a> - steffen.berg@shell.com

Project Link: [https://github.com/sede-open/Core2Relperm](https://github.com/sede-open/Core2Relperm)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to acknowledge 

* Sherin Mirza, Aarthi Thyagarajan and Luud Heck from Shell supporting the OpenSource release on GitHub 
* Vishal Ahuja supporting the project in general and also the initial release on Github.
* <A HREF="https://www.unileoben.ac.at/universitaet/lehrstuehle/institute/department-petroleum-engineering/lehrstuhl-fuer-reservoir-engineering/">Holger Ott</a>, Omidreza Amrollahinasab (University of Leoben), and <A HREF="http://jgmaas.com/">Jos Maas</a> (PanTerra) for helpful discussions
* Tibi Sorop and Yingxue Wang for reviewing the paper manuscript
* Daan de Kort for providing an updated relperm uncertainty quantification making use of the covariance matrix. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- How-to-Cite -->
## How to Cite

1. S. Berg, H. Dijk, E. Unsal, R. Hofmann, B. Zhao, V. Ahuja, Simultaneous Determination of Relative Permeability and Capillary Pressure from an Unsteady-State Core Flooding Experiment ? <A HREF="https://doi.org/10.1016/j.compgeo.2024.106091">Computers and Geotechnics 168, 106091, 2024.</a> 

2. S. Berg, E. Unsal, H. Dijk, Non-Uniqueness and Uncertainty Quantification of Relative Permeability Measurements by Inverse Modelling, <A HREF="https://www.sciencedirect.com/science/article/pii/S0266352X20305279?dgcid=author">Computers and Geotechnics 132, 103964, 2021.</a>

3. S. Berg, E. Unsal, H. Dijk, Sensitivity and uncertainty analysis for parameterization of multi phase flow models, <A HREF="https://doi.org/10.1007/s11242-021-01576-4">Transport in Porous Media 140(1), 27-57, 2021.</a>



<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[product-screenshot]: images/Core2Relperm-logo.png
[license-url]: https://github.com/sede-open/Core2Relperm/blob/main/license.txt
[linkedin-url]: www.linkedin.com/in/steffen-berg-5409a672
[contributors-url]: https://github.com/sede-open/Core2Relperm/graphs/contributors
[forks-url]: https://github.com/sede-open/Core2Relperm/network/members
[issues-url]: https://github.com/sede-open/Core2Relperm/issues
[stars-url]: https://github.com/sede-open/Core2Relperm/stargazers
[BestReadme-url]: https://github.com/othneildrew/Best-README-Template

