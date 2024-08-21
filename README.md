<!-- ![Alt text](/Bone_Vessel_Segmentation_Diagram.png?raw=true)-->
<div id="top"></div>
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- PROJECT LOGO -->
<div align="center">

  <h3 align="center">MDSPACER</h3>

  <p align="center">
    <u>S</u>patial <u>P</u>attern <u>A</u>nalysis with <u>C</u>omparable and <u>E</u>xtendable <u>R</u>ipley’s K 
    <br />
    <a href="https://github.com/dkermany/spacer3d/issues">Report Bug</a>
    ·
    <a href="https://github.com/dkermany/spacer3d/issues">Request Feature</a>
  </p>
  
  <img src="img/cover-photo.gif">
</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<ol>
  <!-- <li>
    <a href="#project-overview">About The Project</a>
  </li> -->
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#prerequisites">Prerequisites</a></li>
    </ul>
  </li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
  <li><a href="#useful-links">Useful Links</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
<!-- ## Project Overview

### Objectives

Determine if there is any morphological changes to bone blood vessels in the vicinity of identified metastasized breast cancer cells, as well as any 3D spatial relationships between NG2+ stromal cells and disseminated tumor cells.

<p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- ROADMAP -->
<!-- ## Roadmap
- [x] Reproduce previous work in matlab
- [x] Manually labeled 2D samples
- [x] Segmentation Datasets Loaded 

<p align="right">(<a href="#top">back to top</a>)</p> -->
<!-- GETTING STARTED -->
## Getting Started
Libraries and environment setup using the <a href="https://pipenv.pypa.io/en/latest/install/">pipenv</a> library.

### Prerequisites

Clone repository
  ```sh
  git clone git@github.com:Translational-Biophotonics-Laboratory/mdspacer.git
  ```

Install library (virtualenv recommended)
  ```sh
  # After navigating to root directory (where setup.py is)
  pip install -e .
  ```

Install dependencies
  ```sh
  pipenv install --dev
  ```
<!-- USAGE EXAMPLES -->
### Usage
  - [Link to Ripley's K tutorial](ipynb/spacer3d.ipynb)
  - [Link to visualization tutorial](ipynb/plot_rstats.ipynb) 

  Import
  ```python
  from spacer3d.Ripley import CrossRipley, run_ripley 
  ```

  Load points [(N, 3) shape for 3D, (N, 2) for 2D]
  Random points in this example
  ```python
  random_set1 = stats.uniform.rvs(loc=0, scale=100, size=(100,3))
  random_set2 = stats.uniform.rvs(loc=0, scale=100, size=(100,3))
"""
 >   [  35.  668. 1928.]
 >   [  26. 1294. 2030.]
 >   [  26. 1243. 1731.]
 >   [  13.  752.  823.]
 >   [   4. 1226. 1690.]
 >   [   0. 1351. 2243.]
 >           ...
 """
  ```

  Set parameters
  ```python
  # Search radii
  radii=np.arange(2, 67) 

  # Binary mask to define sample space
  volume_mask = np.ones((100, 100, 100))

  # Number of Monte Carlo simulations to run
  n_samples = 5
  ```

  Run MDSPACER K Function
  ```python
    rand_rstats = monte_carlo(points, mask, radii, n_samples=100, n_processes=55, boundary_correction=False)
    results = run_ripley(points, points, mask, radii, n_processes=55, boundary_correction=False)
    rstats = pd.DataFrame(results, columns=["Radius (r)", "K(r)", "L(r)", "H(r)"])
  ```

  Save rstats DataFrames to CSV files for caching and plotting
  ```python
    rand_rstats.to_csv(f"/home/dkermany/ripley_results/{filename}_random_univariate_rstats.csv")
    rstats.to_csv(os.path.join(output_dir, f"{filename}_univariate_rstats.csv"))
  ```

  Plot individuals
  ```python
  plot_individuals(path_to_rstats_folder)
  combined_plot_univariate(path_to_rstats_folder)
  combined_plot_multivariate(path_to_rstats_folder)
  ```

  Plot combined_univariate
  ```python
  combined_plot_univariate(path_to_rstats_folder)
  ```

  Plot combined_multivariate
  ```python
  combined_plot_multivariate(path_to_rstats_folder)
  ```

### Bone Metastasis Data
  Binary masks [(Link)](https://drive.google.com/drive/folders/1X8yfHktQ513SK646tZ_oU5vIuKeu-0Hr?usp=sharing)  
  Point data [(Link)](https://drive.google.com/drive/folders/1BghqrDwZf6uf0CWKvw_sbw2kMEFznNtq?usp=sharing)  
  Example results [(Link)](https://drive.google.com/drive/folders/19Z7xVW3ONVCFWiFHoqCjIoGpvcSV4vBg?usp=sharing)  

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
<p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#top">back to top</a>)</p> -->


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* Dr. Stephen Wong
* Dr. Xiang Zhang
* Dr. Weijie Zhang
* Dr. Jianting Sheng
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Useful Links -->
## Useful Links
* Bone metastasis initiation is coupled with bone remodeling through osteogenic differentiation of NG2+ cells <a href="https://aacrjournals.org/cancerdiscovery/article/doi/10.1158/2159-8290.CD-22-0220/710013/Bone-metastasis-initiation-is-coupled-with-bone">(Article)</a>
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
