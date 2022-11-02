# Generative Design for Precision Geo-Interventions

Richard Wen, Songnian Li  
rrwen.dev@gmail.com, snli@ryerson.ca

* [Read Paper (PDF)](docs/wen_li_2022_gi4dm_paper.pdf)
* [View Slides (PDF)](wen_li_2022_gi4dm_slides.pdf)

## Abstract

The purpose of this research is to develop an approach for a Spatial Decision Support System (SDSS) that integrates Geographic Information Systems (GIS), Automated Machine Learning (AutoML), and Hyperparameter Optimization (HPO) to generate precision geo-interventions based on standardized geospatial data and user design constraints. The geo-intervention generation approach involves three steps: (1) Geo-binning, (2) AutoML, and (3) Prediction Optimization. Geo-binning is used to standardize geospatial data into regularized grids as inputs into AutoML models. Prediction optimization generates geo-interventions by applying user-design constraints and optimizing AutoML model output to find optimized input variables that form precise geo-interventions. An experiment in reducing road traffic collisions using infrastructural changes in Toronto, Ontario, Canada was done to evaluate the geo-intervention generation approach. The results of the experiment found that changing the number of schools, red light cameras, and transit shelters in high traffic areas could potentially halve the total number of traffic collisions according to a 80 by 80 geo-binned grid Auto-Sklearn model with a Mean Absolute Error (MAE) of 117.68. It was also found that user design constraints heavily affected the prediction optimization step as when the areas were altered to an alternative grid of cells with scarce infrastructure, the number of predicted collisions rose by 6127 collisions. Thus, limitations of this study included subjectivity in user design constraints, scalability, and interactivity. Future work involves improving modelling/optimization efficiency and developing an interactive interface for exploring generated precision geo-interventions.

## History

| Date | Note |
| --- | --- |
| Tuesday, October 23, 2022 | Presentation slides submitted to Urban Geo-Information track for November 2, 2022 Urban Geo-Info Session at 8:30-10am ([PDF](docs/wen_li_2022_gi4dm_slides.pdf) \| [PPT](docs/wen_li_2022_gi4dm_slides.pptx)) |
| Tuesday, September 20, 2022 | Full paper submitted to the Urban Geo-Information track ([PDF](docs/wen_li_2022_gi4dm_paper.pdf) \| [DOC](docs/wen_li_2022_gi4dm_paper.doc)) |
| Thursday, September 1, 2022 | Abstract accepted to the Urban Geo-Inforomation track ([PDF](docs/wen_li_2022_gi4dm_abstract.pdf) \| [DOC](docs/wen_li_2022_gi4dm_abstract.doc)) |
| Thursday, August 18, 2022 | Abstract submitted to the Urban Geo-Information track ([PDF](docs/wen_li_2022_gi4dm_abstract.pdf) \| [DOC](docs/wen_li_2022_gi4dm_abstract.doc)) |

## Install

In Windows:

1. Install [Anaconda3](https://www.anaconda.com/)
2. Install [Windows Subsystem](https://docs.microsoft.com/en-us/windows/wsl/) `wsl --install`
3. Install [dos2unix](https://dos2unix.sourceforge.io/) `wsl sudo apt install dos2unix`
    * **Note**: You may be promoted for a password due to `sudo` privileges
4. Run `bin/setup.bat` to convert scripts in `bin` folder for windows use
5. Enter a Windows Subsystem terminal with `wsl`
6. Run `bin/install` to create a `conda` environment
7. Activate the `conda` environment (named `gi4dm-2022-paper`)

```
wsl --install
wsl sudo apt install dos2unix
bin\setup
wsl
source bin/install.sh
source bin/activate.sh
```

In Linux/Mac OS:

1. Install [Anaconda3](https://www.anaconda.com/)
2. Run `bin/install` to create a `conda` environment
3. Activate the `conda` environment (named `gi4dm-2022-paper`)


```
source bin/install.sh
source bin/activate.sh
```

## Contact

Richard Wen <rwen@ryerson.ca> and Songnian Li <snli@ryerson.ca>
