# Generative Design for Precision Geo-Interventions

Conference paper for Gi4DM and Urban Geo-Informatics 2022.

## History

| Date | Note |
| --- | --- |
| Tuesday, September 20, 2022 | Full paper submitted to the Urban Geo-Information track ([DOC](docs/wen_li_2022_gi4dm_paper.doc) \| [PDF](docs/wen_li_2022_gi4dm_paper.pdf)) |
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
