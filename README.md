![hierarchicallearning](fig/GITHUB_header_HierarchicalLearning.jpg)

# Hierarchical Learning

This repository presents Hierarchical Learning, a package for hierarchical forecasting of time series developed in the open-access journal article [Hierarchical learning, forecasting coherent spatio-temporal individual and aggregated building loads](https://doi.org/10.1016/j.apenergy.2023.121510). The repository reports both the python implementations of the article using the open data set of the [building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) as well as smaller tutorial jupiter notebooks to illustrate usage of the developed classes.

To get started, simply clone this repository on your computer or Fork it via GitHub. After installing dependencies from  the `requirements.txt` file, the code should run properly.

### Citation
If you find this code useful and use it in your work, please reference our journal article:

[Leprince, J., Madsen, H., Kloppenborg Møller, J. and Zeiler, W., 2023. Hierarchical learning, forecasting coherent spatio-temporal individual and aggregated building loads. Applied Energy, p.112095.](https://doi.org/10.1016/j.apenergy.2023.121510)

```
@article{LEPRINCE2023121510,
title = {Hierarchical learning, forecasting coherent spatio-temporal individual and aggregated building loads},
journal = {Applied Energy},
volume = {348},
pages = {121510},
year = {2023},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2023.121510},
url = {https://www.sciencedirect.com/science/article/pii/S0306261923008747},
author = {Julien Leprince and Henrik Madsen and Jan Kloppenborg MÃ¸ller and Wim Zeiler},
}
```

## Repository structure
```
hierarchical learning
├─ fig                                              <- repository readme figures
├─ io
|   ├─ input                                        <- input data-set of the Fox site
|   └─ output
|       ├─ regressor                                <- fitted hierarchical models and saved performances
|       ├─ results                                  <- processed results for forecast performance plots
|       └─ visuals                                  <- forecast performance plots
├─ src
|   ├─ hl                                           <- hierarchical learning package
|   ├─ publication
|   |   ├─ 0 preprocessing                          <- BDG2 data preprocessing
|   |   ├─ 1 cluster2tree                           <- reducing spatial hierarchical dimension with clustering
|   |   ├─ 2 hierarchical forecasting <dimension>   <- hierarchical learning and reconciliation
|   |   ├─ 3 base forecasting <dimension>           <- base forecast and reconciliation
|   |   ├─ 4 results processing                     <- forecast performance result processing
|   |   └─ 5 results visualization                  <- forecast performance visualizations
|   └─ tutorial
|       ├─ 0 manipulating trees                     <- tree class initialization exemplified on all dimensions
|       └─ 1 forecasting trees                      <- tree regressor initialization and training
└─ README.md                                        <- README for developers using this code
```


## Authors

[Julien Leprince](https://github.com/JulienLeprince),
Prof. [Henrik Madsen](https://henrikmadsen.org/),
Ass. Prof. [Jan Kloppenborg Møller](https://orbit.dtu.dk/en/persons/jan-kloppenborg-m%C3%B8ller),
Prof. [Wim Zeiler](https://www.tue.nl/en/research/researchers/wim-zeiler/).


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details