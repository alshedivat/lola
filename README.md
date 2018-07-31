Learning with Opponent-Learning Awareness
=========================================

Implements the LOLA ([AAMAS'18](https://arxiv.org/abs/1709.04326)) and LOLA-DiCE ([ICML'18](https://arxiv.org/abs/1802.05098)) algorithms.


**Further resources:**
- A pytorch implementation of LOLA-DiCE is available at https://github.com/alexis-jacq/LOLA_DiCE.
- A colab notebook with the nummerical evalution for DiCE is available at https://goo.gl/xkkGxN.

## Installation

To run the code, you need to pip-install it as follows:

```bash
$ pip install -e .
```

After installation, you can run different experiments using the run scripts provided in `scripts/`.
Use `run_lola.py` and `run_tournament.py` for running experiments from the [AAMAS'18 paper](https://arxiv.org/abs/1709.04326).
Use `run_lola_dice.py` for reproducing experiments from the [ICML'18 paper](https://arxiv.org/abs/1802.05098).
Check out `notebooks/` for IPython notebooks with plots.

**Note:** this code is not tested on GPU, so there might be unexpected issues.

*Disclaimer:* This is a research code release that has not been tested beyond the use cases and experiments discussed in the original papers.

## Contribution

Contributions to further enhance and improve the code are welcome.
Please email `jakob.foerster` at `cs.ox.ac.uk` and `alshedivat` at `cs.cmu.edu` with comments and suggestions.


## Citations

LOLA:
```bibtex
@inproceedings{foerster2018lola,
  title={Learning with opponent-learning awareness},
  author={Foerster, Jakob and Chen, Richard Y and Al-Shedivat, Maruan and Whiteson, Shimon and Abbeel, Pieter and Mordatch, Igor},
  booktitle={Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={122--130},
  year={2018},
  organization={International Foundation for Autonomous Agents and Multiagent Systems}
}
```

DiCE:
```bibtex
@inproceedings{foerster2018dice,
  title={{D}i{CE}: The Infinitely Differentiable {M}onte {C}arlo Estimator},
  author={Foerster, Jakob and Farquhar, Gregory and Al-Shedivat, Maruan and Rockt{\"a}schel, Tim and Xing, Eric and Whiteson, Shimon},
  booktitle ={Proceedings of the 35th International Conference on Machine Learning},
  pages={1524--1533},
  year={2018},
  volume={80},
  series={Proceedings of Machine Learning Research},
  address={Stockholmsmässan, Stockholm Sweden},
  month={10--15 Jul},
  publisher={PMLR},
  pdf={http://proceedings.mlr.press/v80/foerster18a/foerster18a.pdf},
  url={http://proceedings.mlr.press/v80/foerster18a.html},
}
```

## License

MIT
