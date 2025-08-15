# TULMGAT

**Trajectory-User Linking via Multi-Scale Graph Attention Network**  

This repository provides the implementation of **TULMGAT**, a novel framework for trajectory-user linking that leverages multi-scale graph attention mechanisms. This work has been accepted by the journal *Pattern Recognition* (Elsevier).  

## Citation

If you use this code or refer to TULMGAT in your research, please cite our work:

```bibtex
@article{li2025trajectory,
  title={Trajectory-user linking via multi-scale graph attention network},
  author={Li, Yujie and Sun, Tao and Shao, Zezhi and Zhen, Yiqiang and Xu, Yongjun and Wang, Fei},
  journal={Pattern Recognition},
  volume={158},
  pages={110978},
  year={2025},
  publisher={Elsevier}
}
```

Article link: [https://www.sciencedirect.com/science/article/abs/pii/S0031320324007295](https://www.sciencedirect.com/science/article/abs/pii/S0031320324007295)  

## Environment

- Python 3.10+  
- Numpy 1.24.2  
- PyTorch 1.13.1  
- torch-geometric 2.2.0
- gensim==4.3.2
- scipy==1.12.0

## Baselines

TULMGAT is compared with several traditional trajectory-user linking methods:

| Method | Description | Publication |
|--------|-------------|------------|
| **TULER** | Identifying Human Mobility via Trajectory Embeddings | IJCAI 2017 [Paper](https://www.ijcai.org/Proceedings/2017/0234.pdf) |
| **TULVAE** | Trajectory-User Linking via Variational AutoEncoder | IJCAI 2018 [Paper](https://www.ijcai.org/Proceedings/2018/0446.pdf) |
| **TULAR** | Trajectory-User Link with Attention Recurrent Networks | ICPR 2020 [Paper](https://ailb-web.ing.unimore.it/icpr/media/posters/11413.pdf) |
| **STULIG** | Toward Discriminating and Synthesizing Motion Traces Using Deep Probabilistic Generative Models | TNNLS 2021 [Paper](https://ieeexplore.ieee.org/document/9165954) |
| **GNNTUL** | Trajectory-User Linking via Graph Neural Network | ICC 2021 [Paper](https://ieeexplore.ieee.org/document/9500836) |
| **TULRN** | TULRN: Trajectory user linking on road networks | WWWJ 2023 [Paper](https://link.springer.com/article/10.1007/s11280-022-01124-0) |

## Usage

1. **Prepare dataset**: Before running `tulmgat.py`, ensure that your dataset is properly set up. The dataset should follow the format expected by the data loader.
- PS：We have already provided data and embedding files for Gowalla user=247 in `datasets`.  

2. **Data loader**: The script `datasets/data_loader.py` provides functionality to load and preprocess your dataset. Adjust paths as needed.  

3. **Run TULMGAT**:  `tulmgat.py`

