# References

Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). [Recurrent neural networks for multivariate time series with missing values](https://doi.org/10.1038/s41598-018-24271-9). *Scientific Reports, 8*(1), 6085.

Chen, T., & Guestrin, C. (2016). [XGBoost: A scalable tree boosting system](https://doi.org/10.1145/2939672.2939785). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., & Galstyan, A. (2019). [Multitask learning and benchmarking with clinical time series data](https://doi.org/10.1038/s41597-019-0103-9). *Scientific Data, 6*(1), 96.

Hochreiter, S., & Schmidhuber, J. (1997). [Long short-term memory](https://doi.org/10.1162/neco.1997.9.8.1735). *Neural Computation, 9*(8), 1735–1780.

Lundberg, S. M., & Lee, S. (2017). [A unified approach to interpreting model predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). *Advances in Neural Information Processing Systems, 30*.

Moor, M., Rieck, B., Horn, M., Jutzeler, C. R., & Borgwardt, K. (2021). [Early prediction of sepsis in the ICU using machine learning: A systematic review](https://doi.org/10.3389/fmed.2021.607952). *Frontiers in Medicine, 8*, 607952.

Paszke, A., et al. (2019). [PyTorch: An imperative style, high-performance deep learning library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html). *Advances in Neural Information Processing Systems, 32*.

Pedregosa, F., et al. (2011). [Scikit-learn: Machine learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html). *Journal of Machine Learning Research, 12*, 2825–2830.

Reyna, M. A., et al. (2020). [Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019](https://doi.org/10.1097/CCM.0000000000004145). *Critical Care Medicine, 48*(2), 210–217.

Rubin, D. B. (1976). [Inference and missing data](https://doi.org/10.1093/biomet/63.3.581). *Biometrika, 63*(3), 581–592.

Seymour, C. W., et al. (2016). [Assessment of clinical criteria for sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)](https://doi.org/10.1001/jama.2016.0288). *JAMA, 315*(8), 762–774.

Singer, M., et al. (2016). [The third international consensus definitions for sepsis and septic shock (Sepsis-3)](https://doi.org/10.1001/jama.2016.0287). *JAMA, 315*(8), 801–810.

---

## AI Assistance

Throughout this project we used AI assistance to support parts of our development workflow. Rather than using it to generate results or analysis, we used it as a practical tool to move faster on tasks that were outside the core machine learning work.

**AI Used:** Claude

- **Git workflow automation:** We used Claude to help automate repetitive git tasks such as creating pull requests, writing commit messages, pushing branches, and editing commit history without an interactive editor.

- **Code debugging:** When we encountered errors in our code, we used Claude to help identify the root cause and understand what was going wrong before applying a fix.

**Example of specific prompts used:**


- "Our LSTM training loss is printing as NaN from the very first epoch and never recovers. The model is not learning anything. We checked the architecture and it looks fine. What in the input data could cause the loss to immediately become NaN and how do we track down where the NaN is coming from?"

- "The LSTM training is crashing with an out of memory error when we try to batch the sequences. Some patients have over 200 timesteps. How does sequence length affect memory during batching and what is the right way to handle patients with very long stays without losing their data?"

- " LSTM loss starts reasonable then suddenly rises to a huge number and never comes back down. Training is running but the model is not converging at all. What causes this kind of sudden loss explosion in recurrent networks and how do we stabilize it?"

- "The model finishes training with a low loss but when we evaluate it the recall for sepsis patients is zero. It is predicting no sepsis for every single patient. The loss looked fine during training so we did not catch it until evaluation. What is happening and how do we fix it?"
