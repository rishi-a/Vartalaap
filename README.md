

## Vartalaap: What Drives #AirQuality Discussions: Politics, Pollution or Pseudo-science?
**Paper: https://dl.acm.org/doi/abs/10.1145/3449170 . To appear in the The 24th ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW).
<hr>
This repository contains codes to reproduce the figures and results. Each folder represents a section in the paper. 

- **Dataset**: Represents **Section 3** of the paper. We are making the dataset public 
- **Research-Question-1** : Represents **Section 5.1 (Sentiment Analysis)** of the paper. The subdirectory contains further details.

- **Research-Question-2** : Represents **Section 5.2 (Granger Causality)** of the paper.
	- `figure-11.ipynb`: Python Notebook to generate Figure 11.
	- `figure-12-granger-causality-stationarity-test.ipynb`: Code to test stationarity of time-series.
	- `figure-12-granger-causality-test.ipynb`: Code to generate Figure 12.
	- `figure-13.ipynb`: Code to genrate Figure 13.
- **Evaluate-topic-model**:  Represents **Section 5.3 (Topic Model)** of the paper.
	- `figure-14.html`: Topic model visualisation. Topics are represented as numbers. 
	- `figure-15-topic-eval-over-time.ipynb`: Details the complete topic modelling analysis before settling for Figure 15 in the paper. 
	- `figure-16-coherence-score.ipynb`: Shows multiple analysis and multiple coherence plot, including the one shown in Figure 16.
	- `topic-modeling-df.py`: Python code to run LDA, save the visualisation as an HTML file and store the word compositions of each topic.

- **Research-Question-3**: Represents **Section 5.4 (Power Law)** of the paper.
	- `figure-17-power-tail.ipynb`: Code to generate Figure 17. 

- For citation purposes, please use the following Bib.
`
@article{adhikary2021vartalaap,
  title={Vartalaap: what drives\# airquality discussions: politics, pollution or pseudo-science?},
  author={Adhikary, Rishiraj and Patel, Zeel B and Srivastava, Tanmay and Batra, Nipun and Singh, Mayank and Bhatia, Udit and Guttikunda, Sarath},
  journal={Proceedings of the ACM on Human-Computer Interaction},
  volume={5},
  number={CSCW1},
  pages={1--29},
  year={2021},
  publisher={ACM New York, NY, USA}
}
`
