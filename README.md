
## CSCW2020 Submission | Paper #9746
### Anonymous Repository
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


