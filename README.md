## Winning Solution of PCIC 2021: Causal Inference and Recommendation([link](https://competition.huaweicloud.com/information/1000041488/introduction))
This repository provides our winning solution for the PCIC 2021: Causal Inference and Recommendation.

If you have any questions, please feel free to contact by issues or yitianartsky@gmail.com.

### Introduction
   *  We proposed and design a debias model which leverage the rating data for more accurate prediction.
   * The file "NNEnsembleSubmit.py" generates two part results:  one from the basic debias model framework, the other was achieved by weighted training based on inverse propensity score.
   *  Users can get the intermediate results of the debias model: the `dotProduct', `userBias', 'itemBias' through the file "dataPreprocessMovie.py".
   * For those data which include "userid, tagid, rating, aveMovieRate" we retrain the model, which generate more accurate prediction results due to richer features: the.
###  Reproduce result
    * run "python3 NNEnsembleSubmit.py" to generate data file "submit20210826.csv".
    * run "dataPreprocessMovie.py" to generate  files "testDt.csv, validDt.csv, totalDt.csv"
    * run "Rscript stackModeling.R" to generate final submit file "revision20210825.csv".
