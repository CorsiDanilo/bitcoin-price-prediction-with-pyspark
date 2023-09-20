# Hyperparameter tuning
To select the best parameters to be used in the final model I assign a score to each value in the "Parameters" column based on the following criteria:
* Calculate the frequency of each unique value in the "Parameters" column.
* Normalize the frequencies to a scale of 0 to 1, where 1 represents the highest frequency.
* Calculate the split weight for each value in the "Parameters" column, where a higher split number corresponds to a higher weight.
* Normalize the split weights to a scale of 0 to 1, where 1 represents the highest split weight.
* Calculate the RMSE weight for each value in the "Parameters" column, where a lower RMSE value corresponds to a higher weight.
* Normalize the RMSE weights to a scale of 0 to 1, where 1 represents the highest RMSE weight.
* Calculate the overall score for each value in the "Parameters" column by combining the normalized frequency, split weight, and RMSE weight.

I take into consideration the parameters that have the highest score.


# Accuracy
Finally let's se how good the model is at predicting whether the price will go up or down. 

For each row let's consider the current market-price, the next-market-price (the next day's market price), and our prediction (referred to next-market-price).
We compute whether the current prediction is correct or not this way:

$$ 
correct\_prediction
= 
\begin{cases}
0 \text{ if [(market-price > next-market-price) and (market-price < prediction)] or [(market-price < next-market-price) and (market-price > prediction)]} \\
1 \text{ if [(market-price > next-market-price) and (market-price > prediction)] or [(market-price < next-market-price) and (market-price < prediction)]}
\end{cases}
$$

This procedure is performed for the entire set under consideration, after that we calculate the percentage of accuracy of the model in this way:
$$
\\ 
accuracy 
= 
(correct\_predictions / total\_rows) 
* 100
$$

where:
* correct\_predictions contains the number of correct predictions (1).
* total\_rows contains the number of total predictions