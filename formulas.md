# Hyperparameter tuning
To select the best parameters to be used in the final model I assign a score to each value in the "Parameters" column based on the following criteria:
* Calculate the frequency of each unique value in the "Parameters" column.
* Normalize the frequencies to a scale of 0 to 1, where 1 represents the highest frequency.
* Calculate the split weight for each value in the "Parameters" column, where a higher split number corresponds to a higher weight.
* Normalize the split weights to a scale of 0 to 1, where 1 represents the highest split weight.
* Calculate the RMSE weight for each value in the "Parameters" column, where a lower RMSE value corresponds to a higher weight.
* Normalize the RMSE weights to a scale of 0 to 1, where 1 represents the highest RMSE weight.

Then calculate the overall score for each value in the "Parameters" column by combining the normalized frequency, split weight, and RMSE weight and take into consideration the parameters that have the highest score.


# ❗Model accuracy

Since predicting the price accurately is very difficult let's se how good the model is at predicting whether the price will go up or down. 

For each row in our predictions let's consider the actual market-price, next-market-price and our predicted next-market-price (prediction).
We compute whether the current prediction is correct (1) or not (0):

$$ 
prediction\_is\_correct
= 
\begin{cases}
0 \text{ if [(market-price > next-market-price) and (market-price < prediction)] or [(market-price < next-market-price) and (market-price > prediction)]} \\
1 \text{ if [(market-price > next-market-price) and (market-price > prediction)] or [(market-price < next-market-price) and (market-price < prediction)]}
\end{cases}
$$

After that we count the number of correct prediction:
$$ 
correct\_predictions
= 
\sum_{i=0}^{total\_rows} prediction\_is\_correct
$$

Finally we compute the percentage of accuracy of the model:
$$
\\ 
accuracy 
= 
(correct\_predictions / total\_rows) 
* 100
$$