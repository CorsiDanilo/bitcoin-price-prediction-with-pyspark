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