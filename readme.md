# Experimental Details
These are the experimental settings for the paper "Model-Agnostic Causal Embedding Learning for Counterfactually Group-Fair Recommendation".

## Dataset Description
The experiments were conducted on the following publicly available recommendation benchmarks:

[$\texttt{MovieLens-1M}$](https://grouplens.org/datasets/movielens/1m/): It contains 1,000,209 user-system interactions from 6,040 users on 3,952 movies. 
We considered the user's occupation as the sensitive attribute, which contains 21 classes of occupations. For the user feedback, we treated the 5-star and 4-star ratings made by users as positive feedbacks (labeled with $1$), and others as negative feedbacks (labeled with $0$).
For each user, we took the last interaction for model testing. 

[$\texttt{Insurance}$](https://www.kaggle.com/mrmorj/insurance-recommendation): 
It is an insurance recommendation dataset on Kaggle, containing 5,382 interactions from 1,231 users on 21 insurances. We selected user's gender as the sensitive attribute that is a binary attribute, and split the dataset into a training set (80\%), a  validation set (10\%) and a testing set (10\%). 

[$\texttt{RentTheRunWay}$](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation): It contains 192,544 user-system interactions from 105,508 customers on 5,850 products, which were collected from a platform that allows women to rent clothes for various occasions. 
We selected user's age as the sensitive attribute. We assigned users into 12 equal-length groups according to the age range. We also split the dataset into a training set (80\%), a validation set (10\%) and a testing set (10\%). 

## Implementation Details
All the baselines and base models were trained on a single NVIDIA Tesla P100 GPU, with the batch size tuned among $\{64, 128, 256, 512, 1024\}$ and the learning rate tuned in $\{1\text{E-}1, 1\text{E-}2, 1\text{E-}3, 1\text{E-}4\}$.  
In the implementation of MACE, we set the networks $g_{\mathrm{exo}},$ $g_{\mathrm{endo}}$, $\mathrm{MLP}_1$ and $\mathrm{MLP}_2$ to 3-layer  fully connected neural networks respectively where the activation functions were $\mathrm{tanh}$ and $\mathrm{sigmoid}$. 
The dimensions of the exogenous part $g_{\mathrm{exo}}  (\mathbf{v}_u)$ and the endogenous part $g_{\mathrm{endo}}  (\mathbf{v}_u)$ were set to $d_{\mathrm{exo}} = d_{\mathrm{exo}} = 32$, and the dimensions of the user and item embeddings were all set to $128$. 
In the training process of MACE, the batch size $B$ and the maximum number of iteration $N$ were set to $128$ and $20\times \text{sample~size}/ 128$, respectively,  
the update cycle $\rho$ was set according to the sample size of training data ensuring that the MI minimization was executed $20$ times, 
the fair weight $\lambda$ in the final loss 
$$
\mathcal{L}(\bm{\theta}_{\mathrm{all}}) :=
\sum_{ (\mathbf{s}_u, \mathbf{x}_u, \mathbf{v}_i, y_{u, i}) \in \mathcal{D}_{\mathrm{train}} }  
\underbrace{\ell\left[~f (h_{\mathrm{re}}(\mathbf{v}_u), \mathbf{v}_i), y_{u, i} \right]}_{\text{Prediction Loss}} + 
\underbrace{\lambda\cdot \mathrm{ECGF} (f, h_{\mathrm{re}}, \mathcal{T})}_{\text{CGF-Oriented Constraint}} +
%\underbrace{\beta \mathrm{I} (g_{\mathrm{exo}}  (\mathbf{v}_u); \mathbf{s}_u) }_{\text{Exogeneity Construction}}+
\gamma \| \bm{\theta}_{\mathrm{all}} \|_2^2, 
$$
was tuned among $[0:+0.1:5]$, 
the regularization parameter $\gamma$ in the final loss was set to $0.001$, 
and the regularization parameter $\tau$ in the IV regression 
$$
\widehat{\mathbf{\Gamma}}
= \argmin_{\mathbf{\Gamma} \in \mathbb{R}^{d_{\mathrm{endo}} \times d_{\mathrm{exo}}}} \sum_{\mathbf{v} \in \mathcal{V}_{\mathrm{ue}} } \|g_{\mathrm{endo}} (\mathbf{v}) -  \mathbf{\Gamma} g_{\mathrm{exo}}  (\mathbf{v}) \|_2^2 + \tau \| \mathbf{\Gamma} \|_{\mathrm{F}}^2
= \mathbf{G}_{\mathrm{endo}}^{(B)} \mathbf{G}_{\mathrm{exo}}^{(B)^\intercal} \left(\mathbf{G}_{\mathrm{exo}}^{(B)} \mathbf{G}_{\mathrm{exo}}^{(B)^\intercal} + \tau \mathbf{I}_{d_{\mathrm{exo}}}\right)^{-1} ,
$$
was set to $0.9$. 

For the baselines Mixup and MACE-GapReg, we chose $\Delta$ DP as the fairness constraint. 
For fair comparisons, in the CGF-oriented constraint of MACE, we chose a weighted version of $\Delta$ DP as the GF metric in ECGF, where the set of perturbation parameters in ECGF was set to $\mathcal{T} = \{0, 0.2, 0.4, 0.6, 0.8, 1\}.$  